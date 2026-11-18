"""
detector.py - Solar panel inspection via Roboflow REST API
==========================================================
Uses requests library directly. Works on Python 3.14+.

Classes detected:
  panel     : solar panel boundary / vague labels (ignored)
  clean     : no defect
  dusty     : dust/soiling
  bird_drop : bird droppings
  cracked   : physical damage

A main model decides dust vs clean; optional specialist models add bird
droppings and cracks (see TEAM_HELPERS). Power loss uses typical values per
condition (TYPICAL_EFF_LOSS).
"""

import cv2
import numpy as np
import base64
import requests
from dotenv import load_dotenv
import os

try:
    from . import panel_analysis
except ImportError:                      # run as a plain script/folder
    import panel_analysis

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
# Public Universe detection model (boxes around each problem), chosen by
# testing 5 candidate models on labelled photos:
#   https://universe.roboflow.com/solar-panel-damage-detectionsegmentation/solar-panel-o6dwf
#   classes: Bird-drop, Dusty, Physical-Damage, Electrical-Damage,
#            Non-Defective, Defective (generic, ignored), Snow (ignored)
# Not used: "solar-panel-classification-ndfur/2" returns ~0.01 % for every
# class on its own training photos (broken), and the earlier
# "trial-lrosz/..." trial model isn't public.
ROBOFLOW_MODEL_ID   = "solar-panel-o6dwf/3"
OLD_MODEL_IDS = ("solar-panel-classification-ndfur/2", "trial-lrosz/")
CONFIDENCE_THRESHOLD = 0.5

# Smart mode: three more hosted models run in parallel with the main one,
# then panel_analysis finds the panel outline and judges what's on it.
#   bird model    : solar-panel-detector-b9iyt/1 (bird drops; whole-panel
#                   "bird" boxes are rejected - they mean heavy dust)
#   crack model   : solar-panel-w3f7s/7 (cracks/damage + a whole-panel box)
#   outline model : photovoltaic-0enc4/9 (exact panel polygon)
# Result on one owner-labelled Tello flight: 12/12 photos incl. severity
# (main model alone: 2 of 6 panels right). Public close-ups: 7/13.
MAIN_MODEL_NAME = "main model"
BIRD_MODEL_NAME = "bird model"
CRACK_MODEL_NAME = "crack model"
OUTLINE_MODEL_NAME = "outline model"
TEAM_HELPERS = (
    ("solar-panel-detector-b9iyt/1", BIRD_MODEL_NAME),
    ("solar-panel-w3f7s/7", CRACK_MODEL_NAME),
    ("photovoltaic-0enc4/9", OUTLINE_MODEL_NAME),
)

# Normalises class names from different Roboflow models to the internal ones
CLASS_ALIASES = {
    'panel': 'panel', 'solar-panel': 'panel',
    'clean': 'clean',
    'dusty': 'dusty', 'dust': 'dusty', 'dirty': 'dusty', 'soiled': 'dusty',
    'bird-drop': 'bird_drop', 'bird_drop': 'bird_drop', 'birddrop': 'bird_drop',
    'bird-droppings': 'bird_drop',
    'cracked': 'cracked', 'physical-damage': 'cracked', 'damaged': 'cracked',
    'damage': 'cracked', 'broken': 'cracked', 'physical-damge': 'cracked',
    'electrical-damage': 'cracked', 'electrical-damge': 'cracked',
    'non-defective': 'clean', 'non_defective': 'clean',
    'bird-drop-': 'bird_drop', 'electrical': 'cracked', 'mechanical': 'cracked',
    # too vague / not relevant in India: treated like the panel outline
    'defective': 'panel', 'snow': 'panel',
}

# Typical power lost per condition (field studies, rounded):
#   dusty     15 %  dry-season soiling after 2–4 weeks (light 2–5 %, heavy 30 %)
#   bird_drop 33 %  one covered cell pushes 1/3 of the panel onto its bypass diode
#   cracked   25 %  visible cracks; grows over time and is a safety risk
# The photos can't measure "how much" reliably (reflections, angle), so the
# same honest typical value is used for every panel with that condition.
TYPICAL_EFF_LOSS = {'dusty': 0.15, 'bird_drop': 0.33, 'cracked': 0.25}
CLASSIFICATION_EFF_LOSS = TYPICAL_EFF_LOSS


def inspection_area(w, h):
    """The central box the pilot frames each panel in (matches the white
    guide box drawn on the app's live video)."""
    return w / 6, h / 6, w * 5 / 6, h * 5 / 6


def in_inspection_area(p, w, h):
    """True when the prediction box's centre lies inside the checked area.
    (A "half the box inside" rule was tried and scored worse: 7/13 vs 8/13
    on the labelled test photos, with no gain on real flights.)"""
    x0, y0, x1, y1 = inspection_area(w, h)
    return x0 <= p['x'] <= x1 and y0 <= p['y'] <= y1


def normalise_class(name):
    key = str(name).strip().lower().replace(' ', '-')
    return CLASS_ALIASES.get(key, CLASS_ALIASES.get(key.replace('-', '_'), key))


class PanelDetector:

    SEVERITY_RANK = {
        'panel':     0,
        'clean':     0,
        'dusty':     2,
        'bird_drop': 3,
        'cracked':   5,
    }

    def __init__(
        self,
        api_key    = ROBOFLOW_API_KEY,
        model_id   = ROBOFLOW_MODEL_ID,
        confidence = CONFIDENCE_THRESHOLD,
        team       = False,
    ):
        self.api_key    = api_key
        self.model_id   = model_id
        self.confidence = confidence
        self.api_url    = f"https://serverless.roboflow.com/{model_id}"
        self.team       = TEAM_HELPERS if team else ()

        print(f"[DETECTOR] Roboflow REST API ready")
        print(f"[DETECTOR] Endpoint: {self.api_url}")
        if self.team:
            print("[DETECTOR] Smart analysis with: " +
                  ", ".join(f"{m} ({n})" for m, n in self.team))

    def _call_model(self, model_id, encoded):
        """POST one image to a hosted model; returns its predictions list."""
        response = requests.post(
            f"https://serverless.roboflow.com/{model_id}",
            params  = {"api_key": self.api_key},
            data    = encoded,
            headers = {"Content-Type": "application/x-www-form-urlencoded"},
            timeout = 30,
        )
        if response.status_code != 200:
            raise ConnectionError(
                f"API returned {response.status_code}: {response.text[:120]}")
        return response.json().get('predictions', [])

    def detect(self, frame):
        """Main model only, or (smart mode) four models + panel analysis."""
        result = self._detect_main(frame)
        if not self.team or result.get('error') or result.get('classification'):
            return result
        for p in result.get('raw_predictions') or []:
            if isinstance(p, dict):
                p.setdefault('source', MAIN_MODEL_NAME)

        from concurrent.futures import ThreadPoolExecutor
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        encoded = base64.b64encode(buffer).decode('utf-8')
        with ThreadPoolExecutor(len(self.team)) as pool:
            futures = [(pool.submit(self._call_model, m, encoded), name)
                       for m, name in self.team]

        helper, team_errors = {}, []
        for fut, name in futures:
            try:
                preds = fut.result()
            except Exception as exc:
                team_errors.append(f"{name}: {exc}")
                print(f"[DETECTOR] {name} failed: {exc}")
                preds = []
            for p in preds:
                p['class_raw'] = p.get('class', '')
                p['class'] = normalise_class(p['class_raw'])
                p['source'] = name
            helper[name] = preds

        # AI evidence from the three condition models ("snow" on an Indian
        # rooftop = a compact white blob, i.e. a possible bird dropping)
        ai = [p for p in result['raw_predictions'] if isinstance(p, dict) and 'x' in p]
        ai += helper.get(BIRD_MODEL_NAME, []) + helper.get(CRACK_MODEL_NAME, [])
        for p in ai:
            if 'snow' in str(p.get('class_raw', '')).lower():
                p['class'] = 'white_blob'

        mask, outline_src = panel_analysis.panel_mask(
            frame, helper.get(OUTLINE_MODEL_NAME, []),
            helper.get(CRACK_MODEL_NAME, []))
        m = panel_analysis.measure(frame, mask)
        dec = panel_analysis.decide(frame, mask, ai, m)
        for q in dec['evidence']:
            q['used'] = True

        cond = dec['condition']
        detections = []
        for q in dec['evidence']:
            detections.append({
                'class':            'bird_drop' if q['class'] == 'white_blob' else q['class'],
                'confidence':       round(q['confidence'], 3),
                'bbox':             [int(q['x'] - q['width'] / 2),
                                     int(q['y'] - q['height'] / 2),
                                     int(q['width']), int(q['height'])],
                'severity':         self.SEVERITY_RANK.get(cond, 0),
                'soiling_coverage': 0.0,
                'efficiency_loss':  dec['loss'],
                'source':           q.get('source', ''),
            })
        result.update({
            'annotated_frame': panel_analysis.draw(frame, mask, m, dec),
            'detections':      detections,
            'worst_class':     cond,
            'worst_severity':  self.SEVERITY_RANK.get(cond, 0),
            'worst_eff_loss':  dec['loss'],
            'worst_coverage':  0.0,
            'level':           dec['level'],
            'dirt_share':      round(float(m['dirt_share']), 3),
            'specks':          round(float(m['specks']), 1),
            'panel_source':    outline_src,
            'decided_by':      dec['by'],
            'why':             dec['why'],
            'decision_confidence': dec['confidence'],
            'mixed':           bool(dec.get('mixed')),
            'loss_range':      tuple(dec.get('loss_range') or (dec['loss'], dec['loss'])),
            'loss_note':       dec.get('loss_note', ''),
            'raw_predictions': ai + helper.get(OUTLINE_MODEL_NAME, []),
            'team_errors':     team_errors,
            'smart':           True,
            'no_coverage':     True,
        })
        return result

    # ── Option A: pixel-level soiling coverage ───────────────────────────────

    def calculate_soiling_coverage(self, frame, bbox):
        """
        Estimates what fraction of the panel surface appears soiled
        using HSV colour analysis.

        A clean solar panel is dark blue/grey. Dust, bird droppings,
        and debris shift the colour toward brown, tan, white, or grey.
        This method measures how many pixels inside the detected
        bounding box fall outside the clean panel colour range.

        Args:
            frame : full BGR frame from camera
            bbox  : [x, y, w, h] bounding box of detected region

        Returns:
            float 0.0 to 1.0 — fraction of region that appears soiled
        """
        x, y, w, h = bbox

        # Guard against zero-size or out-of-bounds boxes
        if w <= 0 or h <= 0:
            return 0.0

        frame_h, frame_w = frame.shape[:2]
        x  = max(0, x)
        y  = max(0, y)
        w  = min(w, frame_w - x)
        h  = min(h, frame_h - y)

        if w <= 0 or h <= 0:
            return 0.0

        # Crop to the detected region
        region = frame[y:y+h, x:x+w]
        hsv    = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Clean solar panel colour range in HSV:
        #   Hue 100-130   (blue-grey)
        #   Saturation 15-80 (muted, not vivid)
        #   Value 20-80   (dark to mid-tone, not bright)
        lower_clean = np.array([100, 15, 20])
        upper_clean = np.array([130, 80, 80])

        clean_mask   = cv2.inRange(hsv, lower_clean, upper_clean)
        clean_pixels = cv2.countNonZero(clean_mask)
        total_pixels = w * h

        soiling_ratio = 1.0 - (clean_pixels / total_pixels)

        # Clamp to valid range
        return float(np.clip(soiling_ratio, 0.0, 1.0))

    def soiling_ratio_to_efficiency_loss(self, ratio, severity):
        """
        Converts pixel coverage ratio to efficiency loss fraction.

        Uses published PV soiling research curves:
          Goossens & Van Kerschaever (1999)
          Sayyah, Boehm & Ghosh (2014)

        For cracked panels, pixel coverage is less meaningful —
        a small crack causes outsized electrical damage via hotspot
        effect. We apply a minimum floor for cracks.

        Args:
            ratio    : float 0.0-1.0 from calculate_soiling_coverage
            severity : int severity rank of detected class

        Returns:
            float 0.0-1.0 efficiency loss
        """
        if severity == 5:
            # Cracked: pixel coverage understates damage
            # Even small crack = significant efficiency loss
            # Floor at 40%, scale up with coverage
            return float(np.clip(0.40 + ratio * 0.30, 0.40, 0.70))

        if severity == 3:
            # Bird dropping: discrete hotspot effect
            # Even small coverage causes string-level loss
            # Floor at 20%, scale with coverage
            return float(np.clip(0.20 + ratio * 0.30, 0.20, 0.50))

        if severity == 2:
            # Dusty: linear-ish curve from research
            # 0-5%   coverage → 1-3% loss
            # 5-20%  coverage → 3-10% loss
            # 20-50% coverage → 10-25% loss
            # 50%+   coverage → 25-40% loss
            if ratio < 0.05:
                return ratio * 0.40
            elif ratio < 0.20:
                return 0.02 + (ratio - 0.05) * 0.47
            elif ratio < 0.50:
                return 0.09 + (ratio - 0.20) * 0.53
            else:
                return float(np.clip(0.25 + (ratio - 0.50) * 0.30, 0.0, 0.40))

        # Panel boundary or unknown — no loss
        return 0.0

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_boxes(self, frame, predictions):
        annotated = frame.copy()

        colours = {
            'panel':     (200, 200, 200),
            'clean':     (60,  180, 60),
            'dusty':     (0,   165, 255),
            'bird_drop': (0,   0,   255),
            'cracked':   (0,   0,   180),
        }

        for pred in predictions:
            cls  = pred['class']
            conf = pred['confidence']
            if cls == 'panel':          # generic "defective"/outline boxes
                continue
            x    = int(pred['x'] - pred['width']  / 2)
            y    = int(pred['y'] - pred['height'] / 2)
            w    = int(pred['width'])
            h    = int(pred['height'])

            colour = colours.get(cls, (255, 255, 0))
            cv2.rectangle(annotated, (x, y), (x+w, y+h), colour, 2)

            # Include coverage % in label if available
            # "AI is N % sure" – the only number the model really gives
            name = {'clean': 'no defect', 'bird_drop': 'bird dropping',
                    'cracked': 'damage'}.get(cls, cls)
            label = f"{name} {conf:.0%} sure"

            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            cv2.rectangle(
                annotated,
                (x, y - lh - 6), (x + lw + 4, y),
                colour, -1
            )
            cv2.putText(
                annotated, label,
                (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 1
            )

        return annotated

    # ── Classification models (one label set for the whole photo) ────────────

    @staticmethod
    def _is_classification(raw, preds):
        if 'predicted_classes' in raw or 'top' in raw:
            return True
        if isinstance(preds, dict):
            return True
        return bool(preds) and isinstance(preds[0], dict) and 'x' not in preds[0]

    def _classification_result(self, frame, raw, preds):
        """
        Handles Roboflow classification responses:
          multi-label : {"predictions": {"Dusty": {"confidence": 0.9}, ...},
                         "predicted_classes": ["Dusty"]}
          single-label: {"predictions": [{"class": "Dusty", "confidence": 0.9}],
                         "top": "Dusty", "confidence": 0.9}
        The whole photo is treated as one panel.
        """
        scores = {}
        if isinstance(preds, dict):
            for name, val in preds.items():
                conf = val.get('confidence', 0) if isinstance(val, dict) else val
                scores[name] = float(conf)
        else:
            for p in preds:
                scores[p.get('class', '?')] = float(p.get('confidence', 0))

        if 'predicted_classes' in raw:
            chosen = list(raw['predicted_classes'])
        else:
            chosen = [n for n, c in scores.items() if c >= self.confidence]
            if not chosen and raw.get('top'):
                chosen = [raw['top']]

        h, w = frame.shape[:2]
        detections = []
        for name in chosen:
            cls = normalise_class(name)
            if cls not in self.SEVERITY_RANK:
                print(f"[DETECTOR] Unknown class '{name}' – ignored")
                continue
            if cls in ('clean', 'panel'):
                continue
            detections.append({
                'class':            cls,
                'confidence':       round(scores.get(name, 0.0), 3),
                'bbox':             [0, 0, w, h],
                'severity':         self.SEVERITY_RANK[cls],
                'soiling_coverage': 0.0,
                'efficiency_loss':  CLASSIFICATION_EFF_LOSS.get(cls, 0.0),
            })

        if detections:
            worst = max(detections, key=lambda d: d['severity'])
            worst_class, worst_sev = worst['class'], worst['severity']
            worst_eff = worst['efficiency_loss']
        else:
            worst_class, worst_sev, worst_eff = 'clean', 0, 0.0

        # Banner with the labels instead of boxes
        annotated = frame.copy()
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        label = "  |  ".join(f"{n} {c:.0%}" for n, c in ranked[:4]) or "no result"
        colour = {
            'clean': (60, 180, 60), 'dusty': (0, 165, 255),
            'bird_drop': (0, 0, 255), 'cracked': (0, 0, 180),
        }.get(worst_class, (200, 200, 200))
        bar_h = max(30, h // 14)
        cv2.rectangle(annotated, (0, 0), (w, bar_h), colour, -1)
        cv2.putText(annotated, f"{worst_class.upper()}   ({label})",
                    (10, int(bar_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
                    bar_h / 50, (255, 255, 255), max(1, bar_h // 25),
                    cv2.LINE_AA)
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), colour, 4)

        return {
            'annotated_frame': annotated,
            'detections':      detections,
            'worst_class':     worst_class,
            'worst_severity':  worst_sev,
            'worst_coverage':  0.0,
            'worst_eff_loss':  worst_eff,
            'panel_count':     1,
            'classification':  True,
            'scores':          {n: round(c, 3) for n, c in scores.items()},
            'raw_predictions': preds,
        }

    # ── Main detection ────────────────────────────────────────────────────────

    def _detect_main(self, frame):
        """
        Sends frame to the main Roboflow model and returns a structured
        result (detect() adds the specialist team on top of this).

        Returns dict:
        {
            'annotated_frame'  : frame with boxes + coverage labels,
            'detections'       : list of detection dicts,
            'worst_class'      : most severe condition name,
            'worst_severity'   : severity score 0-5,
            'worst_coverage'   : pixel coverage ratio of worst detection,
            'worst_eff_loss'   : efficiency loss fraction (Option A),
            'panel_count'      : number of panel boundaries detected,
            'raw_predictions'  : raw Roboflow response
        }
        """
        try:
            _, buffer = cv2.imencode(
                '.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            encoded = base64.b64encode(buffer).decode('utf-8')

            response = requests.post(
                self.api_url,
                params  = {"api_key": self.api_key},
                data    = encoded,
                headers = {"Content-Type": "application/x-www-form-urlencoded"},
                timeout = 30   # serverless models can take ~10 s to wake
            )

            if response.status_code != 200:
                raise ConnectionError(
                    f"API returned {response.status_code}: {response.text}"
                )

            raw   = response.json()
            preds = raw.get('predictions', [])

            if self._is_classification(raw, preds):
                return self._classification_result(frame, raw, preds)

            # Detection models: map class names to the internal ones
            for p in preds:
                p['class_raw'] = p.get('class', '')
                p['class'] = normalise_class(p['class_raw'])

        except Exception as e:
            print(f"[DETECTOR] API call failed: {e}")
            return {
                'annotated_frame': frame,
                'detections':      [],
                'worst_class':     'unknown',
                'worst_severity':  -1,
                'worst_coverage':  0.0,
                'worst_eff_loss':  0.0,
                'panel_count':     0,
                'error':           str(e)
            }

        # Keep confident boxes whose centre is inside the area the pilot
        # framed the panel in (the app's white guide box). Boxes on the
        # background (roof, ground, a laptop sleeve...) are not the panel.
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = inspection_area(w, h)
        filtered = [
            p for p in preds
            if p['confidence'] >= self.confidence
            and in_inspection_area(p, w, h)
        ]
        ignored_outside = sum(
            1 for p in preds if p['confidence'] >= self.confidence
            and not in_inspection_area(p, w, h))

        # NOTE: the old pixel-colour "soiling coverage" estimate was removed.
        # It counted every non-dark-blue pixel (reflections, frame, sky) as
        # dirt, so it read ~100 % on almost any real photo and pushed every
        # loss to its cap. Losses now use typical values per condition.
        detections = []
        for p in filtered:
            cls      = p['class']
            severity = self.SEVERITY_RANK.get(cls, 0)

            bbox = [
                int(p['x'] - p['width']  / 2),
                int(p['y'] - p['height'] / 2),
                int(p['width']),
                int(p['height'])
            ]
            p['soiling_coverage'] = None

            detections.append({
                'class':            cls,
                'confidence':       round(p['confidence'], 3),
                'bbox':             bbox,
                'severity':         severity,
                'soiling_coverage': 0.0,
                'efficiency_loss':  TYPICAL_EFF_LOSS.get(cls, 0.0),
            })

        # Determine worst condition (ignore panel boundary class)
        condition_detections = [
            d for d in detections if d['class'] not in ('panel', 'clean')
        ]

        if condition_detections:
            # Most confident problem wins (tested better than "most severe",
            # which let weak, wrong 'damage' guesses override clear dust).
            worst         = max(condition_detections,
                                key=lambda d: (d['confidence'], d['severity']))
            worst_class   = worst['class']
            worst_sev     = worst['severity']
            worst_cov     = worst['soiling_coverage']
            worst_eff     = worst['efficiency_loss']
        elif detections:
            worst_class   = 'clean'
            worst_sev     = 0
            worst_cov     = 0.0
            worst_eff     = 0.0
        else:
            worst_class   = 'clean'
            worst_sev     = 0
            worst_cov     = 0.0
            worst_eff     = 0.0

        panel_count = sum(1 for d in detections if d['class'] == 'panel')
        annotated   = self._draw_boxes(frame, filtered)
        # Show which area was judged
        cv2.rectangle(annotated, (int(x0), int(y0)), (int(x1), int(y1)),
                      (255, 255, 255), 1)
        cv2.putText(annotated, "area checked", (int(x0) + 4, int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)

        return {
            'annotated_frame': annotated,
            'detections':      detections,
            'worst_class':     worst_class,
            'worst_severity':  worst_sev,
            'worst_coverage':  worst_cov,
            'worst_eff_loss':  worst_eff,
            'panel_count':     panel_count,
            'no_coverage':     True,      # loss is a typical value, not measured
            'ignored_outside': ignored_outside,
            'raw_predictions': preds
        }