"""
panel_analysis.py - find the solar panel in a photo and judge what's on it
==========================================================================
Combines AI boxes with pixel measurements made ONLY inside the panel:

1. Panel outline
   - "photovoltaic" outline model polygon (exact shape), else
   - the crack model's whole-panel box, trimmed to rows/columns that are
     mostly panel cells, else the app's centre guide box (also trimmed).

2. Measurements inside the outline (grid lines and module frames removed)
   - dirt share : pale, colourless pixels (dust, grime, droppings). Works for
                  blue AND black panels: clean cells are either strongly
                  coloured or dark.
   - white blobs: compact bright spots (possible bird droppings)
   - specks     : density of small bright dots (heavy speckled dust).
                  Calibrated on Tello 960x720 video frames only.

3. Decision (cracks > bird dropping > dust > clean)
   - cracked   : crack model box inside the panel (>= 50 %)
   - bird_drop : an AI "bird / white blob(snow)" box that is SMALL compared
                 with the panel, backed by a white blob in the pixels (or AI
                 >= 70 % sure), on a panel that isn't grimy all over.
                 (A whole-panel "bird drop" box means heavy dust, not a bird.)
   - dusty     : heavy / moderate / light from dirt share and specks, or any
                 AI "dust" box inside the panel (at least light)
   - clean     : none of the above

Calibration: one real Tello flight (6 panels, 12 photos, owner-labelled:
2 light dust, 2 heavy dust, 1 bird dropping, 1 clean) -> 12/12; 13 public
labelled close-ups (different source) -> 7/13. Light vs clean is the
thinnest margin (clean measured ~10 % dirt share, light dust 12-14 %).
"""
import cv2
import numpy as np

# Power loss by finding (rounded field values; see detector.TYPICAL_EFF_LOSS)
DUST_LOSS = {"light": 0.04, "moderate": 0.12, "heavy": 0.25}
CRACK_LOSS = 0.25

LIGHT_DIRT, MODERATE_DIRT, HEAVY_DIRT = 0.12, 0.20, 0.35
MODERATE_SPECKS, HEAVY_SPECKS = 9, 15          # per 10k panel pixels (Tello frames)
BIRD_MAX_SHARE = 0.15                            # bird box must be small vs panel
CONF = 0.5


def _cell_like(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:, :, 1] > 70) | (hsv[:, :, 2] < 90)).astype(np.uint8) * 255


def panel_mask(img, outline_preds, panel_box_preds):
    """Returns (mask uint8 0/255, source description)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    pv = [q for q in outline_preds if q.get("confidence", 0) >= CONF and q.get("points")]
    if pv:
        q = max(pv, key=lambda q: q["width"] * q["height"])
        pts = np.array([[p["x"], p["y"]] for p in q["points"]], np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask, "panel outline found by the outline model"

    box = np.zeros((h, w), np.uint8)
    cands = [q for q in panel_box_preds if q.get("confidence", 0) >= CONF
             and str(q.get("class_raw", q.get("class", ""))).lower()
             in ("defective", "non defective", "non-defective", "dust",
                 "physical damage")]
    if cands:
        q = max(cands, key=lambda q: q["width"] * q["height"])
        x0, y0 = int(q["x"] - q["width"] / 2), int(q["y"] - q["height"] / 2)
        cv2.rectangle(box, (max(0, x0), max(0, y0)),
                      (min(w - 1, int(x0 + q["width"])), min(h - 1, int(y0 + q["height"]))),
                      255, -1)
        src = "panel box from the crack model, trimmed to the cells"
    else:
        cv2.rectangle(box, (w // 6, h // 6), (w * 5 // 6, h * 5 // 6), 255, -1)
        src = "centre guide box, trimmed to the cells"

    ys, xs = np.where(box > 0)
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    cells = cv2.morphologyEx(_cell_like(img), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    sub = cells[by0:by1 + 1, bx0:bx1 + 1] > 0

    def band(profile):
        best, cur, start, span = 0, 0, 0, (0, len(profile) - 1)
        for i, good in enumerate(profile > 0.6):
            cur = cur + 1 if good else 0
            if cur == 1:
                start = i
            if cur > best:
                best, span = cur, (start, i)
        return span

    r0, r1 = band(sub.mean(axis=1))
    c0, c1 = band(sub[r0:r1 + 1].mean(axis=0))
    cv2.rectangle(mask, (bx0 + c0, by0 + r0), (bx0 + c1, by0 + r1), 255, -1)
    return mask, src


def _line_mask(binary, diag):
    """Long straight structures at any angle (frames, rails, thick grid)."""
    length = max(15, int(0.22 * diag))
    out = np.zeros_like(binary)
    c = length // 2
    for ang in range(0, 180, 12):
        k = np.zeros((length, length), np.uint8)
        dx, dy = np.cos(np.radians(ang)) * c, np.sin(np.radians(ang)) * c
        cv2.line(k, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 1, 1)
        out |= cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    return cv2.dilate(out, np.ones((7, 7), np.uint8))


def measure(img, mask):
    """Dirt share, white blobs and speck density inside the panel."""
    inside = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0
    n = max(int(inside.sum()), 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = cv2.medianBlur(hsv[:, :, 1], 11)
    v = cv2.medianBlur(hsv[:, :, 2], 11)
    pale = ((s < 70) & (v > 105)).astype(np.uint8)
    ys, xs = np.where(inside)
    diag = float(np.hypot(np.ptp(xs), np.ptp(ys))) if len(xs) else 1.0
    k = max(5, int(0.035 * diag))
    bulk = cv2.morphologyEx(pale, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) > 0
    lines = (_line_mask(pale, diag) > 0) & ~bulk        # long AND thin = frame/grid
    dirt = (pale > 0) & ~lines & inside

    inner = cv2.erode(mask, np.ones((25, 25), np.uint8)) > 0
    white = ((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 175) & inside).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnt, lab, st, cents = cv2.connectedComponentsWithStats(white, 8)
    blobs = []
    for i in range(1, cnt):
        a, bw, bh = st[i, cv2.CC_STAT_AREA], st[i, 2], st[i, 3]
        if (a >= 0.004 * n and max(bw, bh) <= 3.5 * min(bw, bh)
                and a / max(bw * bh, 1) > 0.35):
            # solidity: area / convex-hull area. Droppings are irregular
            # splats (~0.6-0.7); neat shapes (e.g. the white diamonds between
            # black mono cells, reflections) are ~0.9+.
            comp = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea)))
            # a real dropping sits ON the panel, surrounded by cells; pale
            # patches touching the outline are usually edge reflections
            ring = cv2.dilate(comp, np.ones((15, 15), np.uint8)) > 0
            blobs.append({"share": a / n, "x": float(cents[i][0]),
                          "y": float(cents[i][1]),
                          "r": float(max(bw, bh)) / 2,
                          "solidity": float(a / max(hull, 1.0)),
                          "on_edge": bool((ring & ~inner).any())})

    # Specks = small bright dots NOT on the grid (grid crossings blur into
    # bright dots in video and must not count as dust)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thin bright structures of any colour (white or blue-white grid lines)
    thin_bright = (cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                                    np.ones((15, 15), np.uint8)) > 25).astype(np.uint8)
    grid = _line_mask(thin_bright, diag) > 0
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((7, 7), np.uint8))
    cnt2, _, st2, c2 = cv2.connectedComponentsWithStats(
        ((th > 35) & inside).astype(np.uint8), 8)
    speck_pts = [(float(c2[i][0]), float(c2[i][1])) for i in range(1, cnt2)
                 if 2 <= st2[i, 4] <= 60 and not grid[int(c2[i][1]), int(c2[i][0])]]
    specks = len(speck_pts) / (n / 10000)
    return {"dirt_share": dirt.sum() / n, "blobs": blobs, "specks": specks,
            "speck_pts": speck_pts, "groups_hit": _groups_hit(mask, speck_pts, blobs),
            "dirt_mask": dirt, "panel_share": mask.mean() / 255}


def _groups_hit(mask, pts, blobs):
    """How many of the panel's 3 cell groups have spots. The groups run as
    three bands side by side across the panel's short side (standard 60/72
    cell layout), so spots are binned into thirds along that axis."""
    pts = list(pts) + [(b["x"], b["y"]) for b in blobs]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or not pts:
        return 0
    (cx, cy), (w, h), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    theta = np.radians(ang)
    # unit vector along the rectangle's SHORT side
    ux, uy = (np.cos(theta), np.sin(theta)) if w < h else (-np.sin(theta), np.cos(theta))
    short = max(min(w, h), 1.0)
    counts = [0, 0, 0]
    for x, y in pts:
        t = ((x - cx) * ux + (y - cy) * uy) / short + 0.5      # 0..1 across
        if 0 <= t <= 1:
            counts[min(2, int(t * 3))] += 1
    need = max(2, int(0.1 * len(pts)))       # a few spots, not one stray
    return sum(1 for c in counts if c >= need)


# ── Power-loss model ─────────────────────────────────────────────────────────
# A typical panel has ~60 cells wired as 3 groups ("substrings"), each with a
# bypass diode. An opaque dropping covering about half a cell or more makes
# its whole group bypass -> ~1/3 of the panel lost. Smaller ones dim the group.
# Many droppings spread over the panel hit 2 or all 3 groups. Exact dropping
# counts aren't reliable from drone video (reflections/texture also make
# spots: a clean panel gave 14-28 "spots"), so spread is judged from the
# speck density, and every estimate carries a likely range.
CELLS_PER_PANEL = 60
CELL_SHARE = 1.0 / CELLS_PER_PANEL
GROUPS = 3
GROUP_LOSS_WHEN_HIT = 0.6     # average loss of a group hit by several droppings


def single_drop_loss(share):
    """One dropping covering `share` of the panel (0 = size unknown)."""
    s = 0.5 if share <= 0 else min(1.0, share / (0.5 * CELL_SHARE))
    loss = max(0.08, s / GROUPS)
    cells = share / CELL_SHARE if share > 0 else None
    note = (f"covers about {cells:.1f} cell{'s' if cells and cells >= 1.5 else ''}"
            if cells else "size unclear")
    if s >= 1.0:
        note += " – enough to switch off its whole group of cells (~1/3 of the panel)"
    elif s >= 0.8:
        note += (" – close to switching off its whole group of cells "
                 "(~1/3 of the panel)")
    else:
        note += " – dims its group of cells (about a third of the panel)"
    return loss, (round(loss * 0.7, 3), round(min(0.34, loss * 1.3), 3)), note


def many_drop_loss(groups_hit):
    groups = min(3, max(2, groups_hit))      # "many" means at least 2 hit
    loss = groups / GROUPS * GROUP_LOSS_WHEN_HIT
    rng = (0.45, 0.75) if groups == 3 else (0.30, 0.55)
    note = (f"spots found in {groups} of the panel's {GROUPS} cell groups "
            f"(the three bands across the panel) – each hit group loses much "
            f"of its power")
    return loss, rng, note


def dust_loss(level, dirt_share):
    floor = {"light": 0.03, "moderate": 0.08, "heavy": 0.18}[level]
    loss = min(0.35, max(floor, 0.02 + 0.30 * dirt_share))
    return loss, (round(loss * 0.6, 3), round(min(0.40, loss * 1.5), 3))


def _inside(mask, q):
    x, y = int(q["x"]), int(q["y"])
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0


def decide(img, mask, ai_preds, m):
    """ai_preds: dicts with class (internal name), confidence, x, y, width,
    height, source. Returns a decision dict."""
    panel_area = max(int((mask > 0).sum()), 1)
    tello = img.shape[:2] == (720, 960)
    inside = [q for q in ai_preds if q.get("confidence", 0) >= CONF and _inside(mask, q)]
    cracks = [q for q in inside if q["class"] == "cracked"
              and q.get("source") == "crack model"]
    birds = [q for q in inside if q["class"] in ("bird_drop", "white_blob")
             and q["width"] * q["height"] / panel_area <= BIRD_MAX_SHARE]
    dust_ai = [q for q in inside if q["class"] == "dusty"]
    small_blobs = [b for b in m["blobs"] if b["share"] < BIRD_MAX_SHARE]
    blob_share = sum(b["share"] for b in small_blobs)
    rest = max(m["dirt_share"] - blob_share, 0.0)
    specks = m["specks"] if tello else 0.0
    facts = (f"{m['dirt_share']:.0%} of the panel looks dirty"
             + (f", speck density {m['specks']:.0f}" if tello else ""))

    if cracks:
        top = max(cracks, key=lambda q: q["confidence"])
        return {"condition": "cracked", "level": "", "loss": CRACK_LOSS,
                "loss_range": (0.05, 0.50),
                "loss_note": "cracks start small but grow and let water in",
                "confidence": top["confidence"], "by": top["source"],
                "evidence": [top], "why": "the crack model marked damage on the panel"}
    # spots all over (with an AI-confirmed dropping, a moderate count is enough)
    covered = specks >= MODERATE_SPECKS or rest >= 0.20
    ai_bird = bool(birds) and (small_blobs or max(q["confidence"] for q in birds) >= 0.7)
    if ai_bird and covered:
        top = max(birds, key=lambda q: q["confidence"])
        loss, rng, note = many_drop_loss(m["groups_hit"])
        return {"condition": "bird_drop", "level": "many", "loss": loss,
                "loss_range": rng, "loss_note": note,
                "confidence": top["confidence"], "by": top["source"],
                "evidence": birds,
                "why": f"spots all over the panel, and an AI model ({top['source']}) "
                       f"confirmed at least one is a bird dropping "
                       f"({top['confidence']:.0%} sure)"}
    if ai_bird and rest < 0.30:
        top = max(birds, key=lambda q: q["confidence"])
        loss, rng, note = single_drop_loss(blob_share if small_blobs else 0.0)
        size = (f"white spot covering {blob_share:.1%} of the panel"
                if small_blobs else "small spot")
        return {"condition": "bird_drop", "level": "single", "loss": loss,
                "loss_range": rng, "loss_note": note,
                "confidence": top["confidence"], "by": top["source"],
                "evidence": birds,
                "why": f"an AI model ({top['source']}) marked a small white spot "
                       f"({top['confidence']:.0%} sure) and the photo shows a "
                       f"matching {size}"}
    # A dropping the AI missed: 1-3 irregular white splats on an otherwise
    # clean panel (neat shapes like mono-cell diamonds are excluded)
    splats = [b for b in small_blobs if b.get("solidity", 1) < 0.88
              and 0.003 <= b["share"] <= 0.05 and not b.get("on_edge")]
    if splats and len(small_blobs) <= 3 and rest < 0.10 and specks < MODERATE_SPECKS:
        share = sum(b["share"] for b in splats)
        loss, rng, note = single_drop_loss(share)
        return {"condition": "bird_drop", "level": "single", "loss": loss,
                "loss_range": rng, "loss_note": note,
                "confidence": None, "by": "photo (splat shape)", "evidence": [],
                "why": f"an irregular white splat covering {share:.1%} of an "
                       f"otherwise clean panel – the typical shape of a bird "
                       f"dropping (the AI models didn't mark it)"}
    level, reason = None, ""
    if m["dirt_share"] >= HEAVY_DIRT:
        level, reason = "heavy", f"{m['dirt_share']:.0%} of the panel looks dirty"
    elif specks >= HEAVY_SPECKS or (specks >= MODERATE_SPECKS
                                    and m["dirt_share"] >= MODERATE_DIRT):
        level, reason = "heavy", ("spots all over the panel – dense dust/debris "
                                  "or many small bird droppings; this photo "
                                  "can't tell them apart")
    elif m["dirt_share"] >= MODERATE_DIRT:
        level, reason = "moderate", f"{m['dirt_share']:.0%} of the panel looks dirty"
    elif specks >= MODERATE_SPECKS:
        level, reason = "moderate", f"many dust specks on the panel ({specks:.0f})"
    elif m["dirt_share"] >= LIGHT_DIRT:
        level, reason = "light", f"a thin layer: {m['dirt_share']:.0%} of the panel looks slightly dirty"
    elif dust_ai:
        level, reason = "light", (f"an AI model ({dust_ai[0]['source']}) saw dust "
                                  f"({dust_ai[0]['confidence']:.0%} sure)")
    if level:
        conf = max([q["confidence"] for q in dust_ai], default=None)
        mixed = "can't tell" in reason
        loss, rng = dust_loss(level, m["dirt_share"])
        note = (f"a dust film blocks light evenly; {m['dirt_share']:.0%} of the "
                f"panel looks dirty")
        if mixed:
            d_loss = loss
            b_loss, b_rng, _ = many_drop_loss(m["groups_hit"])
            loss = round((d_loss + b_loss) / 2, 3)
            rng = (round(min(rng[0], d_loss), 3), b_rng[1])
            note = (f"if it's dust ≈{d_loss:.0%}; if they're droppings spread "
                    f"over the cell groups ≈{b_loss:.0%} – the middle is used")
        return {"condition": "dusty", "level": level, "loss": loss,
                "loss_range": rng, "loss_note": note,
                "confidence": conf, "by": "pixel measurement" if not dust_ai
                else dust_ai[0]["source"], "evidence": dust_ai,
                "mixed": mixed,
                "why": (f"heavy soiling – {reason}" if mixed
                        else f"{level} dust – {reason}")}
    return {"condition": "clean", "level": "", "loss": 0.0, "confidence": None,
            "loss_range": (0.0, 0.0), "loss_note": "",
            "by": "pixel measurement", "evidence": [],
            "why": f"looks clean: {facts}"}


def draw(img, mask, m, decision):
    """Outline (green), dirt (red tint), blobs (circles), verdict banner."""
    out = img.copy()
    dirt = m["dirt_mask"]
    out[dirt] = (0.55 * out[dirt] + np.array([0, 0, 115])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (60, 220, 60), 2)
    for b in m["blobs"]:
        if b["share"] < BIRD_MAX_SHARE:
            cv2.circle(out, (int(b["x"]), int(b["y"])), int(b["r"] + 8), (0, 0, 255), 2)
    for q in decision.get("evidence", []):
        x0, y0 = int(q["x"] - q["width"] / 2), int(q["y"] - q["height"] / 2)
        cv2.rectangle(out, (x0, y0), (int(x0 + q["width"]), int(y0 + q["height"])),
                      (0, 165, 255), 2)
    name = {"clean": "CLEAN", "dusty": "DUST", "bird_drop": "BIRD DROPPING",
            "cracked": "DAMAGE"}[decision["condition"]]
    lvl = decision["level"] if decision["condition"] == "dusty" else ""
    if decision.get("mixed"):
        name = "DUST OR DROPPINGS"
    if decision["condition"] == "bird_drop" and decision["level"] == "many":
        name, lvl = "BIRD DROPPINGS", "many"
    text = f"{name} {lvl.upper()}  ~{decision['loss']:.0%} power loss  |  {m['dirt_share']:.0%} looks dirty"
    colour = {"clean": (60, 170, 60), "dusty": (0, 140, 255),
              "bird_drop": (40, 40, 220), "cracked": (30, 30, 160)}[decision["condition"]]
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), colour, -1)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out
