# SPIDrone: Solar Panel Inspection Drone

A desktop app that flies a DJI/Ryze **Tello** drone over solar panels, takes clean photos of each panel despite a weak Wi-Fi video link, works out what is on each panel (dust, bird droppings, cracks) and estimates **how much power and money each panel is losing**, with a cleaning priority for every panel.

## What it does

1. **Fly.** Keyboard or on-screen flight controls with live video, battery, height and temperature, and a safety monitor that warns on low battery or high temperature and can retrace the flight path home.
2. **Capture.** Press **I** at a panel and the app watches the video for about 2.5 s, then keeps the 2 sharpest frames that the decoder confirms are undamaged by Wi-Fi packet loss. If every frame was damaged, it says so and **R** retakes the panel.
3. **Analyse.** After landing, every photo is checked by hosted Roboflow models plus pixel measurements made *inside the panel outline only*:
   - **Panel outline:** found by a photovoltaic outline model. If that misses, a whole-panel box is trimmed to the cells.
   - **Dust:** the share of the panel that looks pale and colourless (works for blue and black panels) and the density of specks, ignoring grid lines and frames. Rated light, moderate or heavy.
   - **Bird droppings:** irregular white splats, confirmed by an AI model or by their shape. Many droppings are detected when spots cover several of the panel's cell groups.
   - **Cracks / damage:** from a damage model.
4. **Report.** A table with condition, priority, AI confidence, power lost (with a likely range) and ₹ lost per year. Each panel opens a plain-English explanation: why it was judged that way, what the AI saw, how the loss was estimated, and photo quality. Saved as CSV (opens in Excel) plus a text summary.

## Power-loss model

A typical panel has ~60 cells wired as 3 groups ("substrings"), each with a bypass diode.

| Finding | Estimated loss |
|---|---|
| Light / moderate / heavy dust | scales with the dirty share, ≈3–35 % |
| One bird dropping | ≈15–33 %: about half a cell or more switches off its whole group (~⅓ of the panel) |
| Many droppings | ≈40 % if 2 of 3 groups are hit, ≈60 % if all 3 are |
| Cracks | ≈25 % and growing; a safety issue, so call the installer |

₹ figures use the panel wattage, tariff, sun hours and cleaning cost set in the app (Setup tab). Cleaning priority follows the payback time (cleaning cost ÷ daily loss).

## Setup (Windows)

```bash
git clone <this repository's URL>
cd <repository folder>
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Get a free Roboflow API key (app.roboflow.com → Settings → API Keys), then either paste it into the app's **Setup** tab or create a `.env` file:

```
ROBOFLOW_API_KEY=your_private_key
```

`.env` is git-ignored; never commit your key.

## Run

```bash
.venv\Scripts\pythonw app.py
```

1. Turn on the Tello and wait for the fast yellow blink.
2. **Flight** tab → **Connect to drone**. The app joins the drone's `TELLO-XXXXXX` Wi-Fi itself and keeps the laptop on it during the flight (Windows otherwise jumps back to a network with internet).
3. Take off, frame each panel inside the white box, press **I** and hold still. Land.
4. **Report** tab → **Analyse this flight**. The app switches back to your previous Wi-Fi for the analysis.

| Key | Action |
|---|---|
| W / S | forward / back |
| A / D | left / right |
| ↑ / ↓ | up / down |
| ← / → | rotate |
| 0 / 1 | take off / land |
| I | capture panel |
| R | retake last panel |
| Space | emergency land |

## Project layout

```
app.py                      desktop app (flight, capture, report)
wifi.py                     joins / keeps the drone's Wi-Fi (Windows netsh)
inspectionModules/
  detector.py               Roboflow models + combining their results
  panel_analysis.py         panel outline, dirt/dropping measurement, loss model
  panel_map.py              keeps the worse of each panel's photos
  reporter.py               CSV + text report
flyingModules/              flight helpers and safety monitor
drone.py                    older terminal-only version
```

Flight photos (`inspections/`), logs, local settings and `.env` stay on your computer and are excluded from git.

## Models used (Roboflow Universe, public)

- Main: `solar-panel-o6dwf/3` (dust, bird drop, damage, no-defect)
- Bird droppings: `solar-panel-detector-b9iyt/1`
- Cracks + whole-panel box: `solar-panel-w3f7s/7`
- Panel outline: `photovoltaic-0enc4/9`

They were chosen by testing 12 public models on labelled photos. The "Smart analysis" option on the Setup tab uses all four (4 Roboflow calls per photo); untick it to use the main model only.

## Limits

- The Tello camera faces forward and its 720p video is compressed over Wi-Fi, so fly within 3–5 m of each panel.
- Exact dropping counts aren't reliable from drone video. The app reports how many of the 3 cell groups are affected and gives a loss **range**.
- Losses are physics-based estimates. Compare inverter output before and after cleaning to confirm them.

## Licence

MIT, see `LICENSE`.
