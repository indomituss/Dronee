"""
panel_map.py – Panel data tracker
===================================
No longer asks for layout dimensions.
Simply records inspection results as the drone visits panels.
Panel IDs come from the filename (panel_001, panel_002, etc.)
"""


class PanelMap:
    """
    Tracks inspection results per panel.
    Panel IDs assigned from filenames — no manual input needed.
    """

    def __init__(self):
        self.panel_data = {}   # panel_id → detection result
        self._current   = 0

    def next_panel_id(self):
        self._current += 1
        return self._current

    def record(self, panel_id: int, detection_result: dict):
        """
        Records result for a panel.
        If panel already has a result, keeps the WORSE of the two
        (so if cap_1 is clean and cap_2 is dusty, panel is dusty).
        """
        if panel_id not in self.panel_data:
            self.panel_data[panel_id] = detection_result
        else:
            existing  = self.panel_data[panel_id]
            # Compare the kind of problem first, then how much power it costs
            # (so heavy dust beats light dust)
            new_worse = (
                (detection_result['worst_severity'],
                 detection_result.get('worst_eff_loss', 0.0))
                > (existing['worst_severity'],
                   existing.get('worst_eff_loss', 0.0))
            )
            if new_worse:
                self.panel_data[panel_id] = detection_result
                print(
                    f"[PANEL MAP] Panel {panel_id} updated to worse condition: "
                    f"{detection_result['worst_class']}"
                )

    def summary(self):
        if not self.panel_data:
            return {
                'total_inspected': 0,
                'clean':           0,
                'needs_attention': 0,
                'panels':          {}
            }

        total     = len(self.panel_data)
        clean     = sum(
            1 for d in self.panel_data.values()
            if d['worst_severity'] <= 0
        )
        attention = total - clean

        return {
            'total_inspected': total,
            'clean':           clean,
            'needs_attention': attention,
            'panels':          self.panel_data
        }