"""
reporter.py - Inspection report with financial impact
=====================================================
Each panel's power loss is the typical value for its condition
(see detector.TYPICAL_EFF_LOSS); the photo is not used to measure "how
much" of the panel is dirty because that estimate was unreliable.

"""

import csv
import os
from datetime import datetime


class InspectionReporter:

    SEVERITY_LABEL = {
        -1: 'API Error / Unknown',
        0:  'Clean',
        2:  'Dusty / Soiled',
        3:  'Bird Dropping',
        5:  'Cracked / Damaged',
    }

    PRIORITY_LABEL = {
        -1: 'Check WiFi Connection',
        0:  'None',
        2:  'Medium - Schedule Cleaning',
        3:  'High - Clean Soon',
        5:  'Urgent - Inspect & Repair',
    }

    def __init__(self, session_dir):
        self.session_dir = session_dir
        os.makedirs(session_dir, exist_ok=True)

    def _calculate_financial_impact(self, data, annual_revenue_per_panel):
        """
        Calculates annual revenue loss for one panel using
        the efficiency loss from Option A pixel coverage analysis.

        Args:
            data                    : panel detection result dict
            annual_revenue_per_panel: float from installation config

        Returns dict with loss figures. No remediation cost included.
        """
        # Use efficiency loss from detector's Option A calculation
        eff_loss = data.get('worst_eff_loss', 0.0)

        # Fallback: if eff_loss missing (old data), use severity-based estimate
        if eff_loss == 0.0 and data.get('worst_severity', 0) > 0:
            fallback = {2: 0.15, 3: 0.33, 5: 0.25}   # = detector.TYPICAL_EFF_LOSS
            eff_loss = fallback.get(data.get('worst_severity', 0), 0.0)

        loss_per_year  = annual_revenue_per_panel * eff_loss
        loss_per_month = loss_per_year / 12
        loss_per_day   = loss_per_year / 365

        coverage_pct = data.get('worst_coverage', 0.0) * 100

        return {
            'efficiency_loss_pct': round(eff_loss * 100, 1),
            'coverage_pct':        round(coverage_pct, 1),
            'loss_per_day':        round(loss_per_day, 2),
            'loss_per_month':      round(loss_per_month, 2),
            'loss_per_year':       round(loss_per_year, 2),
        }

    def save_csv(self, panel_map, config=None):
        path    = os.path.join(self.session_dir, 'inspection_report.csv')
        summary = panel_map.summary()

        annual_rev = (config or {}).get('annual_revenue_per_panel', 0)

        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Panel ID',
                'Condition',
                'Severity',
                'Cleaning Priority',
                'Confidence (%)',
                'Detected Issues',
                'Loss Basis',
                'Typical Power Loss (%)',
                'Daily Loss (Rs.)',
                'Monthly Loss (Rs.)',
                'Annual Loss (Rs.)',
                'Annotated Image'
            ])

            total_annual_loss = 0.0

            for panel_id, data in sorted(summary['panels'].items()):
                severity = data['worst_severity']
                fin      = self._calculate_financial_impact(data, annual_rev)
                total_annual_loss += fin['loss_per_year']

                measured = not (data.get('classification')
                                or data.get('no_coverage'))
                issues = ', '.join(
                    f"{d['class']} ({d['confidence']:.0%} sure)"
                    for d in data.get('detections', [])
                    if d['class'] not in ('panel', 'clean')
                ) or 'None'

                conf = max(
                    (d['confidence'] for d in data.get('detections', [])),
                    default=0.0
                )

                condition = self.SEVERITY_LABEL.get(severity, 'Unknown')
                if data.get('smart') and data.get('worst_class') == 'dusty':
                    condition = f"Dusty ({data.get('level', '')})"
                if data.get('smart'):
                    lo, hi = data.get('loss_range') or (0, 0)
                    basis = (f"{data.get('why', '')} | loss likely "
                             f"{lo:.0%}-{hi:.0%}: {data.get('loss_note', '')} | "
                             f"panel: {data.get('panel_source', '')}")
                elif measured:
                    basis = f"measured {fin['coverage_pct']:.1f}% covered"
                else:
                    basis = 'typical for condition'
                writer.writerow([
                    panel_id,
                    condition,
                    severity,
                    self.PRIORITY_LABEL.get(severity, 'Unknown'),
                    f"{conf:.1%}",
                    issues,
                    basis,
                    f"{fin['efficiency_loss_pct']:.1f}%",
                    f"Rs.{fin['loss_per_day']:.2f}",
                    f"Rs.{fin['loss_per_month']:.0f}",
                    f"Rs.{fin['loss_per_year']:.0f}",
                    data.get('image_path', 'N/A')
                ])

            # Total row
            writer.writerow([])
            writer.writerow([
                'TOTAL', '', '', '', '', '', '', '',
                f"Rs.{total_annual_loss/365:.2f}",
                f"Rs.{total_annual_loss/12:.0f}",
                f"Rs.{total_annual_loss:.0f}",
                ''
            ])

        print(f"[REPORTER] CSV saved: {path}")
        return path

    def save_summary(self, panel_map, config=None):
        path      = os.path.join(self.session_dir, 'inspection_summary.txt')
        summary   = panel_map.summary()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        annual_rev    = (config or {}).get('annual_revenue_per_panel', 0)
        panel_wattage = (config or {}).get('panel_wattage', 0)
        tariff        = (config or {}).get('tariff_per_kwh', 0)
        sun_hours     = (config or {}).get('peak_sun_hours', 0)

        total_annual_loss   = 0.0
        panels_needing_work = 0
        panel_financials    = {}

        for panel_id, data in summary['panels'].items():
            fin = self._calculate_financial_impact(data, annual_rev)
            panel_financials[panel_id] = fin
            total_annual_loss += fin['loss_per_year']
            if data['worst_severity'] > 0:
                panels_needing_work += 1

        lines = [
            "=" * 60,
            "      SOLAR PANEL INSPECTION & FINANCIAL REPORT",
            "=" * 60,
            f"  Date/Time          : {timestamp}",
            f"  Panel wattage      : {panel_wattage}W",
            f"  Electricity tariff : Rs.{tariff}/kWh",
            f"  Peak sun hours     : {sun_hours} hrs/day",
            f"  Revenue per panel  : Rs.{annual_rev:,.0f}/year (100% efficiency)",
            "=" * 60,
            "  INSPECTION SUMMARY",
            "-" * 60,
            f"  Total panels inspected : {summary['total_inspected']}",
            f"  Clean panels           : {summary['clean']}",
            f"  Panels needing action  : {panels_needing_work}",
            "=" * 60,
            "  FINANCIAL IMPACT  (typical loss per detected condition)",
            "-" * 60,
            f"  Total annual revenue loss  : Rs.{total_annual_loss:,.0f}",
            f"  Total monthly revenue loss : Rs.{total_annual_loss/12:,.0f}",
            f"  Total daily revenue loss   : Rs.{total_annual_loss/365:,.2f}",
            "=" * 60,
            "  PANEL-BY-PANEL RESULTS",
            "-" * 60,
        ]

        for panel_id, data in sorted(summary['panels'].items()):
            severity  = data['worst_severity']
            condition = self.SEVERITY_LABEL.get(severity, 'Unknown')
            priority  = self.PRIORITY_LABEL.get(severity, 'Unknown')
            fin       = panel_financials[panel_id]

            issues = ', '.join(
                f"{d['class']} ({d['confidence']:.0%} sure)"
                for d in data.get('detections', [])
                if d['class'] not in ('panel', 'clean')
            ) or 'None'

            lines.append(f"  Panel {panel_id:3d}")
            lines.append(f"    Condition      : {condition}")
            lines.append(f"    Issues         : {issues}")
            lines.append(f"    Priority       : {priority}")

            if severity > 0:
                if not (data.get('classification') or data.get('no_coverage')):
                    lines.append(
                        f"    Surface soiled : {fin['coverage_pct']:.1f}% of panel area"
                    )
                lines.append(
                    f"    Typical loss   : {fin['efficiency_loss_pct']:.1f}%"
                )
                lines.append(
                    f"    Daily loss     : Rs.{fin['loss_per_day']:.2f}"
                )
                lines.append(
                    f"    Annual loss    : Rs.{fin['loss_per_year']:,.0f}"
                )
            else:
                lines.append("    Financial      : No loss - panel is clean")

            lines.append("")

        lines += ["=" * 60]

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print('\n'.join(lines))
        print(f"\n[REPORTER] Summary saved: {path}")
        return path