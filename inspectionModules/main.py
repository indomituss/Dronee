"""
main.py – Inspection pipeline
================================================
Standalone test of the inspection system.
Run this directly to test detection WITHOUT flying:

    python inspectionModules/main.py

For full flight + inspection, use drone.py instead.
"""

import cv2
import os
from datetime import datetime

from detector import PanelDetector
from panel_map import PanelMap
from reporter import InspectionReporter


def run_inspection_on_images(image_folder, model_path='yolov8n.pt'):
    """
    Runs inspection on a folder of saved images.
    Use this to test detection without flying at all.
    """
    detector = PanelDetector(model_path)
    panel_map = PanelMap.from_user_input()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"inspections/{session_id}"
    reporter = InspectionReporter(session_dir)

    images = sorted([
        f for f in os.listdir(image_folder)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"\n[INSPECTION] Processing {len(images)} images from {image_folder}")

    for i, filename in enumerate(images):
        panel_id = panel_map.next_panel_id()
        image_path = os.path.join(image_folder, filename)

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[INSPECTION] Could not read: {filename}")
            continue

        result = detector.detect(frame)
        result['image_path'] = image_path

        panel_map.record(panel_id, result)

        # Show annotated frame
        cv2.imshow(f"Panel {panel_id}", result['annotated_frame'])
        cv2.waitKey(500)  # show for 500ms

        print(
            f"[INSPECTION] Panel {panel_id}: "
            f"{result['worst_class']} "
            f"(severity {result['worst_severity']})"
        )

    cv2.destroyAllWindows()

    # Generate reports
    reporter.save_csv(panel_map)
    reporter.save_summary(panel_map)

    print("\n[INSPECTION] Complete.")


if __name__ == "__main__":
    # Test on saved images without drone
    run_inspection_on_images(
        image_folder="test_images",  # put your cardboard photos here
        model_path='yolov8n.pt'      # replace with custom model later
    )