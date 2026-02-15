"""Realtime face recognition demo with basic anti-spoof heuristics.

This script:
1. Loads a precomputed face embedding database.
2. Runs webcam face recognition.
3. Computes a spoof-risk score (printed photo / screen replay heuristics).
4. Optionally detects a phone if ultralytics is installed.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import cv2
import face_recognition
import numpy as np


try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # optional dependency
    YOLO = None


def load_db(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("rb") as f:
        db: dict[str, np.ndarray] = pickle.load(f)

    names = list(db.keys())
    encodings = np.array([db[name] for name in names])
    return names, encodings


def detect_phone(frame: np.ndarray, model: Any | None) -> bool:
    if model is None:
        return False

    results = model.predict(frame, conf=0.35, verbose=False)
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        class_name = results[0].names.get(cls_id, "")
        if class_name in {"cell phone", "mobile phone"}:
            return True
    return False


def spoof_risk(face_crop: np.ndarray) -> tuple[float, str]:
    """Heuristic spoof score in [0, 1]. Higher means more suspicious."""
    if face_crop.size == 0:
        return 1.0, "invalid-crop"

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Very low texture sharpness can indicate print/screen replay.
    blur_risk = 1.0 if lap_var < 55 else 0.0

    # Detect rectangular border around face region (possible photo/screen edges).
    edges = cv2.Canny(gray, 70, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_like = 0.0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        area = cv2.contourArea(c)
        if len(approx) == 4 and area > 0.25 * gray.shape[0] * gray.shape[1]:
            rect_like = 1.0
            break

    risk = 0.6 * blur_risk + 0.4 * rect_like
    reason = f"lap_var={lap_var:.1f}, rect={rect_like:.0f}"
    return float(risk), reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/face_db.pkl"))
    parser.add_argument("--threshold", type=float, default=0.47)
    parser.add_argument("--detector", choices=["hog", "cnn"], default="hog")
    parser.add_argument("--enable-phone-detection", action="store_true")
    args = parser.parse_args()

    names, known_encodings = load_db(args.db)

    yolo_model = None
    if args.enable_phone_detection and YOLO is not None:
        yolo_model = YOLO("yolov8n.pt")
    elif args.enable_phone_detection:
        print("[WARN] ultralytics not installed; phone detection disabled.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("[INFO] Press 'q' to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb, model=args.detector)
        face_encs = face_recognition.face_encodings(rgb, face_locs)

        phone_present = detect_phone(frame, yolo_model)

        for (top, right, bottom, left), enc in zip(face_locs, face_encs):
            dists = face_recognition.face_distance(known_encodings, enc)
            min_idx = int(np.argmin(dists))
            min_dist = float(dists[min_idx])

            label = names[min_idx] if min_dist <= args.threshold else "Unknown"

            crop = frame[max(0, top):bottom, max(0, left):right]
            risk, reason = spoof_risk(crop)
            spoof_flag = risk >= 0.6 or phone_present

            color = (0, 0, 255) if spoof_flag else (0, 255, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            txt = f"{label} d={min_dist:.2f} spoof={risk:.2f}"
            cv2.putText(frame, txt, (left, max(20, top - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if spoof_flag:
                cv2.putText(
                    frame,
                    f"ALERT: possible spoof ({reason}{', phone' if phone_present else ''})",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Face Research Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
