"""Build a simple face embedding database from labeled images.

Expected structure:

data/known_faces/
    Alice/
        img1.jpg
        img2.jpg
    Bob/
        img1.jpg
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import face_recognition
import numpy as np


def compute_person_encoding(person_dir: Path) -> np.ndarray | None:
    encodings = []
    for image_path in sorted(person_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = face_recognition.load_image_file(str(image_path))
        face_locations = face_recognition.face_locations(image, model="hog")

        if len(face_locations) != 1:
            print(
                f"[WARN] {image_path} has {len(face_locations)} faces; expected 1. Skipping."
            )
            continue

        face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
        encodings.append(face_encoding)

    if not encodings:
        return None

    return np.mean(encodings, axis=0)


def build_database(input_dir: Path, output_path: Path) -> None:
    db = {}
    for person_dir in sorted(input_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        encoding = compute_person_encoding(person_dir)
        if encoding is None:
            print(f"[WARN] No valid images found for {person_dir.name}; skipping.")
            continue

        db[person_dir.name] = encoding
        print(f"[INFO] Added {person_dir.name} with {len(encoding)}-dim embedding.")

    if not db:
        raise RuntimeError("No people were added to the database.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(db, f)

    print(f"[OK] Saved {len(db)} identities to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build face embedding database")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/known_faces"),
        help="Directory with one folder per identity",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/face_db.pkl"),
        help="Output pickle file",
    )
    args = parser.parse_args()

    build_database(args.input_dir, args.output)


if __name__ == "__main__":
    main()
