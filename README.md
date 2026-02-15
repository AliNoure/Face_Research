# Face Research Demo (Python)

This repo gives you a **research-friendly baseline** for face recognition and simple anti-spoof checks that you can present to your supervisor.

## What you get

- Face enrollment script (`src/build_face_db.py`) for adding identities from images.
- Realtime recognition demo (`src/recognize_demo.py`) with:
  - recognition by embedding distance,
  - multi-face support,
  - simple spoof-risk heuristics,
  - optional phone detection (YOLOv8 if installed).
- A comparison document (`docs/approach_comparison.md`) with pros/cons and report structure.

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> On Linux, `face-recognition` may require system packages for `dlib` compilation.

## 2) Add known people (including yourself)

Put images like this:

```text
data/known_faces/
  YourName/
    1.jpg
    2.jpg
  Person2/
    1.jpg
  Person3/
    1.jpg
```

Then build the database:

```bash
python src/build_face_db.py --input-dir data/known_faces --output data/face_db.pkl
```

## 3) Run recognition demo

```bash
python src/recognize_demo.py --db data/face_db.pkl --threshold 0.47 --detector hog
```

Optional phone detection (needs ultralytics + model download):

```bash
pip install ultralytics
python src/recognize_demo.py --db data/face_db.pkl --enable-phone-detection
```

## Notes for your presentation

- Say clearly that spoof detection here is a **starter baseline**.
- Mention that stronger anti-spoofing should include challenge-response and/or dedicated PAD models.
- Use `docs/approach_comparison.md` as the base of your research discussion.

## Suggested message update to your professor

> Good afternoon Professor, I prepared a working baseline that compares practical face recognition options and includes a live multi-person demo. I enrolled myself and two additional subjects, then evaluated recognition quality and runtime. I also implemented initial anti-spoof defenses (photo/screen replay heuristics and optional phone detection) and documented limitations plus next steps toward stronger liveness checks.
