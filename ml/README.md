# RabbitCam ML

`rabbitcam-ml` is the single-frame posture classifier for PetCam / RabbitCam. It
classifies each image as exactly one of `standing`, `lateral`, or `unknown`.
It does **not** decide whether a posture is dangerous and it does not implement
timers, alerts, notifications, or an HTTP service. A homeserver can consume the
stable Python inference API and apply that temporal policy separately.

The default model is the ImageNet-pretrained
`mobilenetv4_conv_small.e3600_r256_in1k` from `timm`. Images are converted to
grayscale, replicated to three channels, aspect-ratio resized, and letterboxed
to 256 x 256. Validation and inference use the same deterministic transform.

## 1. Install

Python 3.11 or newer is required. From this `ml/` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
rabbitcam-ml --help
```

On Linux or macOS, activate with `source .venv/bin/activate`. The pretrained
weights are downloaded by `timm` on the first real training run. The test suite
never uses the network or downloads model weights.

## 2. Capture independent labeled sessions

RabbitCam serves MJPEG at `http://CAMERA_IP:81/stream`. Collect multiple short,
independent sessions for each class, changing the rabbit's position and camera
conditions between sessions. Do not substitute one long capture: adjacent
frames are strongly correlated and the splitter keeps every session wholly in
one split.

```powershell
rabbitcam-ml capture --stream-url http://192.168.100.22:81/stream `
  --label standing --lighting visible --session standing-visible-001 `
  --interval 1.0 --duration 60

rabbitcam-ml capture --stream-url http://192.168.100.22:81/stream `
  --label standing --lighting ir --session standing-ir-001 `
  --interval 1.0 --duration 60

rabbitcam-ml capture --stream-url http://192.168.100.22:81/stream `
  --label lateral --lighting visible --session lateral-visible-001 `
  --interval 1.0 --duration 60

rabbitcam-ml capture --stream-url http://192.168.100.22:81/stream `
  --label lateral --lighting ir --session lateral-ir-001 `
  --interval 1.0 --duration 60
```

Collect several sessions for each combination rather than only these four.
Short sessions avoid asking the rabbit to hold an artificial posture for a long
time.

The `unknown` class is real training data. Capture separate unknown sessions
containing an empty enclosure, a partly visible or obscured rabbit, transitions,
human hands or bodies, severe blur, and otherwise unusable frames:

```powershell
rabbitcam-ml capture --stream-url http://192.168.100.22:81/stream `
  --label unknown --lighting mixed --session unknown-obscured-001 `
  --interval 1.0 --duration 45
```

Capture writes original JPEG payloads when possible plus `session.json` metadata
with the label, lighting, session ID, stream URL, and timestamps. Ctrl+C closes
the current session cleanly. Transient stream failures are retried.

## 3. Prepare and inspect a session-safe split

By default captures live below `data/sessions/`. The reusable sample manifest
and exact session assignments are stored together in `data/splits.json`.

```powershell
rabbitcam-ml prepare --config configs/default.yaml
rabbitcam-ml inspect-split --manifest data/splits.json
```

The default target is 70% train, 15% validation, and 15% test. Allocation is by
`session_id`, never by frame. When too few independent sessions exist to make
the requested split meaningful, preparation reports the shortage instead of
quietly leaking adjacent frames. Re-running uses the saved manifest. Regenerate
deliberately with a seed:

```powershell
rabbitcam-ml prepare --config configs/default.yaml --regenerate --seed 2026
```

Missing and corrupt files are reported. Exact duplicate detection can be
controlled in `configs/default.yaml`.

## 4. Train, observe, stop, and resume

Review `configs/default.yaml`, especially batch size, worker count, epochs, and
the run directory, then start headless training:

```powershell
rabbitcam-ml train --config configs/default.yaml
```

The default run directory is `artifacts/runs/mobilenetv4-v1/`. It contains the
exact config, split, `metrics.csv`, `summary.json`, TensorBoard events, and:

```text
checkpoints/
  last.pt
  best.pt
  periodic-....pt
```

`last.pt` is written atomically after every epoch, periodic step checkpoints
reduce the exposure of a long epoch, and Ctrl+C makes a best-effort resumable
save. A normal invocation automatically resumes a compatible `last.pt`:

```powershell
# Stop the prior command with Ctrl+C, then run the same command.
rabbitcam-ml train --config configs/default.yaml
```

To intentionally discard automatic resume state and begin that run again:

```powershell
rabbitcam-ml train --config configs/default.yaml --fresh
```

Use a new `paths.run_dir` for a genuinely separate experiment. Incompatible
model, class mapping, preprocessing, or material training settings fail with a
clear error instead of partially loading a checkpoint.

Observe any active or completed run in another shell:

```powershell
tensorboard --logdir artifacts/runs
```

TensorBoard includes loss, learning rate, accuracy, balanced accuracy, macro
F1, per-class precision/recall/F1, lateral recall, and validation confusion
matrices. `metrics.csv` and `summary.json` provide non-GUI records.

## 5. Evaluate and inspect checkpoints

Evaluate `best.pt`, `last.pt`, or any periodic checkpoint against a saved split:

```powershell
rabbitcam-ml evaluate `
  --checkpoint artifacts/runs/mobilenetv4-v1/checkpoints/best.pt `
  --split test --config configs/default.yaml
```

The command prints and saves accuracy, balanced accuracy, macro F1, per-class
precision/recall/F1 and counts, lateral recall, and a confusion matrix. Inspect
the self-describing checkpoint without loading a dataset:

```powershell
rabbitcam-ml inspect-checkpoint artifacts/runs/mobilenetv4-v1/checkpoints/last.pt
```

## 6. Classify one image or one live frame

```powershell
rabbitcam-ml predict `
  --checkpoint artifacts/runs/mobilenetv4-v1/checkpoints/best.pt `
  --image rabbit.jpg

rabbitcam-ml predict `
  --checkpoint artifacts/runs/mobilenetv4-v1/checkpoints/best.pt `
  --image rabbit.jpg --json

rabbitcam-ml predict `
  --checkpoint artifacts/runs/mobilenetv4-v1/checkpoints/best.pt `
  --stream-url http://192.168.100.22:81/stream
```

Output contains all three probabilities plus `raw_label`, the direct argmax,
and `label`, the public result. If the winning score is below the configured
confidence threshold (0.55 by default), `label` becomes `unknown` while
`raw_label` and the probabilities remain visible.

## 7. Import the stable inference API

The model is loaded only once and can be reused by a long-running process:

```python
from PIL import Image
from rabbitcam_ml.inference import Predictor

predictor = Predictor(
    checkpoint_path="artifacts/runs/mobilenetv4-v1/checkpoints/best.pt",
    device="auto",
)

with Image.open("rabbit.jpg") as image:
    result = predictor.predict_pil(image)

print(result.label, result.raw_label, result.confidence)
print(result.probabilities)
```

`predict_bytes(...)` and `predict_numpy(...)` are also available. Prediction
runs in evaluation/inference mode and returns a typed result containing label,
raw label, confidence, all probabilities, and inference time. It intentionally
has no cross-frame or alert logic.

## Layout and generated files

```text
ml/
  configs/default.yaml
  src/rabbitcam_ml/       Python package
  tests/                  offline CPU test suite
  data/sessions/          captured images and metadata (ignored)
  data/splits.json        generated sample/session split manifest (ignored)
  artifacts/runs/         checkpoints, logs, metrics (ignored)
```

Only `.gitkeep` placeholders under `data/` and `artifacts/` are versioned. Do
not commit camera frames, weights, checkpoints, TensorBoard events, or other
generated artifacts.

## Test

All tests are CPU-only, synthetic, offline, and use tiny models:

```powershell
python -m pytest
```

Passing tests validate label handling, deterministic session-only splits,
preprocessing, MJPEG parsing, checkpoint round trips, automatic resume,
confidence fallback, metrics, a tiny training/checkpoint cycle, CLI JSON output,
and repeated Predictor use. They do not establish real-world RabbitCam accuracy;
that requires representative labeled visible-light and 850 nm IR sessions.
