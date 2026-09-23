"""Command-line interface for RabbitCam ML."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import sys
from typing import Any, Sequence

from .config import DEFAULT_CONFIG_PATH, load_config, resolve_config_path
from .labels import CLASS_NAMES, LIGHTING_MODES


LOGGER = logging.getLogger("rabbitcam_ml")


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=_path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML configuration (default: {DEFAULT_CONFIG_PATH})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rabbitcam-ml",
        description="Capture, train, evaluate, and run RabbitCam posture models.",
    )
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="capture a labeled MJPEG session")
    capture.add_argument("--stream-url", required=True)
    capture.add_argument("--label", required=True, choices=CLASS_NAMES)
    capture.add_argument("--lighting", required=True, choices=LIGHTING_MODES)
    capture.add_argument("--session", required=True, dest="session_id")
    capture.add_argument("--output-dir", type=_path, default=Path("data/sessions"))
    capture.add_argument("--interval", type=float, default=1.0)
    capture.add_argument("--duration", type=float)
    capture.add_argument("--timeout", type=float, default=10.0)
    capture.add_argument("--max-frames", type=int)
    capture.set_defaults(handler=_capture_command)

    prepare = subparsers.add_parser("prepare", help="build a deterministic session split")
    _add_config(prepare)
    prepare.add_argument("--data-root", type=_path, help="capture sessions directory")
    prepare.add_argument("--output", type=_path, help="split manifest destination")
    prepare.add_argument("--seed", type=int, help="override data.split_seed")
    prepare.add_argument(
        "--regenerate",
        action="store_true",
        help="replace an existing manifest using the requested seed",
    )
    prepare.add_argument(
        "--allow-incomplete-splits",
        action="store_true",
        help="warn instead of failing when a class has fewer than three sessions",
    )
    prepare.set_defaults(handler=_prepare_command)

    inspect_split = subparsers.add_parser(
        "inspect-split", help="summarize and validate a saved split manifest"
    )
    inspect_split.add_argument("--manifest", type=_path, default=Path("data/splits.json"))
    inspect_split.add_argument("--json", action="store_true", dest="as_json")
    inspect_split.set_defaults(handler=_inspect_split_command)

    train = subparsers.add_parser("train", help="train or automatically resume a run")
    _add_config(train)
    train.add_argument("--fresh", action="store_true", help="start over instead of auto-resuming")
    train.set_defaults(handler=_train_command)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a saved checkpoint")
    _add_config(evaluate)
    evaluate.add_argument("--checkpoint", type=_path, required=True)
    evaluate.add_argument("--manifest", type=_path, help="override paths.split_manifest")
    evaluate.add_argument("--split", choices=("train", "validation", "val", "test"), default="test")
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--workers", type=int)
    evaluate.add_argument("--device")
    evaluate.add_argument("--output-json", type=_path)
    evaluate.add_argument("--confusion-matrix", type=_path)
    evaluate.set_defaults(handler=_evaluate_command)

    predict = subparsers.add_parser("predict", help="classify one image or live MJPEG frame")
    predict.add_argument("--checkpoint", type=_path, required=True)
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=_path)
    source.add_argument("--stream-url")
    predict.add_argument("--device", default="auto")
    predict.add_argument("--threshold", type=float, help="override checkpoint confidence threshold")
    predict.add_argument("--timeout", type=float, default=10.0)
    predict.add_argument("--json", action="store_true", dest="as_json")
    predict.set_defaults(handler=_predict_command)

    inspect = subparsers.add_parser(
        "inspect-checkpoint", help="show checkpoint architecture and resume metadata"
    )
    inspect.add_argument("checkpoint", type=_path)
    inspect.set_defaults(handler=_inspect_checkpoint_command)
    return parser


def _capture_command(args: argparse.Namespace) -> int:
    from .capture import capture_session

    result = capture_session(
        stream_url=args.stream_url,
        label=args.label,
        lighting=args.lighting,
        session_id=args.session_id,
        output_dir=args.output_dir,
        interval=args.interval,
        duration=args.duration,
        timeout=args.timeout,
        max_frames=args.max_frames,
    )
    print(f"saved {result.saved_frames} frame(s) in {result.session_dir}")
    print(f"metadata: {result.metadata_path}")
    return 130 if result.interrupted else 0


def _split_summary(manifest: Any) -> dict[str, Any]:
    samples: dict[str, Counter[str]] = defaultdict(Counter)
    sessions: dict[str, Counter[str]] = defaultdict(Counter)
    session_labels: dict[str, str] = {}
    for sample in manifest.samples:
        samples[sample.split][sample.label] += 1
        session_labels.setdefault(sample.session_id, sample.label)
    for session_id, split in manifest.session_splits.items():
        sessions[split][session_labels[session_id]] += 1
    split_names = ("train", "validation", "test")
    return {
        "seed": manifest.seed,
        "ratios": manifest.ratios,
        "sample_count": len(manifest.samples),
        "session_count": len(manifest.session_splits),
        "samples": {
            split: {label: samples[split][label] for label in CLASS_NAMES}
            for split in split_names
        },
        "sessions": {
            split: {label: sessions[split][label] for label in CLASS_NAMES}
            for split in split_names
        },
    }


def _prepare_command(args: argparse.Namespace) -> int:
    from .splits import generate_split_manifest, load_split_manifest

    config = load_config(args.config)
    data_root = args.data_root or (
        resolve_config_path(config["paths"]["data_dir"], config["_config_path"])
        / "sessions"
    )
    destination = args.output or resolve_config_path(
        config["paths"]["split_manifest"], config["_config_path"]
    )
    if destination.exists() and not args.regenerate:
        LOGGER.info("Reusing existing split manifest %s", destination)
        manifest = load_split_manifest(destination)
    else:
        data = config["data"]
        ratios = {
            "train": float(data["train_fraction"]),
            "validation": float(data["validation_fraction"]),
            "test": float(data["test_fraction"]),
        }
        manifest = generate_split_manifest(
            data_root,
            destination,
            seed=int(args.seed if args.seed is not None else data["split_seed"]),
            ratios=ratios,
            strict=not args.allow_incomplete_splits,
            verify_images=bool(data.get("verify_images", True)),
            detect_duplicates=bool(data.get("detect_exact_duplicates", False)),
        )
    print(json.dumps(_split_summary(manifest), indent=2, sort_keys=True))
    print(f"manifest: {destination.resolve()}")
    return 0


def _inspect_split_command(args: argparse.Namespace) -> int:
    from .splits import load_split_manifest

    summary = _split_summary(load_split_manifest(args.manifest))
    if args.as_json:
        print(json.dumps(summary, sort_keys=True))
        return 0
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _train_command(args: argparse.Namespace) -> int:
    from .train import train

    summary = train(args.config, fresh=args.fresh)
    if summary is not None:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    from .evaluate import evaluate_checkpoint

    config = load_config(args.config)
    manifest = args.manifest or resolve_config_path(
        config["paths"]["split_manifest"], config["_config_path"]
    )
    split = "validation" if args.split == "val" else args.split
    checkpoint = args.checkpoint.resolve()
    confusion = args.confusion_matrix or checkpoint.with_name(
        f"{checkpoint.stem}-{split}-confusion.png"
    )
    result = evaluate_checkpoint(
        checkpoint,
        manifest,
        split=split,
        batch_size=int(args.batch_size or config["training"]["batch_size"]),
        num_workers=int(args.workers if args.workers is not None else config["training"]["workers"]),
        device=args.device or config["training"]["device"],
        output_json=args.output_json,
        confusion_matrix_image=confusion,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _predict_command(args: argparse.Namespace) -> int:
    from .inference import Predictor, format_prediction

    predictor = Predictor(
        args.checkpoint,
        device=args.device,
        confidence_threshold=args.threshold,
    )
    prediction = (
        predictor.predict_path(args.image)
        if args.image is not None
        else predictor.predict_stream(args.stream_url, timeout=args.timeout)
    )
    print(prediction.to_json() if args.as_json else format_prediction(prediction))
    return 0


def _inspect_checkpoint_command(args: argparse.Namespace) -> int:
    from .models.checkpoint import inspect_checkpoint

    print(json.dumps(inspect_checkpoint(args.checkpoint), indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted")
        return 130
    except Exception as exc:
        if args.verbose:
            LOGGER.exception("Command failed")
        else:
            LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
