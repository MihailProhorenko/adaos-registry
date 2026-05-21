from __future__ import annotations

import os
import json
import zipfile
import shutil
import io
import base64
import tempfile
import logging
import time
from pathlib import Path
from typing import Any, Mapping

try:
    import numpy as np
    _numpy_import_error = None
except Exception as exc:
    np = None
    _numpy_import_error = exc

try:
    from PIL import Image
    _pillow_import_error = None
except Exception as exc:
    Image = None
    _pillow_import_error = exc

try:
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision.transforms import functional as TF
    _torch_import_error = None
except Exception as exc:
    torch = None
    nn = None
    torchvision = None
    TF = None
    _torch_import_error = exc

_log = logging.getLogger("new_face_vision.engine")


class NewFaceVisionEngine:
    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.frames_dir = self.state_dir / "frames"
        self.masks_dir = self.state_dir / "masks"
        self.frames_dir.mkdir(exist_ok=True)
        self.masks_dir.mkdir(exist_ok=True)

        self._device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        self._model = None
        self._frames: dict[str, Path] = {}
        self._masks: dict[str, Path] = {}
        self._metadata = {}
        self._current_frame_idx = 0
        self._threshold = 0.35
        self._warning_threshold = 0.05
        self._alarm_threshold = 0.15
        self._prediction_cache = {}
        self._run_id = 1
        self._seq = 0
        self._processed_frames = 0
        self._dice_sum = 0.0
        self._iou_sum = 0.0
        self._target_fps = 5.0
        self._playback: dict[str, Any] = {
            "mode": "idle",
            "fps": self._target_fps,
            "run_id": self._run_id,
            "updated_at": None,
        }
        self._model_path = None
        self._files: dict[str, dict[str, Any] | None] = {
            "model": None,
            "frames": None,
            "masks": None,
            "metadata": None,
        }
        self._operation: dict[str, Any] = {
            "id": None,
            "label": "",
            "progress": None,
            "error": None,
        }
        self._latest: dict[str, Any] | None = None
        self.last_error: dict[str, Any] | None = None

        _log.info(f"NewFaceVisionEngine initialized. Device: {self._device}")

    def configure(
        self,
        model_path: str | None = None,
        frames_path: str | None = None,
        masks_path: str | None = None,
        metadata_path: str | None = None,
        threshold: float | None = None,
        warning_threshold: float | None = None,
        alarm_threshold: float | None = None,
    ) -> dict[str, Any]:
        result = {"ok": True, "actions": []}

        if threshold is not None:
            self._threshold = self._normalize_threshold(threshold, fallback=self._threshold)
            result["actions"].append(f"threshold={self._threshold}")
        if warning_threshold is not None:
            self._warning_threshold = self._normalize_threshold(warning_threshold, fallback=self._warning_threshold)
            result["actions"].append(f"warning_threshold={self._warning_threshold}")
        if alarm_threshold is not None:
            self._alarm_threshold = self._normalize_threshold(alarm_threshold, fallback=self._alarm_threshold)
            result["actions"].append(f"alarm_threshold={self._alarm_threshold}")

        if model_path:
            load_result = self.load_model(model_path)
            result.update(load_result)
            if not load_result.get("ok", True):
                result["ok"] = False
                return result

        if frames_path:
            load_result = self.load_frames(frames_path)
            result.update(load_result)
            if not load_result.get("ok", True):
                result["ok"] = False
                return result

        if masks_path:
            load_result = self.load_masks(masks_path)
            result.update(load_result)
            if not load_result.get("ok", True):
                result["ok"] = False
                return result

        if metadata_path:
            load_result = self.load_metadata(metadata_path)
            result.update(load_result)
            if not load_result.get("ok", True):
                result["ok"] = False
                return result

        return result

    def load_model(self, path: str, *, source_ref: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            self._begin_operation("load_model", "Load model")
            _log.info(f"Loading model from {path}")

            deps_ok, deps_details = self._ensure_model_dependencies()
            if not deps_ok:
                return self._fail_operation(
                    {
                        "message": "torch/torchvision are not installed",
                        "details": deps_details,
                    },
                    code="dependency_missing",
                )

            if not os.path.exists(path):
                return self._fail_operation(f"Model file not found: {path}", code="file_not_found")

            checkpoint = torch.load(path, map_location=self._device)

            model = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=None,
                weights_backbone=None
            )
            model.classifier[-1] = nn.Conv2d(256, 1, kernel_size=1)

            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'], strict=False)
                _log.info(f"Loaded checkpoint epoch: {checkpoint.get('epoch', '?')}")
            else:
                model.load_state_dict(checkpoint, strict=False)

            model.to(self._device)
            model.eval()
            self._model = model
            self._model_path = path
            self._files["model"] = self._file_ref(path, source_ref=source_ref)

            size_mb = os.path.getsize(path) / 1024 / 1024
            _log.info(f"Model loaded: {size_mb:.1f} MB on {self._device}")

            cleanup = self._cleanup_previous_uploads(Path(path))
            if cleanup:
                self._files["model"]["cleanup"] = cleanup
            self._end_operation()
            return {"ok": True, "model_loaded": True, "device": self._device, "size_mb": round(size_mb, 1)}

        except Exception as e:
            _log.error(f"Failed to load model: {e}")
            return self._fail_operation(str(e), code="load_model_failed")

    def load_frames(self, path: str, *, source_ref: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            self._begin_operation("load_frames", "Load frames")
            _log.info(f"Loading frames from {path}")

            deps_ok, deps_details = self._ensure_image_dependencies()
            if not deps_ok:
                return self._fail_operation(
                    {
                        "message": "Pillow/numpy are not installed",
                        "details": deps_details,
                    },
                    code="dependency_missing",
                )

            if not os.path.exists(path):
                return self._fail_operation(f"Frames path not found: {path}", code="file_not_found")

            source_path = Path(path)
            if source_path.is_file() and source_path.suffix.lower() == '.zip':
                if self.frames_dir.exists():
                    shutil.rmtree(self.frames_dir)
                self.frames_dir.mkdir(exist_ok=True)

                self._extract_zip_safely(source_path, self.frames_dir)
                images_dir = self.frames_dir
            elif source_path.is_dir():
                images_dir = source_path
            else:
                images_dir = self.frames_dir

            self._frames = self._load_images_from_folder(str(images_dir))
            self._current_frame_idx = 0
            self._prediction_cache = {}
            self._latest = None
            self._begin_run(mode="idle")

            if len(self._frames) == 0:
                return self._fail_operation("No images found", code="empty_dataset")

            _log.info(f"Loaded {len(self._frames)} frames")
            self._files["frames"] = self._file_ref(path, source_ref=source_ref)
            cleanup = self._cleanup_previous_uploads(Path(path))
            if cleanup:
                self._files["frames"]["cleanup"] = cleanup
            self._end_operation()
            return {"ok": True, "total_frames": len(self._frames)}

        except Exception as e:
            _log.error(f"Failed to load frames: {e}")
            return self._fail_operation(str(e), code="load_frames_failed")

    def load_masks(self, path: str, *, source_ref: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            self._begin_operation("load_masks", "Load masks")
            _log.info(f"Loading masks from {path}")

            deps_ok, deps_details = self._ensure_image_dependencies()
            if not deps_ok:
                return self._fail_operation(
                    {
                        "message": "Pillow/numpy are not installed",
                        "details": deps_details,
                    },
                    code="dependency_missing",
                )

            if not os.path.exists(path):
                return self._fail_operation(f"Masks path not found: {path}", code="file_not_found")

            source_path = Path(path)
            if source_path.is_file() and source_path.suffix.lower() == '.zip':
                if self.masks_dir.exists():
                    shutil.rmtree(self.masks_dir)
                self.masks_dir.mkdir(exist_ok=True)

                self._extract_zip_safely(source_path, self.masks_dir)
                masks_dir = self.masks_dir
            elif source_path.is_dir():
                masks_dir = source_path
            else:
                masks_dir = self.masks_dir

            self._masks = self._load_images_from_folder(str(masks_dir))

            _log.info(f"Loaded {len(self._masks)} masks")
            self._files["masks"] = self._file_ref(path, source_ref=source_ref)
            cleanup = self._cleanup_previous_uploads(Path(path))
            if cleanup:
                self._files["masks"]["cleanup"] = cleanup
            self._end_operation()
            return {"ok": True, "loaded_masks": len(self._masks)}

        except Exception as e:
            _log.error(f"Failed to load masks: {e}")
            return self._fail_operation(str(e), code="load_masks_failed")

    def load_metadata(self, path: str, *, source_ref: Mapping[str, Any] | None = None) -> dict[str, Any]:
        try:
            self._begin_operation("load_metadata", "Load metadata")
            _log.info(f"Loading metadata from {path}")

            if not os.path.exists(path):
                return self._fail_operation(f"Metadata file not found: {path}", code="file_not_found")

            self._metadata = {}
            with open(path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            frame_idx = data.get('frame_idx', i)
                            self._metadata[int(frame_idx)] = data
                        except json.JSONDecodeError:
                            continue

            _log.info(f"Loaded {len(self._metadata)} metadata entries")
            self._files["metadata"] = self._file_ref(path, source_ref=source_ref)
            cleanup = self._cleanup_previous_uploads(Path(path))
            if cleanup:
                self._files["metadata"]["cleanup"] = cleanup
            self._end_operation()
            return {"ok": True, "loaded_metadata": len(self._metadata)}

        except Exception as e:
            _log.error(f"Failed to load metadata: {e}")
            return self._fail_operation(str(e), code="load_metadata_failed")

    def process_frame(self, frame_idx: int | None = None) -> dict[str, Any]:
        try:
            self._begin_operation("process_frame", "Process frame")
            deps_ok, deps_details = self._ensure_image_dependencies()
            if not deps_ok:
                return self._fail_operation(
                    {
                        "message": "Pillow/numpy are not installed",
                        "details": deps_details,
                    },
                    code="dependency_missing",
                )

            if not self._frames:
                return self._fail_operation("No frames loaded", code="frames_missing")

            frame_keys = sorted(self._frames.keys())

            if frame_idx is None:
                frame_idx = self._current_frame_idx

            if frame_idx >= len(frame_keys):
                frame_idx = 0

            frame_key = frame_keys[frame_idx]

            cache_key = str(frame_idx)
            if cache_key in self._prediction_cache:
                result = self._record_frame_result(dict(self._prediction_cache[cache_key]), total_frames=len(frame_keys))
                self._end_operation()
                return result

            frame = self._load_image_ref(self._frames[frame_key])

            gt_mask = None
            for key in self._masks:
                if frame_key in key or key in frame_key:
                    gt_mask = self._load_image_ref(self._masks[key])
                    break

            if self._model is not None:
                predicted_mask, _ = self._predict_with_model(frame)
                predicted_mask = Image.fromarray(predicted_mask)
            else:
                predicted_mask = self._create_dummy_prediction(frame)

            side_by_side = self._create_side_by_side_image(frame, gt_mask, predicted_mask)

            buffered = io.BytesIO()
            side_by_side.save(buffered, format="JPEG", quality=85, optimize=True)
            preview_base64 = base64.b64encode(buffered.getvalue()).decode()

            pred_ratio = float(np.mean(np.array(predicted_mask) > 0))

            true_ratio = None
            if frame_idx in self._metadata:
                true_ratio = self._metadata[frame_idx].get('ratio_bad_true')

            if pred_ratio >= self._alarm_threshold:
                status, status_color = "Alarm", "red"
            elif pred_ratio >= self._warning_threshold:
                status, status_color = "Warning", "yellow"
            else:
                status, status_color = "OK", "green"

            metrics = {"dice": 0, "iou": 0}
            if gt_mask is not None:
                dice_val, iou_val = self._calculate_metrics(predicted_mask, gt_mask)
                metrics = {"dice": round(dice_val, 4), "iou": round(iou_val, 4)}

            result = {
                "ok": True,
                "frame_idx": frame_idx,
                "frame_key": frame_key,
                "total_frames": len(frame_keys),
                "preview_base64": preview_base64,
                "pred_ratio": round(pred_ratio, 4),
                "true_ratio": round(true_ratio, 4) if true_ratio is not None else None,
                "status": status,
                "status_color": status_color,
                "metrics": metrics,
            }

            if len(self._prediction_cache) > 100:
                self._prediction_cache.pop(next(iter(self._prediction_cache)))
            self._prediction_cache[cache_key] = dict(result)
            result = self._record_frame_result(result, total_frames=len(frame_keys))
            self._end_operation()

            return result

        except Exception as e:
            _log.error(f"Failed to process frame: {e}")
            return self._fail_operation(str(e), code="frame_processing_failed")

    def reset(self) -> dict[str, Any]:
        self._begin_operation("reset", "Reset")
        self._current_frame_idx = 0
        self._prediction_cache = {}
        self._latest = None
        self._begin_run(mode="idle")
        self._end_operation()
        return {"ok": True, "message": "Reset completed"}

    def set_playback(self, mode: str, *, fps: float | None = None) -> dict[str, Any]:
        normalized = str(mode or "idle").strip().lower()
        if normalized not in {"idle", "playing", "paused", "stopped"}:
            normalized = "idle"
        if fps is not None:
            self._target_fps = self._normalize_fps(fps)
        self._playback = {
            **self._playback,
            "mode": normalized,
            "fps": self._target_fps,
            "run_id": self._run_id,
            "updated_at": time.time(),
        }
        return {"ok": True, "playback": dict(self._playback)}

    def replay(self, *, fps: float | None = None) -> dict[str, Any]:
        self._current_frame_idx = 0
        self._prediction_cache = {}
        self._latest = None
        self._begin_run(mode="playing", fps=fps)
        return {"ok": True, "message": "Replay started", "playback": dict(self._playback)}

    def stop(self) -> dict[str, Any]:
        self._current_frame_idx = 0
        self._latest = None
        self.set_playback("stopped")
        return {"ok": True, "message": "Playback stopped", "playback": dict(self._playback)}

    def clear(self) -> dict[str, Any]:
        self._model = None
        self._model_path = None
        self._frames = {}
        self._masks = {}
        self._metadata = {}
        self._current_frame_idx = 0
        self._prediction_cache = {}
        self._latest = None
        self._begin_run(mode="idle", bump=False)
        self._files = {
            "model": None,
            "frames": None,
            "masks": None,
            "metadata": None,
        }
        self._operation = {
            "id": None,
            "label": "",
            "progress": None,
            "error": None,
        }
        self.last_error = None

        for dir_path in [self.frames_dir, self.masks_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                dir_path.mkdir(exist_ok=True)

        _log.info("Engine cleared")
        return {"ok": True, "message": "All data cleared"}

    def snapshot(self) -> dict[str, Any]:
        status = "error" if self.last_error else ("ready" if self._frames else "init")
        return {
            "ok": True,
            "status": status,
            "operation": dict(self._operation),
            "files": dict(self._files),
            "file_items": self._file_items(),
            "model": {
                "loaded": self._model is not None,
                "path": self._model_path,
                "device": self._device,
            },
            "stats": {
                "total_frames": len(self._frames),
                "loaded_masks": len(self._masks),
                "loaded_metadata": len(self._metadata),
                "model_loaded": self._model is not None,
                "current_frame": self._latest.get("frame_idx") if self._latest else None,
                "next_frame": self._current_frame_idx,
                "processed_frames": self._processed_frames,
                "avg_dice": self._round_optional(
                    self._dice_sum / self._processed_frames if self._processed_frames else None
                ),
                "avg_iou": self._round_optional(
                    self._iou_sum / self._processed_frames if self._processed_frames else None
                ),
                "fps": self._target_fps,
                "run_id": self._run_id,
            },
            "playback": dict(self._playback),
            "thresholds": {
                "warning": self._warning_threshold,
                "alarm": self._alarm_threshold,
                "prediction": self._threshold,
            },
            "latest": self._latest or self._empty_latest(),
            "error": self.last_error,
            "history": [],
        }

    def frame_stream_payload(self, result: Mapping[str, Any]) -> dict[str, Any]:
        preview = str(result.get("preview_base64") or "")
        frame_idx = result.get("frame_idx")
        total_frames = result.get("total_frames") or len(self._frames)
        return {
            "ok": bool(result.get("ok", True)),
            "id": result.get("id"),
            "seq": result.get("seq"),
            "run_id": result.get("run_id"),
            "frame_idx": frame_idx,
            "frame_key": result.get("frame_key"),
            "frame_label": self._frame_label(frame_idx, total_frames),
            "total_frames": total_frames,
            "image": {
                "mime": "image/jpeg",
                "encoding": "base64",
                "data": preview,
                "src": f"data:image/jpeg;base64,{preview}" if preview else "",
            },
            "prediction": {
                "pred_ratio": result.get("pred_ratio"),
                "true_ratio": result.get("true_ratio"),
            },
            "status": {
                "label": result.get("status"),
                "color": result.get("status_color"),
            },
            "metrics": dict(result.get("metrics") or {}),
            "ts": time.time(),
        }

    def empty_frame_stream_payload(self, *, label: str = "No frame") -> dict[str, Any]:
        return {
            "ok": True,
            "id": f"{self._run_id}:empty:{self._seq}",
            "seq": self._seq,
            "run_id": self._run_id,
            "frame_idx": None,
            "frame_key": None,
            "frame_label": "",
            "total_frames": len(self._frames),
            "image": {
                "mime": "image/jpeg",
                "encoding": "base64",
                "data": "",
                "src": "",
            },
            "prediction": {
                "pred_ratio": None,
                "true_ratio": None,
            },
            "status": {
                "label": label,
                "color": "medium",
            },
            "metrics": {"dice": 0, "iou": 0},
            "ts": time.time(),
        }

    def metrics_stream_payload(self, result: Mapping[str, Any]) -> dict[str, Any]:
        metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
        frame_idx = result.get("frame_idx")
        total_frames = result.get("total_frames") or len(self._frames)
        return {
            "id": result.get("id"),
            "seq": result.get("seq"),
            "run_id": result.get("run_id"),
            "frame_idx": frame_idx,
            "frame_label": self._frame_label(frame_idx, total_frames),
            "total_frames": total_frames,
            "ts": time.time(),
            "series": {
                "pred_ratio": result.get("pred_ratio"),
                "true_ratio": result.get("true_ratio"),
                "dice": metrics.get("dice"),
                "iou": metrics.get("iou"),
            },
        }

    def _record_frame_result(self, result: Mapping[str, Any], *, total_frames: int) -> dict[str, Any]:
        if not result.get("ok"):
            return dict(result)
        frame_idx = int(result.get("frame_idx") or 0)
        metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
        recorded = dict(result)
        self._seq += 1
        recorded["seq"] = self._seq
        recorded["run_id"] = self._run_id
        recorded["id"] = f"{self._run_id}:{self._seq}"
        pred_ratio = result.get("pred_ratio")
        true_ratio = result.get("true_ratio")
        description_parts = []
        if pred_ratio is not None:
            description_parts.append(f"pred={pred_ratio}")
        if true_ratio is not None:
            description_parts.append(f"true={true_ratio}")
        if metrics:
            description_parts.append(f"dice={metrics.get('dice', 0)}")
            description_parts.append(f"iou={metrics.get('iou', 0)}")
        self._latest = {
            "value": recorded.get("status") or "ok",
            "label": f"frame {frame_idx + 1}/{total_frames}" if total_frames else f"frame {frame_idx}",
            "description": " ".join(description_parts),
            "id": recorded["id"],
            "seq": self._seq,
            "run_id": self._run_id,
            "frame_idx": frame_idx,
            "frame_key": recorded.get("frame_key"),
            "total_frames": total_frames,
            "pred_ratio": pred_ratio,
            "true_ratio": true_ratio,
            "metrics": dict(metrics),
            "status": {
                "label": recorded.get("status"),
                "color": recorded.get("status_color"),
            },
            "ts": time.time(),
        }
        self._processed_frames += 1
        self._dice_sum += self._numeric(metrics.get("dice"))
        self._iou_sum += self._numeric(metrics.get("iou"))
        self.last_error = None
        self._current_frame_idx = (frame_idx + 1) % total_frames if total_frames > 0 else 0
        self._playback = {
            **self._playback,
            "run_id": self._run_id,
            "last_frame_idx": frame_idx,
            "updated_at": self._latest["ts"],
        }
        return recorded

    def _empty_latest(self) -> dict[str, Any]:
        return {
            "value": "--",
            "label": "",
            "description": "",
            "id": None,
            "seq": self._seq,
            "run_id": self._run_id,
            "frame_idx": None,
            "frame_key": None,
            "total_frames": len(self._frames),
            "pred_ratio": None,
            "true_ratio": None,
            "metrics": {"dice": 0, "iou": 0},
            "status": {"label": "", "color": ""},
            "ts": None,
        }

    def _begin_run(self, *, mode: str, fps: float | None = None, bump: bool = True) -> None:
        if bump:
            self._run_id += 1
        if fps is not None:
            self._target_fps = self._normalize_fps(fps)
        self._seq = 0
        self._processed_frames = 0
        self._dice_sum = 0.0
        self._iou_sum = 0.0
        self._playback = {
            "mode": mode,
            "fps": self._target_fps,
            "run_id": self._run_id,
            "updated_at": time.time(),
        }

    def _begin_operation(self, operation_id: str, label: str) -> None:
        self._operation = {
            "id": operation_id,
            "label": label,
            "progress": 0.0,
            "error": None,
        }

    def _end_operation(self) -> None:
        self._operation = {
            **self._operation,
            "progress": 1.0,
            "error": None,
        }
        self.last_error = None

    def _fail_operation(
        self,
        error: Any,
        *,
        code: str = "operation_failed",
        retryable: bool = False,
    ) -> dict[str, Any]:
        normalized = self._normalize_error(error, code=code, retryable=retryable)
        self.last_error = normalized
        self._operation = {
            **self._operation,
            "error": normalized,
        }
        return {"ok": False, "error": normalized}

    def _normalize_error(
        self,
        error: Any,
        *,
        code: str = "operation_failed",
        retryable: bool = False,
    ) -> dict[str, Any]:
        if isinstance(error, Mapping):
            message = str(error.get("message") or error.get("error") or error.get("code") or code)
            out: dict[str, Any] = {
                "code": str(error.get("code") or code),
                "message": message,
                "retryable": bool(error.get("retryable", retryable)),
                "ts": float(error.get("ts")) if isinstance(error.get("ts"), (int, float)) else time.time(),
            }
            if "details" in error:
                out["details"] = error.get("details")
            return out
        return {
            "code": code,
            "message": str(error or code),
            "retryable": retryable,
            "ts": time.time(),
        }

    def _normalize_threshold(self, value: Any, *, fallback: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            return fallback
        if not 0 <= parsed <= 1:
            return fallback
        return round(parsed, 4)

    def _normalize_fps(self, value: Any) -> float:
        try:
            parsed = float(value)
        except Exception:
            parsed = self._target_fps
        if parsed < 0.5:
            parsed = 0.5
        if parsed > 30:
            parsed = 30
        return round(parsed, 2)

    def _numeric(self, value: Any) -> float:
        try:
            parsed = float(value)
        except Exception:
            return 0.0
        return parsed if parsed == parsed else 0.0

    def _round_optional(self, value: Any) -> float | None:
        try:
            parsed = float(value)
        except Exception:
            return None
        if parsed != parsed:
            return None
        return round(parsed, 4)

    def _ensure_image_dependencies(self) -> tuple[bool, dict[str, str]]:
        global Image, np, _numpy_import_error, _pillow_import_error

        details: dict[str, str] = {}
        if np is None:
            try:
                import numpy as imported_np

                np = imported_np
                _numpy_import_error = None
            except Exception as exc:
                _numpy_import_error = exc
        if Image is None:
            try:
                from PIL import Image as imported_image

                Image = imported_image
                _pillow_import_error = None
            except Exception as exc:
                _pillow_import_error = exc

        if np is None and _numpy_import_error is not None:
            details["numpy"] = repr(_numpy_import_error)
        if Image is None and _pillow_import_error is not None:
            details["pillow"] = repr(_pillow_import_error)
        return Image is not None and np is not None, details

    def _ensure_model_dependencies(self) -> tuple[bool, dict[str, str]]:
        global TF, nn, torch, torchvision, _torch_import_error

        if torch is None or nn is None or torchvision is None or TF is None:
            try:
                import torch as imported_torch
                import torch.nn as imported_nn
                import torchvision as imported_torchvision
                from torchvision.transforms import functional as imported_tf

                torch = imported_torch
                nn = imported_nn
                torchvision = imported_torchvision
                TF = imported_tf
                _torch_import_error = None
            except Exception as exc:
                _torch_import_error = exc

        if torch is not None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        details: dict[str, str] = {}
        if (torch is None or nn is None or torchvision is None or TF is None) and _torch_import_error is not None:
            details["torch"] = repr(_torch_import_error)
        return torch is not None and nn is not None and torchvision is not None and TF is not None, details

    def _frame_label(self, frame_idx: Any, total_frames: Any) -> str:
        if frame_idx is None:
            return ""
        try:
            idx = int(frame_idx)
            total = int(total_frames or 0)
        except Exception:
            return str(frame_idx)
        return f"{idx + 1}/{total}" if total else str(idx)

    def _file_ref(self, path: str, *, source_ref: Mapping[str, Any] | None = None) -> dict[str, Any]:
        file_path = Path(path)
        stat = None
        if file_path.exists() and file_path.is_file():
            stat = file_path.stat()
        out = {
            "path": str(file_path),
            "name": file_path.name,
            "exists": file_path.exists(),
            "size_bytes": stat.st_size if stat is not None else None,
            "modified_at": stat.st_mtime if stat is not None else None,
            "updated_at": stat.st_mtime if stat is not None else None,
        }
        if source_ref:
            out["source"] = dict(source_ref)
        return out

    def _file_items(self) -> list[dict[str, Any]]:
        labels = {
            "model": "Model",
            "frames": "Frames",
            "masks": "Masks",
            "metadata": "Metadata",
        }
        icons = {
            "model": "cube-outline",
            "frames": "images-outline",
            "masks": "layers-outline",
            "metadata": "document-text-outline",
        }
        counters = {
            "model": "loaded" if self._model is not None else "",
            "frames": f"{len(self._frames)} frames" if self._frames else "",
            "masks": f"{len(self._masks)} masks" if self._masks else "",
            "metadata": f"{len(self._metadata)} rows" if self._metadata else "",
        }
        items: list[dict[str, Any]] = []
        for kind in ("model", "frames", "masks", "metadata"):
            ref = self._files.get(kind)
            if not ref:
                continue
            name = str(ref.get("name") or Path(str(ref.get("path") or "")).name or kind)
            size_label = self._format_bytes(ref.get("size_bytes"))
            counter = counters.get(kind) or ""
            title_suffix = f" ({counter})" if counter else ""
            details = {
                "kind": kind,
                "path": ref.get("path"),
                "size": size_label,
                "size_bytes": ref.get("size_bytes"),
                "modified_at": ref.get("modified_at"),
                "exists": ref.get("exists"),
            }
            cleanup = ref.get("cleanup") if isinstance(ref, Mapping) else None
            if cleanup:
                details["cleanup"] = cleanup
            items.append(
                {
                    "id": kind,
                    "kind": kind,
                    "icon": icons.get(kind),
                    "label": f"{labels[kind]}: {name}{title_suffix}",
                    "name": name,
                    "updated_at": ref.get("updated_at"),
                    "modified_at": ref.get("modified_at"),
                    "size_bytes": ref.get("size_bytes"),
                    "size_label": size_label,
                    "details": details,
                }
            )
        return items

    def _format_bytes(self, value: Any) -> str:
        try:
            size = int(value)
        except Exception:
            return ""
        if size >= 1024 * 1024 * 1024:
            return f"{size / (1024 * 1024 * 1024):.1f} GB"
        if size >= 1024 * 1024:
            return f"{size / (1024 * 1024):.1f} MB"
        if size >= 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size} B"

    def _cleanup_previous_uploads(self, current_path: Path) -> dict[str, Any] | None:
        try:
            current = current_path.resolve()
        except Exception:
            return None
        if not current.exists() or not current.is_file():
            return None

        purpose_dir = self._upload_purpose_dir(current)
        if purpose_dir is None:
            return None

        deleted_names: list[str] = []
        deleted_bytes = 0
        for candidate in sorted(purpose_dir.rglob("*")):
            if not candidate.is_file():
                continue
            try:
                resolved = candidate.resolve()
            except Exception:
                continue
            if resolved == current or candidate.name.startswith("."):
                continue
            try:
                stat = candidate.stat()
                candidate.unlink()
                deleted_names.append(candidate.name)
                deleted_bytes += int(stat.st_size)
            except Exception as exc:
                _log.warning("failed to remove stale upload %s: %s", candidate, exc)

        for directory in sorted(
            [item for item in purpose_dir.rglob("*") if item.is_dir()],
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass

        if not deleted_names:
            return None
        return {
            "deleted_count": len(deleted_names),
            "deleted_names": deleted_names[:20],
            "deleted_bytes": deleted_bytes,
        }

    def _upload_purpose_dir(self, current: Path) -> Path | None:
        for parent in current.parents:
            if parent.name and parent.parent.name == "uploads":
                return parent
        return None

    def _load_images_from_folder(self, folder_path: str) -> dict[str, Path]:
        images: dict[str, Path] = {}
        folder = Path(folder_path)

        if Image is None:
            return images

        if not folder.exists():
            return images

        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}

        for img_path in sorted(folder.rglob('*')):
            if img_path.suffix.lower() in image_extensions:
                images[img_path.stem] = img_path

        return images

    def _extract_zip_safely(self, zip_path: Path, dest_dir: Path) -> None:
        dest_root = dest_dir.resolve()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.infolist():
                if member.is_dir():
                    continue
                member_name = str(member.filename or "").replace("\\", "/")
                if not member_name or member_name.startswith("/") or ".." in Path(member_name).parts:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
                target = (dest_root / member_name).resolve()
                if dest_root not in target.parents and target != dest_root:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_ref.open(member) as source, open(target, "wb") as sink:
                    shutil.copyfileobj(source, sink)

    def _load_image_ref(self, img_path: Path) -> Image.Image:
        with Image.open(img_path) as img:
            return img.copy()

    def _create_dummy_prediction(self, frame: Image.Image) -> Image.Image:
        img_array = np.array(frame.convert('L'))
        threshold = np.mean(img_array) * 0.8
        pred_mask = (img_array < threshold).astype(np.uint8) * 255
        return Image.fromarray(pred_mask)

    def _predict_with_model(self, frame: Image.Image):
        img_tensor = TF.to_tensor(frame).unsqueeze(0).to(self._device)

        with torch.no_grad():
            if self._device == 'cuda':
                with torch.amp.autocast("cuda"):
                    logits = self._model(img_tensor)["out"]
            else:
                logits = self._model(img_tensor)["out"]

            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob > self._threshold).astype(np.uint8) * 255

        return pred, prob

    def _create_side_by_side_image(self, original: Image.Image, gt_mask: Image.Image | None = None, pred_mask: Image.Image | None = None) -> Image.Image:
        if original.mode != 'RGB':
            original = original.convert('RGB')
        original_arr = np.array(original)

        h, w = original_arr.shape[:2]

        panel1 = original_arr.copy()

        panel2 = np.zeros((h, w, 3), dtype=np.uint8)
        if gt_mask is not None:
            gt_arr = np.array(gt_mask)
            if len(gt_arr.shape) == 3:
                gt_arr = gt_arr[:, :, 0]
            if gt_arr.max() > 0:
                gt_arr = (gt_arr > 30).astype(np.uint8) * 255
            panel2[gt_arr > 128] = [255, 255, 255]

        panel3 = np.zeros((h, w, 3), dtype=np.uint8)
        if pred_mask is not None:
            pred_arr = np.array(pred_mask)
            if len(pred_arr.shape) == 3:
                pred_arr = pred_arr[:, :, 0]
            if pred_arr.max() > 0:
                pred_arr = (pred_arr > 30).astype(np.uint8) * 255
            panel3[pred_arr > 128] = [255, 255, 255]

        panel4 = original_arr.copy()
        if pred_mask is not None:
            pred_arr = np.array(pred_mask)
            if len(pred_arr.shape) == 3:
                pred_arr = pred_arr[:, :, 0]
            if pred_arr.max() > 0:
                mask = pred_arr > 30
                if mask.any():
                    panel4[mask] = [255, 0, 0]
                    alpha = 0.6
                    panel4[mask] = (alpha * panel4[mask] + (1 - alpha) * original_arr[mask]).astype(np.uint8)

        combined = np.concatenate([panel1, panel2, panel3, panel4], axis=1)
        return Image.fromarray(combined)

    def _calculate_metrics(self, pred_mask: Image.Image, gt_mask: Image.Image) -> tuple[float, float]:
        pred = (np.array(pred_mask) > 128).astype(np.uint8)
        gt = (np.array(gt_mask) > 128).astype(np.uint8)

        if len(pred.shape) == 3:
            pred = pred[:, :, 0]
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]

        intersection = (pred & gt).sum()
        pred_sum = pred.sum()
        gt_sum = gt.sum()

        eps = 1e-6
        dice = (2.0 * intersection + eps) / (pred_sum + gt_sum + eps)
        iou = (intersection + eps) / (pred_sum + gt_sum - intersection + eps)

        return float(dice), float(iou)
