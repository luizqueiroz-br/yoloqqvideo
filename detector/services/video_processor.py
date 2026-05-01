from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
from ultralytics import YOLO

from detector.services.db_manager import DatabaseManager

try:
    from cv2_enumerate_cameras import enumerate_cameras as _enumerate_cameras
except Exception:
    _enumerate_cameras = None


class VideoProcessor:
    """Processa video, detecta/rasteia objetos e contabiliza eventos unicos."""

    HF_DETECTION_PRESETS = [
        {"repo_id": "Ultralytics/YOLOv8", "filename": "", "label": "Ultralytics/YOLOv8 (colecao oficial)"},
        {"repo_id": "amd/yolov8m", "filename": "yolov8m.pt", "label": "amd/yolov8m"},
        {"repo_id": "amd/yolov8n", "filename": "yolov8n.pt", "label": "amd/yolov8n"},
        {"repo_id": "keremberke/yolov8n-coco", "filename": "", "label": "keremberke/yolov8n-coco"},
        {"repo_id": "keremberke/yolov8m-coco", "filename": "", "label": "keremberke/yolov8m-coco"},
    ]

    @staticmethod
    def _backend_display_name(backend: Optional[int]) -> str:
        if backend is None:
            return "CAP_ANY"
        try:
            return cv2.videoio_registry.getBackendName(int(backend))
        except Exception:
            return str(backend)

    @staticmethod
    def _encode_camera_source(index: int, backend: Optional[int]) -> str:
        backend_token = "any" if backend is None else str(int(backend))
        return f"cam:{int(index)}:{backend_token}"

    @staticmethod
    def _decode_source(source: str | int):
        if isinstance(source, int):
            return source, None, VideoProcessor._encode_camera_source(source, None)

        text = str(source).strip()

        if text.startswith("cam:"):
            parts = text.split(":")
            if len(parts) == 3:
                try:
                    index = int(parts[1])
                    backend = None if parts[2] == "any" else int(parts[2])
                    return index, backend, text
                except ValueError:
                    pass

        if text.isdigit():
            index = int(text)
            return index, None, VideoProcessor._encode_camera_source(index, None)

        return text, None, text

    @staticmethod
    def _open_capture(source, backend: Optional[int] = None):
        if isinstance(source, str) and not source.isdigit():
            if backend is not None:
                return cv2.VideoCapture(source, int(backend))
            return cv2.VideoCapture(source)

        source_index = int(source)
        candidates = []
        if backend is not None:
            candidates.append(int(backend))

        for api in (getattr(cv2, "CAP_ANY", None), getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None)):
            if api is None:
                continue
            api = int(api)
            if api not in candidates:
                candidates.append(api)

        for api in candidates:
            cap = cv2.VideoCapture(source_index, api)
            if cap.isOpened():
                return cap
            cap.release()

        return cv2.VideoCapture(source_index)

    def __init__(
        self,
        source: str = "0",
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        stable_frames: int = 5,
    ):
        parsed_source, parsed_backend, selector_value = self._decode_source(source)
        self.source = parsed_source
        self.source_backend = parsed_backend
        self.source_selector_value = selector_value

        self.model_provider = "ultralytics"
        self.model_source = model_path
        self.hf_repo_id = ""
        self.hf_filename = ""
        self.hf_token = ""
        self.hf_user = ""
        self.iou_threshold = 0.5
        self.imgsz = 640
        self.target_camera_fps = 30
        self.target_client_fps = 12
        self.capture_width = 1280
        self.capture_height = 720
        self.class_filter_text = ""
        self.allowed_class_ids: Optional[List[int]] = None
        self.zoom_enabled = False
        self.detect_zoom_only = True
        self.zoom_scale = 1.0
        self.zoom_center_x = 0.5
        self.zoom_center_y = 0.5
        self.roi_enabled = False
        self.roi_x = 0.1
        self.roi_y = 0.1
        self.roi_w = 0.8
        self.roi_h = 0.8
        self.display_roi_only = False

        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.confidence = confidence
        self.stable_frames = stable_frames
        self.db = DatabaseManager()
        self.capture = self._open_capture(self.source, self.source_backend)
        self._apply_capture_settings(self.capture)
        self.running = True
        self.lock = threading.Lock()

        self.class_counts: Dict[str, int] = defaultdict(int)
        self.id_seen_frames: Dict[int, int] = defaultdict(int)
        self.id_last_center: Dict[int, Tuple[float, float]] = {}
        self.id_counted: set[int] = set()
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        self.line_orientation = "horizontal"
        self.line_position_ratio = 0.5
        self.last_status_message = "Inicializando detector..."
        self.last_inference_ok = False
        self.last_inference_ts = 0.0
        self.last_inference_error = ""
        self.last_frame_ts = 0.0
        self.log_all_detections = True
        self.log_counter_events = True
        self.class_detect_counts: Dict[str, int] = defaultdict(int)
        self.id_logged_detection: set[int] = set()

    def get_model_health(self) -> Dict[str, object]:
        with self.lock:
            now = time.time()
            seconds_since_inference = (now - self.last_inference_ts) if self.last_inference_ts > 0 else None
            seconds_since_frame = (now - self.last_frame_ts) if self.last_frame_ts > 0 else None
            health = "warming_up"
            if self.last_inference_ok:
                if seconds_since_inference is not None and seconds_since_inference <= 3.0:
                    health = "ok"
                else:
                    health = "stale"
            elif self.last_inference_error:
                health = "inference_error"
            elif seconds_since_frame is not None and seconds_since_frame > 3.0:
                health = "camera_error"

            return {
                "health": health,
                "model_provider": self.model_provider,
                "model_source": self.model_source,
                "last_status_message": self.last_status_message,
                "last_inference_ok": self.last_inference_ok,
                "last_inference_error": self.last_inference_error,
                "seconds_since_inference": seconds_since_inference,
                "seconds_since_frame": seconds_since_frame,
            }

    def _apply_capture_settings(self, cap):
        if cap is None:
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.capture_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.capture_height))
        cap.set(cv2.CAP_PROP_FPS, float(self.target_camera_fps))

    def _build_status_frame(self, message: str):
        width = max(self.frame_width, 640)
        height = max(self.frame_height, 360)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (20, 28, 38)
        cv2.putText(frame, "VisionQQ - Aguardando video", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
        cv2.putText(frame, message, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 200, 255), 2)
        cv2.putText(frame, "Verifique a fonte da camera no painel.", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 185, 220), 2)
        return frame

    def _line_position_px(self) -> int:
        if self.line_orientation == "vertical":
            return int(self.frame_width * self.line_position_ratio)
        return int(self.frame_height * self.line_position_ratio)

    def _build_zoomed_frame(self, frame):
        if not self.zoom_enabled or self.zoom_scale <= 1.01:
            return frame

        h, w = frame.shape[:2]
        roi_w = max(64, int(w / self.zoom_scale))
        roi_h = max(64, int(h / self.zoom_scale))
        cx = int(self.zoom_center_x * w)
        cy = int(self.zoom_center_y * h)

        x1 = max(0, min(w - roi_w, cx - roi_w // 2))
        y1 = max(0, min(h - roi_h, cy - roi_h // 2))
        x2 = min(w, x1 + roi_w)
        y2 = min(h, y1 + roi_h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def _extract_roi_by_norm(self, frame, x: float, y: float, w_ratio: float, h_ratio: float):
        h, w = frame.shape[:2]
        x = max(0.0, min(0.98, float(x)))
        y = max(0.0, min(0.98, float(y)))
        w_ratio = max(0.02, min(1.0, float(w_ratio)))
        h_ratio = max(0.02, min(1.0, float(h_ratio)))

        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int(min(w, (x + w_ratio) * w))
        y2 = int(min(h, (y + h_ratio) * h))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop, x1, y1, x2, y2

    def _draw_roi_overlay(self, frame):
        if not self.roi_enabled:
            return
        info = self._extract_roi_by_norm(frame, self.roi_x, self.roi_y, self.roi_w, self.roi_h)
        if info is None:
            return
        _, x1, y1, x2, y2 = info
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 215, 0), 2)
        cv2.putText(
            frame,
            "ROI",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 215, 0),
            2,
        )

    def _reconnect_if_needed(self) -> bool:
        if self.capture.isOpened():
            return True
        self.capture.release()
        self.capture = self._open_capture(self.source, self.source_backend)
        self._apply_capture_settings(self.capture)
        opened = self.capture.isOpened()
        if not opened:
            self.last_status_message = "Nao foi possivel abrir a fonte de video."
        return opened

    @staticmethod
    def discover_webcams(max_index: int = 10):
        cameras = []
        seen_values = set()

        if _enumerate_cameras is not None:
            try:
                for info in _enumerate_cameras():
                    index = int(getattr(info, "index"))
                    backend = int(getattr(info, "backend")) if getattr(info, "backend", None) is not None else None
                    name = str(getattr(info, "name", f"Webcam {index}")).strip() or f"Webcam {index}"
                    source_value = VideoProcessor._encode_camera_source(index, backend)
                    if source_value in seen_values:
                        continue
                    seen_values.add(source_value)
                    backend_name = VideoProcessor._backend_display_name(backend)
                    cameras.append({
                        "value": source_value,
                        "label": f"{name} (idx {index}, {backend_name})",
                    })
            except Exception:
                cameras = []

        if cameras:
            return cameras

        # Fallback sem dependencia extra.
        for idx in range(max_index):
            cap = VideoProcessor._open_capture(idx)
            opened = cap.isOpened()
            ok, _ = cap.read() if opened else (False, None)
            cap.release()
            if opened or ok:
                source_value = VideoProcessor._encode_camera_source(idx, None)
                if source_value in seen_values:
                    continue
                seen_values.add(source_value)
                cameras.append({"value": source_value, "label": f"Webcam {idx} (scan)"})

        return cameras

    def _resolve_class_ids(self, class_filter_text: str) -> Optional[List[int]]:
        text = (class_filter_text or "").strip()
        if not text:
            return None

        wanted = {item.strip().lower() for item in text.split(",") if item.strip()}
        if not wanted:
            return None

        mapping = self.class_names or {}
        if isinstance(mapping, dict):
            iterable = mapping.items()
        else:
            iterable = enumerate(mapping)
        ids = [int(idx) for idx, name in iterable if str(name).lower() in wanted]
        return ids if ids else None

    def get_hf_auth_status(self) -> Dict[str, str | bool]:
        with self.lock:
            return {
                "connected": bool(self.hf_token),
                "user": self.hf_user,
            }

    def set_hf_token(self, token: str) -> Dict[str, str | bool]:
        clean = (token or "").strip()
        with self.lock:
            if not clean:
                self.hf_token = ""
                self.hf_user = ""
                os.environ.pop("HF_TOKEN", None)
                os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)
                return {"connected": False, "user": ""}

        try:
            from huggingface_hub import HfApi
        except Exception as exc:
            raise ValueError("Pacote huggingface_hub nao instalado.") from exc

        try:
            profile = HfApi().whoami(token=clean)
        except Exception as exc:
            raise ValueError(f"Token Hugging Face invalido: {exc}") from exc

        username = str(profile.get("name") or profile.get("fullname") or profile.get("email") or "conectado")
        with self.lock:
            self.hf_token = clean
            self.hf_user = username
            os.environ["HF_TOKEN"] = clean
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = clean
            return {"connected": True, "user": self.hf_user}

    def list_hf_detection_models(self, query: str = "yolo", limit: int = 20):
        normalized_query = (query or "yolo").strip() or "yolo"
        safe_limit = max(5, min(40, int(limit)))

        try:
            from huggingface_hub import HfApi
        except Exception:
            presets = []
            for item in VideoProcessor.HF_DETECTION_PRESETS[:safe_limit]:
                presets.append(
                    {
                        "repo_id": item["repo_id"],
                        "filename": item.get("filename", ""),
                        "label": item.get("label", item["repo_id"]),
                        "source": "preset",
                    }
                )
            return presets, "Pacote huggingface_hub nao encontrado; exibindo presets locais."

        models = []
        error_message = ""
        try:
            api = HfApi()
            try:
                candidates = api.list_models(
                    search=normalized_query,
                    filter="object-detection",
                    sort="downloads",
                    direction=-1,
                    limit=safe_limit,
                    full=True,
                    token=(self.hf_token or None),
                )
            except TypeError:
                # Compatibilidade com versoes do huggingface_hub sem parametro `direction`.
                candidates = api.list_models(
                    search=normalized_query,
                    filter="object-detection",
                    sort="downloads",
                    limit=safe_limit,
                    full=True,
                    token=(self.hf_token or None),
                )
            for model in candidates:
                repo_id = str(getattr(model, "id", "") or "").strip()
                if not repo_id:
                    continue

                siblings = getattr(model, "siblings", []) or []
                pt_files = []
                for sibling in siblings:
                    name = str(getattr(sibling, "rfilename", "") or "").strip()
                    if name.endswith(".pt"):
                        pt_files.append(name)

                downloads = getattr(model, "downloads", None)
                likes = getattr(model, "likes", None)
                stats = []
                if downloads is not None:
                    stats.append(f"downloads {downloads}")
                if likes is not None:
                    stats.append(f"likes {likes}")
                stats_label = f" | {' | '.join(stats)}" if stats else ""

                models.append(
                    {
                        "repo_id": repo_id,
                        "filename": (pt_files[0] if pt_files else ""),
                        "label": f"{repo_id}{stats_label}",
                        "pt_files": pt_files[:3],
                        "source": "huggingface",
                    }
                )
        except Exception as exc:
            error_message = f"Falha ao buscar no Hugging Face ({exc}); exibindo presets locais."

        if not models:
            for item in VideoProcessor.HF_DETECTION_PRESETS[:safe_limit]:
                models.append(
                    {
                        "repo_id": item["repo_id"],
                        "filename": item.get("filename", ""),
                        "label": item.get("label", item["repo_id"]),
                        "source": "preset",
                    }
                )

        return models, error_message

    def _load_model_from_source(
        self,
        provider: str,
        model_source: str,
        hf_filename: str = "",
    ):
        provider = (provider or "ultralytics").strip().lower()
        model_source = (model_source or "").strip()

        if provider == "huggingface":
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
            except Exception as exc:
                raise ValueError(
                    "Pacote huggingface_hub nao instalado. Rode: pip install huggingface_hub"
                ) from exc

            if not model_source:
                raise ValueError("Informe o repo_id do Hugging Face.")

            filename = (hf_filename or "").strip()
            if not filename:
                try:
                    files = list_repo_files(repo_id=model_source, repo_type="model", token=(self.hf_token or None))
                except Exception as exc:
                    raise ValueError(f"Falha ao listar arquivos do repo HF '{model_source}': {exc}") from exc
                pt_files = [item for item in files if item.endswith(".pt")]
                if not pt_files:
                    raise ValueError("Repositorio sem arquivo .pt para YOLO.")
                filename = pt_files[0]

            try:
                resolved_path = hf_hub_download(
                    repo_id=model_source,
                    filename=filename,
                    repo_type="model",
                    token=(self.hf_token or None),
                )
                loaded_model = YOLO(resolved_path)
            except Exception as exc:
                if "A2C2f" in str(exc):
                    raise ValueError(
                        "Modelo exige uma versao mais nova do Ultralytics (camada A2C2f). "
                        "Atualize para ultralytics>=8.4.x e reinicie o servidor."
                    ) from exc
                raise ValueError(
                    f"Falha ao carregar modelo HF '{model_source}/{filename}': {exc}. "
                    "Se o repo for privado/gated, conecte o token HF no painel."
                ) from exc
            return loaded_model, provider, model_source, filename

        if not model_source:
            model_source = "yolov8n.pt"

        try:
            loaded_model = YOLO(model_source)
        except Exception as exc:
            if "A2C2f" in str(exc):
                raise ValueError(
                    "Modelo exige uma versao mais nova do Ultralytics (camada A2C2f). "
                    "Atualize para ultralytics>=8.4.x e reinicie o servidor."
                ) from exc
            raise ValueError(f"Falha ao carregar modelo '{model_source}': {exc}") from exc
        return loaded_model, "ultralytics", model_source, ""

    def get_runtime_config(self) -> Dict[str, object]:
        with self.lock:
            if isinstance(self.class_names, dict):
                classes = list(self.class_names.values())
            else:
                classes = list(self.class_names)
            return {
                "model_provider": self.model_provider,
                "model_source": self.model_source,
                "hf_repo_id": self.hf_repo_id,
                "hf_filename": self.hf_filename,
                "hf_connected": bool(self.hf_token),
                "hf_user": self.hf_user,
                "confidence": self.confidence,
                "iou": self.iou_threshold,
                "imgsz": self.imgsz,
                "stable_frames": self.stable_frames,
                "class_filter": self.class_filter_text,
                "camera_fps": self.target_camera_fps,
                "client_fps": self.target_client_fps,
                "camera_width": self.capture_width,
                "camera_height": self.capture_height,
                "zoom_enabled": self.zoom_enabled,
                "detect_zoom_only": self.detect_zoom_only,
                "zoom_scale": self.zoom_scale,
                "zoom_center_x": self.zoom_center_x,
                "zoom_center_y": self.zoom_center_y,
                "roi_enabled": self.roi_enabled,
                "roi_x": self.roi_x,
                "roi_y": self.roi_y,
                "roi_w": self.roi_w,
                "roi_h": self.roi_h,
                "display_roi_only": self.display_roi_only,
                "log_all_detections": self.log_all_detections,
                "log_counter_events": self.log_counter_events,
                "available_classes": classes,
            }

    def set_runtime_config(self, payload: Dict[str, str]) -> Dict[str, object]:
        provider = payload.get("model_provider")
        model_source = payload.get("model_source")
        hf_repo_id = payload.get("hf_repo_id")
        hf_filename = payload.get("hf_filename")

        confidence = payload.get("confidence")
        iou = payload.get("iou")
        imgsz = payload.get("imgsz")
        stable_frames = payload.get("stable_frames")
        class_filter = payload.get("class_filter")
        camera_fps = payload.get("camera_fps")
        client_fps = payload.get("client_fps")
        camera_width = payload.get("camera_width")
        camera_height = payload.get("camera_height")
        vehicle_focus = payload.get("vehicle_focus")
        zoom_enabled = payload.get("zoom_enabled")
        detect_zoom_only = payload.get("detect_zoom_only")
        zoom_scale = payload.get("zoom_scale")
        zoom_center_x = payload.get("zoom_center_x")
        zoom_center_y = payload.get("zoom_center_y")
        roi_enabled = payload.get("roi_enabled")
        roi_x = payload.get("roi_x")
        roi_y = payload.get("roi_y")
        roi_w = payload.get("roi_w")
        roi_h = payload.get("roi_h")
        display_roi_only = payload.get("display_roi_only")
        log_all_detections = payload.get("log_all_detections")
        log_counter_events = payload.get("log_counter_events")

        with self.lock:
            if provider is not None:
                provider_name = (provider or "ultralytics").strip().lower()
                should_reload_model = False

                if provider_name == "huggingface":
                    requested_repo = (hf_repo_id or self.hf_repo_id).strip()
                    requested_filename = (hf_filename or self.hf_filename).strip()
                    if not requested_repo:
                        raise ValueError("Informe o repo_id do Hugging Face.")
                    should_reload_model = (
                        self.model_provider != "huggingface"
                        or requested_repo != self.hf_repo_id
                        or requested_filename != self.hf_filename
                    )
                    if should_reload_model:
                        model, resolved_provider, resolved_source, resolved_hf_filename = self._load_model_from_source(
                            provider_name,
                            requested_repo,
                            requested_filename,
                        )
                        self.model = model
                        self.class_names = self.model.names
                        self.model_provider = resolved_provider
                        self.hf_repo_id = resolved_source
                        self.hf_filename = resolved_hf_filename
                        self.model_source = f"hf:{resolved_source}::{resolved_hf_filename}"
                else:
                    requested_source = (model_source or "").strip()
                    if not requested_source:
                        # Se antes era HF e voltou para Ultralytics sem especificar, volta ao padrao.
                        requested_source = "yolov8n.pt" if self.model_provider == "huggingface" else self.model_source
                    should_reload_model = (
                        self.model_provider != "ultralytics"
                        or requested_source != self.model_source
                    )
                    if should_reload_model:
                        model, resolved_provider, resolved_source, resolved_hf_filename = self._load_model_from_source(
                            "ultralytics",
                            requested_source,
                            "",
                        )
                        self.model = model
                        self.class_names = self.model.names
                        self.model_provider = resolved_provider
                        self.hf_repo_id = ""
                        self.hf_filename = resolved_hf_filename
                        self.model_source = resolved_source

            if confidence is not None and str(confidence).strip() != "":
                self.confidence = min(max(float(confidence), 0.01), 0.99)
            if iou is not None and str(iou).strip() != "":
                self.iou_threshold = min(max(float(iou), 0.01), 0.99)
            if imgsz is not None and str(imgsz).strip() != "":
                self.imgsz = max(320, min(1280, int(float(imgsz))))
            if stable_frames is not None and str(stable_frames).strip() != "":
                self.stable_frames = max(1, min(30, int(float(stable_frames))))
            if camera_fps is not None and str(camera_fps).strip() != "":
                self.target_camera_fps = max(1, min(60, int(float(camera_fps))))
            if client_fps is not None and str(client_fps).strip() != "":
                self.target_client_fps = max(1, min(60, int(float(client_fps))))
            if camera_width is not None and str(camera_width).strip() != "":
                self.capture_width = max(320, min(1920, int(float(camera_width))))
            if camera_height is not None and str(camera_height).strip() != "":
                self.capture_height = max(240, min(1080, int(float(camera_height))))
            if zoom_enabled is not None:
                self.zoom_enabled = str(zoom_enabled).lower() in {"1", "true", "yes", "on"}
            if detect_zoom_only is not None:
                self.detect_zoom_only = str(detect_zoom_only).lower() in {"1", "true", "yes", "on"}
            if zoom_scale is not None and str(zoom_scale).strip() != "":
                self.zoom_scale = max(1.0, min(8.0, float(zoom_scale)))
            if zoom_center_x is not None and str(zoom_center_x).strip() != "":
                self.zoom_center_x = max(0.0, min(1.0, float(zoom_center_x)))
            if zoom_center_y is not None and str(zoom_center_y).strip() != "":
                self.zoom_center_y = max(0.0, min(1.0, float(zoom_center_y)))
            if roi_enabled is not None:
                self.roi_enabled = str(roi_enabled).lower() in {"1", "true", "yes", "on"}
            if roi_x is not None and str(roi_x).strip() != "":
                self.roi_x = max(0.0, min(0.98, float(roi_x)))
            if roi_y is not None and str(roi_y).strip() != "":
                self.roi_y = max(0.0, min(0.98, float(roi_y)))
            if roi_w is not None and str(roi_w).strip() != "":
                self.roi_w = max(0.02, min(1.0, float(roi_w)))
            if roi_h is not None and str(roi_h).strip() != "":
                self.roi_h = max(0.02, min(1.0, float(roi_h)))
            if display_roi_only is not None:
                self.display_roi_only = str(display_roi_only).lower() in {"1", "true", "yes", "on"}
            if log_all_detections is not None:
                self.log_all_detections = str(log_all_detections).lower() in {"1", "true", "yes", "on"}
            if log_counter_events is not None:
                self.log_counter_events = str(log_counter_events).lower() in {"1", "true", "yes", "on"}

            if class_filter is not None:
                self.class_filter_text = class_filter.strip()
            if vehicle_focus and str(vehicle_focus).lower() in {"1", "true", "yes", "on"}:
                self.class_filter_text = "car,truck,bus,motorcycle,bicycle"
                self.confidence = min(self.confidence, 0.30)

            self.allowed_class_ids = self._resolve_class_ids(self.class_filter_text)

            # Reaplica configuracao de captura para buscar FPS/resolucao mais altos.
            if self.capture:
                self._apply_capture_settings(self.capture)

            self.id_seen_frames.clear()
            self.id_last_center.clear()
            self.id_counted.clear()
            self.id_logged_detection.clear()

        return self.get_runtime_config()

    def _crossed_line(self, previous_center: Tuple[float, float], current_center: Tuple[float, float]) -> bool:
        threshold = self._line_position_px()
        if self.line_orientation == "vertical":
            return previous_center[0] < threshold <= current_center[0]
        return previous_center[1] < threshold <= current_center[1]

    def _draw_overlay(self, frame):
        threshold = self._line_position_px()
        if self.line_orientation == "vertical":
            cv2.line(frame, (threshold, 0), (threshold, self.frame_height), (0, 0, 255), 2)
        else:
            cv2.line(frame, (0, threshold), (self.frame_width, threshold), (0, 0, 255), 2)
        x = 15
        y = 35
        top_counts = sorted(self.class_counts.items(), key=lambda item: item[1], reverse=True)[:8]
        for name, value in top_counts:
            text = f"{name}: {self.class_counts[name]}"
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            y += 35

    @staticmethod
    def _color_for_class(class_id: int) -> Tuple[int, int, int]:
        # Cor estável por classe para visual consistente.
        return (
            int((int(class_id) * 37) % 255),
            int((int(class_id) * 67) % 255),
            int((int(class_id) * 97) % 255),
        )

    @staticmethod
    def _crop_box(frame, x1: float, y1: float, x2: float, y2: float):
        h, w = frame.shape[:2]
        x1i = max(0, min(w - 1, int(x1)))
        y1i = max(0, min(h - 1, int(y1)))
        x2i = max(0, min(w, int(x2)))
        y2i = max(0, min(h, int(y2)))
        if x2i <= x1i or y2i <= y1i:
            return None
        return frame[y1i:y2i, x1i:x2i].copy()

    def _update_tracking_and_count(
        self,
        class_name: str,
        track_id: int,
        center_x: float,
        center_y: float,
        frame,
        box,
    ):
        self.id_seen_frames[track_id] += 1
        if (
            self.log_all_detections
            and self.id_seen_frames[track_id] >= self.stable_frames
            and track_id not in self.id_logged_detection
        ):
            self.class_detect_counts[class_name] += 1
            self.id_logged_detection.add(track_id)
            x1, y1, x2, y2 = box
            frame_full = frame.copy()
            frame_crop = self._crop_box(frame, x1, y1, x2, y2)
            self.db.enqueue_detection(
                class_name,
                self.class_detect_counts[class_name],
                frame_full=frame_full,
                frame_crop=frame_crop,
                evento_tipo="detected",
                track_id=track_id,
            )

        previous_center = self.id_last_center.get(track_id)
        self.id_last_center[track_id] = (center_x, center_y)

        if previous_center is None:
            return

        if self.id_seen_frames[track_id] < self.stable_frames:
            return

        if track_id in self.id_counted:
            return

        if self._crossed_line(previous_center, (center_x, center_y)):
            self.class_counts[class_name] += 1
            self.id_counted.add(track_id)
            if self.log_counter_events:
                x1, y1, x2, y2 = box
                frame_full = frame.copy()
                frame_crop = self._crop_box(frame, x1, y1, x2, y2)
                self.db.enqueue_detection(
                    class_name,
                    self.class_counts[class_name],
                    frame_full=frame_full,
                    frame_crop=frame_crop,
                    evento_tipo="counter",
                    track_id=track_id,
                )

    def process_frame(self):
        if not self.running:
            self.last_inference_ok = False
            return self._build_status_frame("Processamento encerrado.")
        if not self._reconnect_if_needed():
            self.last_inference_ok = False
            self.last_inference_error = "camera_unavailable"
            return self._build_status_frame(self.last_status_message)

        ok, frame = self.capture.read()
        if not ok:
            self.last_status_message = "Falha ao capturar frame da camera."
            self.last_inference_ok = False
            self.last_inference_error = "camera_read_failed"
            return self._build_status_frame(self.last_status_message)
        self.last_frame_ts = time.time()

        original_frame = frame
        display_frame = original_frame
        detection_frame = original_frame
        map_scale_x = 1.0
        map_scale_y = 1.0
        map_offset_x = 0.0
        map_offset_y = 0.0

        roi_info = None
        if self.roi_enabled:
            roi_info = self._extract_roi_by_norm(original_frame, self.roi_x, self.roi_y, self.roi_w, self.roi_h)

        if roi_info is not None:
            roi_frame, rx1, ry1, rx2, ry2 = roi_info
            roi_w_px = max(1, rx2 - rx1)
            roi_h_px = max(1, ry2 - ry1)

            if self.display_roi_only:
                display_frame = cv2.resize(
                    roi_frame,
                    (original_frame.shape[1], original_frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                detection_frame = roi_frame
                map_scale_x = float(display_frame.shape[1]) / float(roi_w_px)
                map_scale_y = float(display_frame.shape[0]) / float(roi_h_px)
                map_offset_x = 0.0
                map_offset_y = 0.0
            else:
                display_frame = original_frame
                if self.detect_zoom_only:
                    detection_frame = roi_frame
                    map_scale_x = 1.0
                    map_scale_y = 1.0
                    map_offset_x = float(rx1)
                    map_offset_y = float(ry1)

        elif self.zoom_enabled:
            display_frame = self._build_zoomed_frame(original_frame)
            detection_frame = display_frame

        self.frame_height, self.frame_width = display_frame.shape[:2]
        self.last_status_message = "Streaming ativo."
        try:
            results = self.model.track(
                source=detection_frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                classes=self.allowed_class_ids,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )
        except Exception:
            # Evita quebrar a stream caso o rastreador/inferencia falhe momentaneamente.
            self.last_status_message = "Erro temporario na inferencia; tentando recuperar..."
            self.last_inference_ok = False
            self.last_inference_error = "inference_exception"
            cv2.putText(
                display_frame,
                self.last_status_message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (80, 80, 255),
                2,
            )
            return display_frame
        self.last_inference_ok = True
        self.last_inference_error = ""
        self.last_inference_ts = time.time()

        result = results[0]
        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            track_ids = result.boxes.id.cpu().numpy().astype(int)

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                try:
                    x1, y1, x2, y2 = [float(v) for v in box]
                    x1 = (x1 * map_scale_x) + map_offset_x
                    y1 = (y1 * map_scale_y) + map_offset_y
                    x2 = (x2 * map_scale_x) + map_offset_x
                    y2 = (y2 * map_scale_y) + map_offset_y
                    class_id = int(class_id)
                    track_id = int(track_id)

                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    if isinstance(self.class_names, dict):
                        class_name = str(self.class_names.get(class_id, f"class_{class_id}"))
                    else:
                        if 0 <= class_id < len(self.class_names):
                            class_name = str(self.class_names[class_id])
                        else:
                            class_name = f"class_{class_id}"
                    color = self._color_for_class(class_id)
                    self._update_tracking_and_count(
                        class_name,
                        track_id,
                        center_x,
                        center_y,
                        frame=display_frame,
                        box=(x1, y1, x2, y2),
                    )

                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{class_name} #{track_id}"
                    cv2.putText(
                        display_frame,
                        label,
                        (int(x1), max(25, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
                except Exception:
                    # Ignora deteccao invalida sem derrubar stream.
                    continue

        if self.roi_enabled and not self.display_roi_only:
            self._draw_roi_overlay(display_frame)
        self._draw_overlay(display_frame)
        return display_frame

    def get_jpeg_bytes(self) -> Optional[bytes]:
        with self.lock:
            frame = self.process_frame()
        if frame is None:
            frame = self._build_status_frame("Sem frame disponivel.")
        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return jpeg.tobytes()

    def get_counts_snapshot(self) -> Dict[str, int]:
        with self.lock:
            return dict(sorted(self.class_counts.items(), key=lambda item: item[1], reverse=True))

    def get_line_config(self) -> Dict[str, float | str]:
        with self.lock:
            return {
                "orientation": self.line_orientation,
                "position_ratio": self.line_position_ratio,
            }

    def set_line_config(self, orientation: str, position_ratio: float):
        normalized = orientation.strip().lower()
        if normalized not in {"horizontal", "vertical"}:
            raise ValueError("Orientacao invalida")
        if not (0.05 <= position_ratio <= 0.95):
            raise ValueError("Posicao deve estar entre 5 e 95")

        with self.lock:
            self.line_orientation = normalized
            self.line_position_ratio = position_ratio
            self.id_seen_frames.clear()
            self.id_last_center.clear()
            self.id_counted.clear()
            self.id_logged_detection.clear()

    def get_source_label(self) -> str:
        return self.source_selector_value

    def set_source(self, source: str) -> bool:
        parsed_source, parsed_backend, selector_value = self._decode_source(source)
        with self.lock:
            old_source = self.source
            old_backend = self.source_backend
            old_selector = self.source_selector_value
            if self.capture:
                self.capture.release()

            candidate = self._open_capture(parsed_source, parsed_backend)
            if not candidate.isOpened():
                candidate.release()
                self.source = old_source
                self.source_backend = old_backend
                self.source_selector_value = old_selector
                self.capture = self._open_capture(old_source, old_backend)
                return False

            self.source = parsed_source
            self.source_backend = parsed_backend
            self.source_selector_value = selector_value
            self.capture = candidate
            self._apply_capture_settings(self.capture)
            self.id_seen_frames.clear()
            self.id_last_center.clear()
            self.id_counted.clear()
            self.id_logged_detection.clear()
            return True

    def stop(self):
        self.running = False
        if self.capture:
            self.capture.release()
        self.db.stop()
