from __future__ import annotations

import queue
import threading
import uuid

import cv2
from django.core.files.base import ContentFile
from django.utils import timezone

from detector.models import DetectionLog


class DatabaseManager:
    """Gerencia persistencia dos logs sem bloquear a stream de video."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue(maxsize=300)
        self._running = True
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    @staticmethod
    def _encode_jpeg(image):
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            return None
        return encoded.tobytes()

    def enqueue_detection(self, objeto_tipo: str, count_total: int, frame_full=None, frame_crop=None) -> None:
        try:
            self._queue.put_nowait((objeto_tipo, count_total, frame_full, frame_crop))
        except queue.Full:
            # Em carga alta, preferimos descartar log do que travar stream.
            pass

    def _worker_loop(self):
        while self._running:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            objeto_tipo, count_total, frame_full, frame_crop = item
            try:
                self._save_detection(objeto_tipo, count_total, frame_full, frame_crop)
            finally:
                self._queue.task_done()

    def _save_detection(self, objeto_tipo: str, count_total: int, frame_full=None, frame_crop=None) -> DetectionLog:
        detection = DetectionLog.objects.create(
            objeto_tipo=objeto_tipo,
            count_total=count_total,
        )

        timestamp_tag = timezone.now().strftime("%Y%m%d_%H%M%S_%f")
        unique = uuid.uuid4().hex[:8]

        if frame_full is not None:
            payload = self._encode_jpeg(frame_full)
            if payload:
                filename = f"{objeto_tipo}_{count_total}_{timestamp_tag}_{unique}_full.jpg"
                detection.imagem_full.save(filename, ContentFile(payload), save=False)

        if frame_crop is not None and getattr(frame_crop, "size", 0) > 0:
            payload = self._encode_jpeg(frame_crop)
            if payload:
                filename = f"{objeto_tipo}_{count_total}_{timestamp_tag}_{unique}_crop.jpg"
                detection.imagem_crop.save(filename, ContentFile(payload), save=False)

        detection.save()
        return detection

    def stop(self):
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._worker.is_alive():
            self._worker.join(timeout=1.0)
