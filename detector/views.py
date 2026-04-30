import atexit
import time

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from detector.models import DetectionLog
from detector.services.video_processor import VideoProcessor

_processor = VideoProcessor(source=settings.VIDEO_SOURCE)
atexit.register(_processor.stop)


@login_required
def index(request):
    return render(request, "detector/index.html")


@login_required
def video_feed(request):
    def frame_generator():
        while True:
            try:
                frame = _processor.get_jpeg_bytes()
            except Exception:
                time.sleep(0.05)
                continue
            if frame is None:
                time.sleep(0.02)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return StreamingHttpResponse(
        frame_generator(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )

@login_required
def frame_api(request):
    try:
        frame = _processor.get_jpeg_bytes()
    except Exception:
        return HttpResponse(status=503)
    if frame is None:
        return HttpResponse(status=503)

    response = HttpResponse(frame, content_type="image/jpeg")
    response["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response["Pragma"] = "no-cache"
    response["Expires"] = "0"
    return response


@login_required
def counters_api(request):
    return JsonResponse(_processor.get_counts_snapshot())

@login_required
def line_config_api(request):
    return JsonResponse(_processor.get_line_config())

@login_required
def model_config_api(request):
    return JsonResponse(_processor.get_runtime_config())


@login_required
def hf_models_api(request):
    query = request.GET.get("q", "yolo")
    limit_raw = request.GET.get("limit", "20")
    try:
        limit = int(limit_raw)
    except ValueError:
        limit = 20

    models, warning = VideoProcessor.list_hf_detection_models(query=query, limit=limit)
    payload = {"models": models}
    if warning:
        payload["warning"] = warning
    return JsonResponse(payload)


@login_required
def list_sources_api(request):
    webcams = VideoProcessor.discover_webcams()
    current = _processor.get_source_label()
    if not webcams:
        webcams = [
            {"value": "cam:0:any", "label": "Webcam 0 (manual)"},
            {"value": "cam:1:any", "label": "Webcam 1 (manual)"},
            {"value": "cam:2:any", "label": "Webcam 2 (manual)"},
            {"value": "cam:3:any", "label": "Webcam 3 (manual)"},
            {"value": "cam:4:any", "label": "Webcam 4 (manual)"},
        ]

    if current and not any(cam["value"] == current for cam in webcams):
        label = "Fonte atual"
        if current.startswith("cam:"):
            label = "Webcam atual"
        webcams.insert(0, {"value": current, "label": f"{label} ({current})"})

    sources = webcams + [{"value": "rtsp://usuario:senha@ip:porta/stream", "label": "RTSP (editar URL no campo)"}]
    return JsonResponse({"sources": sources, "current_source": current})


@login_required
@require_POST
def select_source_api(request):
    source = request.POST.get("source", "").strip()
    if not source:
        return JsonResponse({"error": "Fonte invalida"}, status=400)
    changed = _processor.set_source(source)
    if not changed:
        return JsonResponse({"error": "Nao foi possivel abrir a fonte selecionada"}, status=400)
    return JsonResponse({"ok": True, "current_source": _processor.get_source_label()})


@login_required
@require_POST
def set_line_config_api(request):
    orientation = request.POST.get("orientation", "").strip().lower()
    position_raw = request.POST.get("position_percent", "").strip()
    try:
        position_percent = float(position_raw)
    except ValueError:
        return JsonResponse({"error": "Posicao invalida"}, status=400)

    try:
        _processor.set_line_config(orientation, position_percent / 100.0)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    return JsonResponse({"ok": True, "line_config": _processor.get_line_config()})


@login_required
@require_POST
def set_model_config_api(request):
    try:
        config = _processor.set_runtime_config(request.POST)
        return JsonResponse({"ok": True, "config": config})
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        return JsonResponse({"error": "Falha interna ao aplicar configuracao do modelo."}, status=500)


@login_required
def detection_logs(request):
    logs = DetectionLog.objects.all().order_by("-timestamp")
    paginator = Paginator(logs, 20)
    page_obj = paginator.get_page(request.GET.get("page"))
    return render(request, "detector/detection_logs.html", {"page_obj": page_obj})
