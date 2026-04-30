from django.conf import settings
from django.core.management.base import BaseCommand
import cv2

from detector.services.video_processor import VideoProcessor


class Command(BaseCommand):
    help = "Executa detector em janela OpenCV e encerra ao pressionar 'q'."

    def handle(self, *args, **options):
        processor = VideoProcessor(source=settings.VIDEO_SOURCE)
        self.stdout.write(self.style.SUCCESS("Detector iniciado. Pressione 'q' para sair."))
        try:
            while True:
                frame = processor.process_frame()
                if frame is None:
                    continue
                cv2.imshow("VisionQQ Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            processor.stop()
            cv2.destroyAllWindows()
            self.stdout.write(self.style.WARNING("Detector finalizado com sucesso."))
