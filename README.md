# VisionQQ - Django + YOLOv8 + Tracking

Aplicacao de deteccao em tempo real com webcam/RTSP usando Django como interface, YOLOv8 para deteccao e ByteTrack para evitar duplicidade de contagem.

## Requisitos

- Python 3.10+
- Webcam local (ou URL RTSP)

## Instalacao

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Observacao: a listagem de webcams usa `cv2-enumerate-cameras` para obter nomes e backends com mais confiabilidade.
Para usar modelo vindo do Hugging Face, o projeto usa `huggingface-hub`.

## Configuracao por variaveis de ambiente

Copie o arquivo `.env.example` para `.env` (ou configure as variaveis no seu sistema):

```bash
DJANGO_SECRET_KEY=change-me-in-production
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost
VIDEO_SOURCE=cam:0:any
```

## Configuracao opcional de fonte de video

PowerShell (webcam padrao):

```powershell
$env:VIDEO_SOURCE="0"
```

PowerShell (RTSP):

```powershell
$env:VIDEO_SOURCE="rtsp://usuario:senha@ip:porta/stream"
```

## Banco de dados (SQLite)

```bash
python manage.py migrate
python manage.py createsuperuser
```

Observacao: apos esta versao, cada log salva duas imagens em `media/`:
- frame completo do evento
- recorte do objeto detectado

## Rodar interface Django com feed e contadores

```bash
python manage.py runserver
```

Acesse:

- http://127.0.0.1:8000/login/
- http://127.0.0.1:8000/admin/

Depois do login, use:
- Dashboard: `http://127.0.0.1:8000/`
- Registros de deteccoes: `http://127.0.0.1:8000/registros/`

Na pagina principal, use o seletor de fonte para trocar entre webcams detectadas e RTSP.
Os contadores e registros sao dinamicos para todas as classes detectadas pelo modelo.
Tambem e possivel configurar a linha de contagem pela interface:
- orientacao (`horizontal` ou `vertical`)
- posicao da linha (slider em percentual da imagem)

## Ajustes avancados no dashboard

Pelo painel da interface voce pode:
- escolher provider de modelo (`Ultralytics` ou `Hugging Face`)
- buscar e selecionar modelos de deteccao no Hugging Face diretamente pela UI
- carregar modelo do Hugging Face por `repo_id` (+ `filename .pt` opcional)
- ajustar `confidence`, `iou`, `imgsz` e `stable_frames`
- filtrar classes (ex.: `car,truck,bus,motorcycle`)
- usar preset rapido para veiculos
- ajustar FPS de camera (tentativa) e FPS de render no navegador
- ajustar resolucao da camera (largura/altura)

## Rodar modo OpenCV com tecla `q` para encerrar

```bash
python manage.py run_webcam_detector
```

## Estrutura principal

- `detector/services/db_manager.py`: gerenciamento de persistencia
- `detector/services/video_processor.py`: deteccao, tracking e contagem
- `detector/models.py`: tabela `DetectionLog` (`id`, `objeto_tipo`, `timestamp`, `count_total`)
- `detector/views.py`: stream MJPEG e API de contadores

## Preparado para GitHub

Este projeto ja inclui `.gitignore` para evitar subir arquivos locais/sensiveis:
- `.venv/`
- `db.sqlite3`
- pesos de modelo (`*.pt`)
- `.env`

## Comandos para publicar no GitHub

```bash
git init
git add .
git commit -m "feat: VisionQQ com Django + YOLOv8 + tracking"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPO.git
git push -u origin main
```
