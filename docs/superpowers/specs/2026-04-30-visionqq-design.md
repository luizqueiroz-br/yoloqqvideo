# Design - VisionQQ Object Detection

## Objetivo
Construir uma aplicacao Django funcional com deteccao em tempo real para `car`, `dog`, `cat`, `person`, com tracking para contagem unica e persistencia em banco relacional.

## Arquitetura
- Django como interface e roteamento.
- `VideoProcessor` para captura, inferencia YOLOv8 e tracking ByteTrack.
- `DatabaseManager` para persistir logs de contagem.
- SQLite como banco inicial.

## Fluxo
1. Inicializa modelo YOLO e conexao de banco.
2. Captura frame de webcam/RTSP.
3. Detecta classes alvo e aplica tracking com IDs persistentes.
4. Aplica regra hibrida de contagem:
   - ID estavel por `N` frames.
   - Cruzamento de linha virtual.
   - ID ainda nao contado.
5. Incrementa contador e salva log em banco.
6. Exibe frame com bounding boxes e contadores no Django.
7. Finaliza limpo em modo OpenCV com tecla `q`.

## Dados
Tabela `DetectionLog`:
- `id`
- `objeto_tipo`
- `timestamp`
- `count_total`

## Riscos e mitigacao
- Duplicidade: mitigada por tracking + estabilidade + linha virtual.
- Queda de stream: tentativa de reconexao da captura.
- CPU limitada: uso do modelo `yolov8n.pt` e threshold de confianca configuravel.
