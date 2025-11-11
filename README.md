# Emotional Attention Docker Project

Proyecto Python empaquetado para entrenar y evaluar el decoder con adapters dual-head basados en la rueda de emociones de Plutchik.

## Estructura
- `src/emotion_attention/`: código fuente (datos, modelo, utilidades, train/infer).
- `scripts/`: wrappers para ejecutar entrenamiento (`run_train.sh`) e inferencia (`run_infer.sh`).
- `requirements.txt`, `Dockerfile`, `docker-compose.yml`.

## Uso rápido
1. Construir imagen:
   ```bash
   docker compose build
   ```
2. Entrenar adapters (con base congelada + distillation opcional):
   ```bash
   docker compose run --rm emo-trainer --epochs 80 --lambda-prop 1.0
   ```
   El checkpoint se guarda en `artifacts/emo_decoder.pt` (creado automáticamente).
3. Inferir y medir alineación:
   ```bash
   docker compose run --rm emo-infer --input "estoy nerviosa por manana"
   ```

Personaliza argumentos extra pasándolos tras el comando en `docker compose run`. Para GPU cambia la imagen base en el `Dockerfile` (p. ej. `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`) y ejecuta con `--gpus all`.
