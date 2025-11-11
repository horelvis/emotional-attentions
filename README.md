# Emotional Attention Docker Project

Proyecto Python empaquetado para entrenar y evaluar un decoder con adapters dual-head sobre diálogos emocionales multi-turno provenientes de `daily_dialog` (Hugging Face). El pipeline usa un tokenizer BPE (`distilroberta-base` por defecto) y carga conversaciones reales para que el modelo aprenda a responder según el tono detectado en el historial del interlocutor.

## Estructura
- `src/emotion_attention/data.py`: descarga `daily_dialog`, prepara ejemplos multi-turno con etiquetas emocionales y construye `DataLoader`s.
- `src/emotion_attention/model.py`: decoder base + adapters dual-head con compuerta emocional.
- `src/emotion_attention/train.py`: entrena sólo los adapters (y opcional distillation) con tokenizer HF.
- `src/emotion_attention/infer.py`: genera respuestas condicionadas y reporta `emo_alignment_score`.
- `scripts/`: wrappers para lanzar entrenamiento (`run_train.sh`) e inferencia (`run_infer.sh`).

## Uso rápido
1. Construir imagen (instala `torch`, `transformers`, `datasets`, etc.):
   ```bash
   docker compose build
   ```
2. Entrenar adapters sobre `daily_dialog` (history de 3 turnos, tokenizer `distilroberta-base` guardado en `artifacts/tokenizer`):
   ```bash
   docker compose run --rm emo-trainer --epochs 5 --batch-size 24 --history-turns 4
   ```
   El checkpoint queda en `artifacts/emo_decoder.pt` y el tokenizer ajustado en `artifacts/tokenizer/`.
3. Inferir en modo chat + coseno emocional:
   ```bash
   docker compose run --rm emo-infer --input "ayer discutí con mi hermano y estoy ansiosa"
   ```

Puedes sobre-escribir cualquier flag del CLI añadiéndolo tras el comando de compose (p. ej. `--tokenizer-name gpt2 --max-length 192`). Para usar GPU cambia la imagen base del `Dockerfile` a una variante CUDA y ejecuta `docker compose run --rm --gpus all ...`.
