# Emotional Attention Docker Project

Proyecto Python empaquetado para entrenar y evaluar un decoder con adapters dual-head sobre diálogos emocionales multi-turno provenientes, por defecto, de `empathetic_dialogues` (Hugging Face). El pipeline usa un tokenizer BPE (`distilroberta-base` por defecto) y carga conversaciones reales para que el modelo aprenda a responder según el tono detectado en el historial del interlocutor.

## Estructura
- `src/emotion_attention/data.py`: descarga `empathetic_dialogues` (o `daily_dialog` si se indica), prepara ejemplos multi-turno con etiquetas emocionales y construye `DataLoader`s.
- `src/emotion_attention/model.py`: decoder base + adapters dual-head con compuerta emocional.
- `src/emotion_attention/train.py`: entrena sólo los adapters (y opcional distillation) con tokenizer HF.
- `src/emotion_attention/infer.py`: genera respuestas condicionadas y reporta `emo_alignment_score`.
- `scripts/`: wrappers para lanzar entrenamiento (`run_train.sh`) e inferencia (`run_infer.sh`).

## Uso rápido
1. Construir imagen (instala `torch`, `transformers`, `datasets`, etc.):
   ```bash
   docker compose build
   ```
2. Entrenar adapters sobre `empathetic_dialogues` (history de 3 turnos, tokenizer `distilroberta-base` guardado en `artifacts/tokenizer`):
   ```bash
   docker compose run --rm emo-trainer ./scripts/run_train.sh --epochs 5 --batch-size 24 --history-turns 4
   ```
   El checkpoint queda en `artifacts/emo_decoder.pt` y el tokenizer ajustado en `artifacts/tokenizer/`.
3. Inferir en modo chat + coseno emocional:
   ```bash
   docker compose run --rm emo-infer --input "ayer discutí con mi hermano y estoy ansiosa"
   ```

Puedes sobre-escribir cualquier flag del CLI añadiéndolo tras `./scripts/run_train.sh` (p. ej. `docker compose run --rm emo-trainer ./scripts/run_train.sh --tokenizer-name gpt2 --max-length 192 --dataset-name empathetic_dialogues`). Si cuentas con una copia local de `daily_dialog`, también puedes apuntar al loader original pasando `--dataset-name daily_dialog`. Para usar GPU cambia la imagen base del `Dockerfile` a una variante CUDA y ejecuta `docker compose run --rm --gpus all ...`.

## API & UI React

Para comparar el modelo emocional con ChatGPT se añadió:

- `scripts/run_api.sh` + servicio `emo-api` en `docker-compose.yml`: expone FastAPI en `http://localhost:8000`.
- `web/`: interfaz Vite+React con React Query y Tailwind. Permite enviar un historial, ver la respuesta emocional, la alineación y la respuesta paralela de ChatGPT.

Pasos:

1. Levantar la API (requiere haber entrenado y tener `artifacts/emo_decoder.pt`):
   ```bash
   docker compose up emo-api
   ```
   Opcional: define `OPENAI_API_KEY` y `OPENAI_MODEL` en `.env` para activar la comparación con ChatGPT.

2. Preparar el frontend:
   ```bash
   cd web
   npm install
   npm run dev
   ```
   El dev server se abre en `http://localhost:5173` (configurable con `VITE_DEV_SERVER_PORT`). Puedes fijar `VITE_API_BASE_URL` en un `.env` dentro de `web/` si la API corre en otra URL.

Cada turno enviado queda registrado con ambas respuestas para analizar diferencias.
