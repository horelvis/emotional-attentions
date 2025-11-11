# Codex Context

- Repo: emotional-attentions (https://github.com/horelvis/emotional-attentions.git)
- Stack: Python package + Docker. Source lives in `src/emotion_attention/`.
- Data: Loads Hugging Face `daily_dialog` (multi-turn) via `datasets`. Tokenizer defaults to `distilroberta-base`, extended with special tokens and stored under `artifacts/tokenizer/`.
- Model: `EmoDecoder` with frozen base blocks and Dual-Head Emotional Attention adapters; optional distillation from `pysentimiento/robertuito-sentiment-analysis`.
- Training entry point: `python -m emotion_attention.train` (wrapper `scripts/run_train.sh`, Docker service `emo-trainer`). Saves checkpoint `artifacts/emo_decoder.pt` (contains config + special token ids).
- Inference entry point: `python -m emotion_attention.infer --input "..."` (wrapper `scripts/run_infer.sh`, Docker service `emo-infer`). Returns generated response + `emo_alignment_score`.
- Docker: CPU image based on `python:3.11-slim`, Compose services for train/infer. For GPU swap to PyTorch CUDA image and run with `--gpus all`.
- Pending: need to actually run training (downloads `daily_dialog`, teacher model) and validate outputs; LoRA integration optional.
