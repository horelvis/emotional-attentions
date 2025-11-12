from __future__ import annotations

import os
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
try:
    from openai import OpenAI
except (ImportError, ModuleNotFoundError):  # Compatibilidad con SDK < 1.0
    OpenAI = None
    try:
        import openai as openai_legacy  # type: ignore
    except ImportError:  # pragma: no cover - manejamos ausencia total del paquete
        openai_legacy = None
else:
    openai_legacy = None
from pydantic import BaseModel

from .infer import (
    DEFAULT_PROMPT,
    emo_alignment_score,
    generate,
    load_model,
)


class ChatRequest(BaseModel):
    message: Optional[str] = None
    max_new: int = 60
    k: int = 5
    temp: float = 0.8
    ema_alpha: float = 0.1


class ChatResponse(BaseModel):
    input: str
    emo_output: str
    emo_alignment: float
    chatgpt_output: Optional[str]
    chatgpt_model: Optional[str]
    warnings: list[str]


def build_openai_client() -> Optional[object]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    # Construimos el cliente solo si contamos con credenciales válidas.
    if OpenAI is not None:
        return OpenAI(api_key=api_key)
    if openai_legacy is not None:
        openai_legacy.api_key = api_key  # type: ignore[attr-defined]
        return openai_legacy
    return None


app = FastAPI(
    title="Emotional Attention Chat API",
    version="1.0.0",
    description="API para comparar respuestas del modelo emocional y ChatGPT.",
)

# Configuración CORS (permite personalizar orígenes mediante EMO_CORS_ORIGINS separados por comas).
allowed_origins = os.environ.get("EMO_CORS_ORIGINS", "")
if allowed_origins:
    origins = [origin.strip() for origin in allowed_origins.split(",") if origin.strip()]
else:
    origins = ["*"]
allow_credentials = origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.environ.get("EMO_CHECKPOINT", "artifacts/emo_decoder.pt")
    tokenizer_dir = os.environ.get("EMO_TOKENIZER_DIR", "artifacts/tokenizer")
    try:
        model, tokenizer, special_ids = load_model(checkpoint_path, tokenizer_dir, device)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No se encontró el checkpoint o tokenizer. Ejecuta el entrenamiento primero."
        ) from exc

    app.state.device = device
    app.state.model = model
    app.state.tokenizer = tokenizer
    app.state.special_ids = special_ids
    app.state.openai_client = build_openai_client()
    app.state.openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/compare", response_model=ChatResponse)
async def compare_chat(req: ChatRequest):
    device = app.state.device
    model = app.state.model
    tokenizer = app.state.tokenizer
    special_ids = app.state.special_ids

    user_text = (req.message or "").strip() or DEFAULT_PROMPT

    gen_ids, emotive_output = generate(
        model=model,
        tokenizer=tokenizer,
        special_ids=special_ids,
        prompt_text=user_text,
        device=device,
        max_new=req.max_new,
        k=req.k,
        temp=req.temp,
        ema_alpha=req.ema_alpha,
        # En modo inferencia limitamos el máximo total para prevenir desbordes.
        max_length=min(req.max_new + 256, 512),
    )
    alignment = emo_alignment_score(model, special_ids, tokenizer, user_text, gen_ids, device)

    warnings: list[str] = []
    chatgpt_text: Optional[str] = None
    chatgpt_model: Optional[str] = None

    client: Optional[object] = getattr(app.state, "openai_client", None)
    if client is None:
        warnings.append("OPENAI_API_KEY no está configurada; la comparación con ChatGPT está deshabilitada.")
    else:
        system_prompt = (
            "Eres un asistente empático. Responde de forma breve y útil, "
            "considerando el tono emocional del usuario."
        )
        try:
            # Ejecutamos la llamada bloqueante en un hilo para no frenar el loop de FastAPI.
            if hasattr(client, "responses"):
                response = await run_in_threadpool(
                    client.responses.create,
                    model=app.state.openai_model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                )
                chatgpt_model = getattr(response, "model", None)
                try:
                    chatgpt_text = response.output_text  # type: ignore[attr-defined]
                except AttributeError:
                    chatgpt_text = None
                    warnings.append("No se pudo extraer la respuesta de ChatGPT.")
            elif hasattr(client, "ChatCompletion"):
                response = await run_in_threadpool(
                    client.ChatCompletion.create,  # type: ignore[attr-defined]
                    model=app.state.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    temperature=req.temp,
                )
                chatgpt_model = response.get("model") if isinstance(response, dict) else None
                try:
                    choices = response.get("choices", [])  # type: ignore[union-attr]
                    chatgpt_text = choices[0]["message"]["content"] if choices else None
                except (KeyError, IndexError, TypeError):
                    chatgpt_text = None
                    warnings.append("No se pudo extraer la respuesta de ChatGPT.")
            else:
                warnings.append("La versión del SDK de OpenAI no es compatible.")
        except Exception as exc:  # noqa: BLE001 - capturamos errores de la librería OpenAI
            warnings.append(f"Error al consultar ChatGPT: {exc.__class__.__name__}")

    return ChatResponse(
        input=user_text,
        emo_output=emotive_output,
        emo_alignment=alignment,
        chatgpt_output=chatgpt_text,
        chatgpt_model=chatgpt_model,
        warnings=warnings,
    )
