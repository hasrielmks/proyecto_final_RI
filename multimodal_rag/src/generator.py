"""
src/generator.py
=================
Wrapper sobre la Gemini API (SDK oficial `google-genai`, NO el paquete
deprecado `google.generativeai`) para el paso de generación del RAG y
para la expansión de consultas basada en LLM.

Requiere la variable de entorno GEMINI_API_KEY (obtenerla gratis en
https://aistudio.google.com/apikey). Sin ella, `generate_raw` levanta
una excepción clara que el resto del sistema captura y degrada con
gracia (ver src/query_expansion.py y src/rag_pipeline.py).
"""

from __future__ import annotations

import os

import config

_client = None


class GeminiNotConfiguredError(RuntimeError):
    pass


def _lazy_client():
    global _client
    if _client is not None:
        return _client
    api_key = os.environ.get(config.GEMINI_API_KEY_ENV_VAR)
    if not api_key:
        raise GeminiNotConfiguredError(
            f"No se encontró la variable de entorno {config.GEMINI_API_KEY_ENV_VAR}. "
            "Obtené una API key gratis en https://aistudio.google.com/apikey y "
            f"exportala como {config.GEMINI_API_KEY_ENV_VAR}=tu_api_key"
        )
    from google import genai

    _client = genai.Client(api_key=api_key)
    return _client


def generate_raw(
    prompt: str,
    system_instruction: str | None = None,
    max_output_tokens: int = None,
    temperature: float = None,
    model: str = None,
) -> str:
    """Llamada simple de generación de texto a la Gemini API. Devuelve texto plano."""
    from google.genai import types

    client = _lazy_client()
    cfg = types.GenerateContentConfig(
        max_output_tokens=max_output_tokens or config.GENERATION_MAX_OUTPUT_TOKENS,
        temperature=(
            config.GENERATION_TEMPERATURE if temperature is None else temperature
        ),
        system_instruction=system_instruction,
    )
    response = client.models.generate_content(
        model=model or config.GEMINI_MODEL,
        contents=prompt,
        config=cfg,
    )
    return (response.text or "").strip()


def generate_rag_answer(query: str, context_blocks: list[str]) -> str:
    """Genera la respuesta final del RAG a partir de la consulta y el
    contexto recuperado (lista de bloques de texto, uno por documento)."""
    numbered_context = "\n\n".join(
        f"[Documento {i + 1}]\n{block}" for i, block in enumerate(context_blocks)
    )
    prompt = (
        f"Contexto recuperado del catálogo musical:\n\n{numbered_context}\n\n"
        f"Pregunta del usuario: {query}\n\n"
        "Responde la pregunta usando solo la información del contexto anterior. "
        "Si citas un documento específico, referenciá su título."
    )
    return generate_raw(prompt, system_instruction=config.RAG_SYSTEM_PROMPT)


if __name__ == "__main__":
    try:
        print(generate_raw("Responde solo con la palabra: OK"))
    except GeminiNotConfiguredError as e:
        print("Config pendiente (esperado si no hay API key en este entorno):", e)
