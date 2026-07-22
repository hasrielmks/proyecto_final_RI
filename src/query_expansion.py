"""
src/query_expansion.py
=======================
Funcionalidad de excelencia: Query Expansion (+15).

Objetivo: mejorar el recall de la búsqueda vectorial reformulando/ampliando
la consulta original del usuario antes de generar su embedding CLIP.

Dos modos (config.QUERY_EXPANSION_MODE):

  - "llm": usa la Gemini API para generar términos relacionados
    (sinónimos, artistas similares, géneros asociados, variantes de
    escritura). Es el modo por defecto y el que mejor funciona en un
    dominio como catálogo musical.

  - "heuristic": expansión sin LLM (para poder correr el sistema sin
    API key), basada en un pequeño diccionario de sinónimos/géneros
    musicales comunes + normalización simple.

En ambos casos la función devuelve la *consulta expandida* (texto),
que luego se embebe con CLIP igual que cualquier consulta normal. Esto
mantiene el resto del pipeline (retrieval, rerank, RAG) sin cambios.
"""

from __future__ import annotations

import re

import config
from src import generator

# Pequeño diccionario de dominio musical para el modo heurístico
# (fallback sin necesidad de API key).
_MUSIC_SYNONYMS = {
    "rock": ["rock and roll", "hard rock", "classic rock"],
    "jazz": ["swing", "bebop", "smooth jazz"],
    "clasica": ["clásica", "orquesta", "sinfonía", "sinfonia"],
    "clásica": ["orquesta", "sinfonía", "concierto"],
    "pop": ["pop music", "top 40"],
    "blues": ["rhythm and blues", "r&b"],
    "electronica": ["electrónica", "dance", "edm"],
    "electrónica": ["dance", "edm", "techno"],
    "country": ["folk", "americana"],
    "navidad": ["christmas", "holiday", "villancicos"],
    "christmas": ["navidad", "holiday songs"],
    "soundtrack": ["banda sonora", "score", "película"],
    "reggae": ["ska", "dub"],
    "metal": ["heavy metal", "hard rock"],
    "gospel": ["música cristiana", "coral"],
}


def _heuristic_expand(query: str) -> str:
    q_lower = query.lower()
    extra_terms = []
    for key, synonyms in _MUSIC_SYNONYMS.items():
        if re.search(rf"\b{re.escape(key)}\b", q_lower):
            extra_terms.extend(synonyms)
    if not extra_terms:
        return query
    extra_terms = extra_terms[: config.QUERY_EXPANSION_MAX_TERMS]
    return f"{query} ({', '.join(extra_terms)})"


def _llm_expand(query: str) -> str:
    prompt = (
        "Eres un asistente de recuperación de información especializado en "
        "un catálogo de música (álbumes, artistas, sellos discográficos). "
        f"Dada la siguiente consulta de un usuario: \"{query}\"\n\n"
        f"Genera hasta {config.QUERY_EXPANSION_MAX_TERMS} términos o frases "
        "cortas relacionadas (sinónimos, géneros musicales asociados, "
        "artistas similares, variantes de escritura) que ayuden a encontrar "
        "más documentos relevantes en una búsqueda vectorial. "
        "Responde SOLO con los términos separados por comas, sin explicaciones, "
        "sin numerarlos, y sin repetir la consulta original."
    )
    try:
        expansion = generator.generate_raw(prompt, max_output_tokens=100, temperature=0.4)
        expansion = expansion.strip().strip(".")
        if not expansion:
            return query
        return f"{query} ({expansion})"
    except Exception:
        # Si la API falla (sin conexión, sin API key, cuota agotada, etc.)
        # degradamos con gracia al modo heurístico en vez de romper la búsqueda.
        return _heuristic_expand(query)


def expand_query(query: str, mode: str = None) -> str:
    """Devuelve la consulta expandida según el modo configurado."""
    mode = mode or config.QUERY_EXPANSION_MODE
    query = query.strip()
    if not query:
        return query
    if mode == "llm":
        return _llm_expand(query)
    return _heuristic_expand(query)


if __name__ == "__main__":
    for q in ["quiero algo de rock clásico", "música para navidad", "grateful dead"]:
        print(f"[{q}] -> {_heuristic_expand(q)}")
