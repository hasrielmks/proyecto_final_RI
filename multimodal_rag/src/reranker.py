"""
src/reranker.py
================
Funcionalidad de excelencia: Re-ranking (+15).

La búsqueda vectorial inicial (FAISS + CLIP) prioriza recall: trae
config.TOP_K_RETRIEVAL candidatos con buena similitud semántica/visual
general, pero CLIP no está optimizado para comparar la relevancia fina
de una consulta de texto contra un documento de texto largo (es un
modelo de propósito general texto-imagen, con un encoder de texto
limitado a 77 tokens).

Para refinar el orden final usamos un *cross-encoder* (MS MARCO
MiniLM), que sí procesa consulta+documento juntos y produce un score
de relevancia mucho más preciso -- a costa de ser más lento, por eso
solo se aplica sobre los top-K candidatos de FAISS, no sobre todo el
corpus (patrón estándar "retrieve-then-rerank").

El score final combina (con pesos configurables) el score del
cross-encoder con el score original de similitud vectorial, así el
reranking no ignora por completo la señal visual/CLIP.
"""

from __future__ import annotations

import config

_cross_encoder = None

# Peso del score del cross-encoder vs. el score original de FAISS/CLIP
# en la combinación final.
_CE_WEIGHT = 0.75
_VECTOR_WEIGHT = 0.25


def _lazy_load():
    global _cross_encoder
    if _cross_encoder is not None:
        return
    from sentence_transformers import CrossEncoder

    _cross_encoder = CrossEncoder(config.CROSS_ENCODER_MODEL)


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rerank(query: str, candidates: list[dict], top_k: int = None) -> list[dict]:
    """Reordena `candidates` (salida de VectorStore.search) usando un
    cross-encoder texto-texto sobre (query, texto_del_documento).

    Cada candidato debe tener una clave "context_text" o "title" con el
    texto a comparar. Devuelve una nueva lista, ordenada, con una clave
    adicional "rerank_score" y "vector_score" (renombrado del "score"
    original) para que la UI pueda mostrar ambas señales.
    """
    top_k = top_k or config.TOP_K_RERANK
    if not candidates:
        return []

    _lazy_load()
    pairs = [
        (query, c.get("context_text") or c.get("title") or "") for c in candidates
    ]
    ce_scores = _cross_encoder.predict(pairs).tolist()
    ce_scores_norm = _min_max_normalize(ce_scores)
    vector_scores_norm = _min_max_normalize([c.get("score", 0.0) for c in candidates])

    reranked = []
    for c, ce_raw, ce_norm, v_norm in zip(
        candidates, ce_scores, ce_scores_norm, vector_scores_norm
    ):
        item = dict(c)
        item["vector_score"] = item.pop("score", 0.0)
        item["rerank_score"] = float(ce_raw)
        item["final_score"] = _CE_WEIGHT * ce_norm + _VECTOR_WEIGHT * v_norm
        reranked.append(item)

    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    for i, item in enumerate(reranked):
        item["rank"] = i + 1
    return reranked[:top_k]


if __name__ == "__main__":
    query = "orquesta clásica de Viena"
    candidates = [
        {
            "doc_id": "a",
            "title": "Viennese Waltzes",
            "context_text": "Título: Viennese Waltzes\nArtista: Vienna Philharmonic Orchestra",
            "score": 0.40,
        },
        {
            "doc_id": "b",
            "title": "Death Metal Anthology",
            "context_text": "Título: Death Metal Anthology\nArtista: Grinding Skulls",
            "score": 0.42,
        },
        {
            "doc_id": "c",
            "title": "Vienna Symphony Live",
            "context_text": "Título: Vienna Symphony Live\nArtista: Vienna Symphony Orchestra",
            "score": 0.38,
        },
    ]
    results = rerank(query, candidates, top_k=3)
    for r in results:
        print(r["rank"], r["doc_id"], round(r["final_score"], 3), round(r["rerank_score"], 3))
