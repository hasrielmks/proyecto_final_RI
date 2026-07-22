"""
src/rag_pipeline.py
====================
Orquesta el flujo completo de Retrieval-Augmented Generation, integrando
todos los componentes del sistema:

    memoria conversacional (condensar pregunta)
        -> query expansion (ampliar términos de búsqueda)
        -> embedding CLIP de la consulta
        -> relevance feedback / Rocchio (si hay likes/dislikes previos)
        -> búsqueda vectorial FAISS (top-K candidatos)
        -> reranking con cross-encoder (top-k final)
        -> construcción de contexto
        -> generación de respuesta con Gemini
        -> se guarda el turno en memoria

Este módulo es el que consume tanto app.py (Streamlit) como evaluate.py
(evaluación offline), para garantizar que la UI y la evaluación usan
exactamente el mismo pipeline de recuperación+generación (trazabilidad).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config
from src import embeddings, feedback as feedback_mod, generator, memory as memory_mod
from src import query_expansion, reranker, vector_store


@dataclass
class RagResponse:
    answer: str
    evidences: list[dict]              # documentos finales usados como contexto
    standalone_query: str              # consulta tras resolver memoria conversacional
    expanded_query: str                # consulta tras expansión de términos
    used_query_expansion: bool
    used_reranking: bool
    used_memory: bool
    used_feedback: bool
    generation_error: str | None = None


class RagPipeline:
    def __init__(self, store: "vector_store.VectorStore"):
        self.store = store

    def answer(
        self,
        user_query: str,
        memory: "memory_mod.ConversationMemory" | None = None,
        session_feedback: "feedback_mod.FeedbackStore" | None = None,
        use_query_expansion: bool = None,
        use_reranking: bool = None,
        top_k_retrieval: int = None,
        top_k_final: int = None,
    ) -> RagResponse:
        use_query_expansion = (
            config.QUERY_EXPANSION_ENABLED_DEFAULT
            if use_query_expansion is None
            else use_query_expansion
        )
        use_reranking = (
            config.RERANK_ENABLED_DEFAULT if use_reranking is None else use_reranking
        )
        top_k_retrieval = top_k_retrieval or config.TOP_K_RETRIEVAL
        top_k_final = top_k_final or config.TOP_K_RERANK

        # 1) Memoria conversacional: resolver referencias al historial
        used_memory = bool(memory and not memory.is_empty())
        standalone_query = (
            memory.condense_query(user_query) if memory is not None else user_query
        )

        # 2) Query expansion
        if use_query_expansion:
            expanded_query = query_expansion.expand_query(standalone_query)
        else:
            expanded_query = standalone_query

        # 3) Embedding CLIP de la consulta (siempre a partir de la
        #    consulta expandida, que es la que se busca en FAISS)
        query_vec = embeddings.embed_query(expanded_query)

        # 4) Relevance feedback (Rocchio), si hay votos previos en la sesión
        used_feedback = bool(session_feedback and session_feedback.has_feedback())
        if used_feedback:
            query_vec = feedback_mod.rocchio_update(query_vec, session_feedback)

        # 5) Búsqueda vectorial
        candidates = self.store.search(query_vec, top_k=top_k_retrieval)

        # 6) Reranking (opcional)
        if use_reranking and candidates:
            final_docs = reranker.rerank(standalone_query, candidates, top_k=top_k_final)
        else:
            final_docs = candidates[:top_k_final]
            for i, d in enumerate(final_docs):
                d["vector_score"] = d.get("score", 0.0)
                d["final_score"] = d.get("score", 0.0)
                d["rank"] = i + 1

        # 7) Construcción de contexto + generación
        context_blocks = [d.get("context_text", d.get("title", "")) for d in final_docs]
        generation_error = None
        try:
            if context_blocks:
                answer = generator.generate_rag_answer(standalone_query, context_blocks)
            else:
                answer = (
                    "No encontré documentos relevantes en el catálogo para "
                    "responder esta consulta."
                )
        except generator.GeminiNotConfiguredError as e:
            generation_error = str(e)
            answer = (
                "⚠️ No se pudo generar una respuesta porque la Gemini API no "
                "está configurada (falta GEMINI_API_KEY). Se muestran igualmente "
                "las evidencias recuperadas más abajo."
            )
        except Exception as e:  # noqa: BLE001
            generation_error = str(e)
            answer = (
                "⚠️ Ocurrió un error al generar la respuesta con el LLM "
                f"({e}). Se muestran las evidencias recuperadas más abajo."
            )

        # 8) Guardar turno en memoria conversacional
        if memory is not None:
            memory.add_turn(user_query, standalone_query, answer)

        return RagResponse(
            answer=answer,
            evidences=final_docs,
            standalone_query=standalone_query,
            expanded_query=expanded_query,
            used_query_expansion=use_query_expansion,
            used_reranking=use_reranking,
            used_memory=used_memory,
            used_feedback=used_feedback,
            generation_error=generation_error,
        )


if __name__ == "__main__":
    # Prueba de humo con un índice sintético (sin depender de FAISS real
    # en disco ni de la API de Gemini).
    import numpy as np

    rng = np.random.default_rng(0)
    dim = config.CLIP_EMBED_DIM
    store = vector_store.VectorStore(dim=dim)
    vecs = rng.normal(size=(5, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    meta = [
        {
            "doc_id": f"doc{i}",
            "title": f"Álbum de prueba {i}",
            "context_text": f"Título: Álbum de prueba {i}\nArtista: Artista {i}",
            "image_url": None,
        }
        for i in range(5)
    ]
    store.add(vecs, meta)

    pipeline = RagPipeline(store)
    # forzamos: sin expansion LLM (no hay API key), sin reranking (evita
    # descargar el cross-encoder en este smoke test offline)
    resp = pipeline.answer(
        "recomiéndame un álbum",
        use_query_expansion=False,
        use_reranking=False,
    )
    print("Respuesta:", resp.answer[:200])
    print("N evidencias:", len(resp.evidences))
    print("used_memory:", resp.used_memory, "used_feedback:", resp.used_feedback)
