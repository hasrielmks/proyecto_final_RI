"""
src/feedback.py
================
Funcionalidad de excelencia: Relevance Feedback (+15).

Permite que el usuario marque documentos recuperados como "Me gusta" /
"No me gusta" (like/dislike) y usa esa señal para mejorar búsquedas
posteriores dentro de la misma sesión, mediante el algoritmo clásico
de Rocchio sobre los embeddings CLIP:

    q' = alpha * q  +  beta * mean(embeddings de docs "me gusta")
                     -  gamma * mean(embeddings de docs "no me gusta")

El vector resultante se vuelve a normalizar (L2) y se usa como nueva
consulta vectorial para FAISS. Los pesos alpha/beta/gamma están en
config.py (ROCCHIO_ALPHA/BETA/GAMMA).

El feedback se guarda en memoria de la sesión (no persiste entre
sesiones ni usuarios), que es lo mínimo pedido por el enunciado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class FeedbackStore:
    """Guarda embeddings de documentos marcados como relevantes/no
    relevantes durante la sesión actual."""

    liked_vecs: list[np.ndarray] = field(default_factory=list)
    disliked_vecs: list[np.ndarray] = field(default_factory=list)
    liked_ids: set = field(default_factory=set)
    disliked_ids: set = field(default_factory=set)

    def like(self, doc_id: str, vector: np.ndarray) -> None:
        self.disliked_ids.discard(doc_id)
        self.liked_ids.add(doc_id)
        self._rebuild(doc_id, vector, positive=True)

    def dislike(self, doc_id: str, vector: np.ndarray) -> None:
        self.liked_ids.discard(doc_id)
        self.disliked_ids.add(doc_id)
        self._rebuild(doc_id, vector, positive=False)

    # --- internals ---------------------------------------------------
    # Nota: para simplicidad guardamos {doc_id: vector} de forma separada
    # a las listas liked/disliked, reconstruyéndolas cada vez que cambia
    # un voto (permite que un usuario cambie de opinión de like a dislike).
    _vectors_by_id: dict = field(default_factory=dict)

    def _all_ids(self):
        return list(self._vectors_by_id.keys())

    def _all_vecs(self):
        return list(self._vectors_by_id.values())

    def _rebuild(self, doc_id: str, vector: np.ndarray, positive: bool) -> None:
        self._vectors_by_id[doc_id] = vector
        self.liked_vecs = [self._vectors_by_id[i] for i in self.liked_ids]
        self.disliked_vecs = [self._vectors_by_id[i] for i in self.disliked_ids]

    def has_feedback(self) -> bool:
        return bool(self.liked_ids or self.disliked_ids)

    def vote_for(self, doc_id: str) -> str | None:
        if doc_id in self.liked_ids:
            return "like"
        if doc_id in self.disliked_ids:
            return "dislike"
        return None


def rocchio_update(query_vec: np.ndarray, feedback: FeedbackStore) -> np.ndarray:
    """Aplica el algoritmo de Rocchio para reformular el vector de consulta
    en base al feedback de relevancia acumulado en la sesión."""
    if not feedback.has_feedback():
        return query_vec

    new_vec = config.ROCCHIO_ALPHA * query_vec.astype("float64")

    if feedback.liked_vecs:
        mean_liked = np.mean(np.stack(feedback.liked_vecs), axis=0)
        new_vec = new_vec + config.ROCCHIO_BETA * mean_liked

    if feedback.disliked_vecs:
        mean_disliked = np.mean(np.stack(feedback.disliked_vecs), axis=0)
        new_vec = new_vec - config.ROCCHIO_GAMMA * mean_disliked

    norm = np.linalg.norm(new_vec)
    if norm < 1e-8:
        return query_vec
    return (new_vec / norm).astype("float32")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dim = 16
    query = rng.normal(size=dim).astype("float32")
    query /= np.linalg.norm(query)

    fb = FeedbackStore()
    liked_doc = rng.normal(size=dim).astype("float32")
    liked_doc /= np.linalg.norm(liked_doc)
    disliked_doc = -query  # documento opuesto a la consulta original

    fb.like("doc_liked", liked_doc)
    fb.dislike("doc_disliked", disliked_doc)

    new_query = rocchio_update(query, fb)
    sim_before_liked = float(query @ liked_doc)
    sim_after_liked = float(new_query @ liked_doc)
    print("similaridad con doc 'me gusta' antes:", round(sim_before_liked, 3))
    print("similaridad con doc 'me gusta' despues:", round(sim_after_liked, 3))
    assert sim_after_liked > sim_before_liked, "Rocchio debe acercar la consulta al doc 'me gusta'"
    print("OK: Rocchio mueve la consulta hacia los documentos marcados como relevantes")
