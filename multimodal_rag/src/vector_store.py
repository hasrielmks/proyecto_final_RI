"""
src/vector_store.py
====================
Wrapper delgado sobre FAISS para indexar y buscar los embeddings
multimodales del corpus.

Usamos IndexFlatIP (producto interno) sobre vectores L2-normalizados,
lo cual es matemáticamente equivalente a similitud coseno exacta
(sin aproximación / sin pérdida de recall), apropiado para un corpus
de decenas de miles de documentos como el de este proyecto.

Se guarda además un archivo JSONL de metadata (doc_id, title, image_url,
etc.) alineado por posición con los vectores del índice, para poder
recuperar la información completa de un resultado a partir de su
posición devuelta por FAISS.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

import config


class VectorStore:
    def __init__(self, dim: int = None):
        self.dim = dim or config.CLIP_EMBED_DIM
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata: list[dict] = []  # alineado por posición con self.index
        self._id_to_pos: dict[str, int] = {}

    def add(self, vectors: np.ndarray, metadata_batch: list[dict]) -> None:
        assert len(vectors) == len(metadata_batch)
        if len(vectors) == 0:
            return
        vectors = np.ascontiguousarray(vectors.astype("float32"))
        start_pos = self.index.ntotal
        self.index.add(vectors)
        self.metadata.extend(metadata_batch)
        for i, m in enumerate(metadata_batch):
            self._id_to_pos[m["doc_id"]] = start_pos + i

    def get_vector(self, doc_id: str) -> np.ndarray | None:
        """Reconstruye el vector original de un documento a partir de su
        doc_id (usado por src/feedback.py para el algoritmo de Rocchio)."""
        pos = self._id_to_pos.get(doc_id)
        if pos is None:
            return None
        return self.index.reconstruct(pos)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[dict]:
        """Busca los top_k documentos más similares a query_vec (1D, normalizado).

        Devuelve una lista de dicts: {**metadata, "score": float, "rank": int}
        """
        if self.index.ntotal == 0:
            return []
        q = np.ascontiguousarray(query_vec.reshape(1, -1).astype("float32"))
        top_k = min(top_k, self.index.ntotal)
        scores, idxs = self.index.search(q, top_k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], idxs[0])):
            if idx < 0:
                continue
            meta = dict(self.metadata[idx])
            meta["score"] = float(score)
            meta["rank"] = rank + 1
            results.append(meta)
        return results

    def save(self, index_path: Path = None, metadata_path: Path = None) -> None:
        index_path = index_path or config.FAISS_INDEX_PATH
        metadata_path = metadata_path or config.DOC_METADATA_PATH
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_path: Path = None, metadata_path: Path = None) -> "VectorStore":
        index_path = index_path or config.FAISS_INDEX_PATH
        metadata_path = metadata_path or config.DOC_METADATA_PATH
        index = faiss.read_index(str(index_path))
        store = cls(dim=index.d)
        store.index = index
        with open(metadata_path, "r", encoding="utf-8") as f:
            store.metadata = [json.loads(line) for line in f]
        assert store.index.ntotal == len(store.metadata), (
            "Desalineación entre índice FAISS y metadata: "
            f"{store.index.ntotal} vectores vs {len(store.metadata)} metadatos"
        )
        store._id_to_pos = {m["doc_id"]: i for i, m in enumerate(store.metadata)}
        return store


if __name__ == "__main__":
    # Prueba rápida con vectores sintéticos
    rng = np.random.default_rng(0)
    dim = 8
    vs = VectorStore(dim=dim)
    vecs = rng.normal(size=(10, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    meta = [{"doc_id": f"doc{i}", "title": f"Documento {i}"} for i in range(10)]
    vs.add(vecs, meta)

    query = vecs[3]  # debería recuperarse a sí mismo con score ~1.0
    results = vs.search(query, top_k=3)
    for r in results:
        print(r["rank"], r["doc_id"], round(r["score"], 4))
    assert results[0]["doc_id"] == "doc3" and results[0]["score"] > 0.99

    test_dir = Path("/tmp/vs_test")
    test_dir.mkdir(exist_ok=True)
    vs.save(test_dir / "idx.faiss", test_dir / "meta.jsonl")
    vs2 = VectorStore.load(test_dir / "idx.faiss", test_dir / "meta.jsonl")
    results2 = vs2.search(query, top_k=1)
    assert results2[0]["doc_id"] == "doc3"
    print("OK: guardado/carga de índice FAISS funciona correctamente")
