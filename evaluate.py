#!/usr/bin/env python3
"""
evaluate.py
===========
Evalúa el sistema de recuperación usando el conjunto de consultas y
qrels generado por scripts/generate_qrels.py, reportando:

    - Precision@k
    - Recall@k
    - NDCG@k

para k en config.EVAL_K_VALUES (por defecto 5 y 10), promediadas sobre
todas las consultas de evaluación.

Por diseño, evaluate.py llama al MISMO componente de recuperación
(src/vector_store.VectorStore + embeddings + opcionalmente reranking)
que usa la aplicación Streamlit (vía src/rag_pipeline.py), para que la
evaluación sea representativa del sistema real y no de una versión
paralela/idealizada del retrieval.

Uso:
    python evaluate.py                          # con reranking (por defecto)
    python evaluate.py --no-rerank               # solo FAISS + CLIP
    python evaluate.py --k 5 10 20
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import config
from src import embeddings, reranker, vector_store


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_ids)
    return hits / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """NDCG@k con relevancia binaria (1 si el doc está en qrels, 0 si no)."""
    top_k = retrieved_ids[:k]
    dcg = sum(
        (1.0 if doc_id in relevant_ids else 0.0) / math.log2(i + 2)
        for i, doc_id in enumerate(top_k)
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def run_query(store: vector_store.VectorStore, query_text: str, top_k: int, use_rerank: bool) -> list[str]:
    query_vec = embeddings.embed_query(query_text)
    retrieval_k = max(top_k, config.TOP_K_RETRIEVAL) if use_rerank else top_k
    candidates = store.search(query_vec, top_k=retrieval_k)
    if use_rerank and candidates:
        candidates = reranker.rerank(query_text, candidates, top_k=top_k)
    return [c["doc_id"] for c in candidates[:top_k]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, nargs="+", default=list(config.EVAL_K_VALUES))
    p.add_argument("--no-rerank", action="store_true", help="Evaluar solo FAISS+CLIP, sin reranking.")
    p.add_argument("--queries-path", type=str, default=str(config.EVAL_QUERIES_PATH))
    p.add_argument("--qrels-path", type=str, default=str(config.EVAL_QRELS_PATH))
    args = p.parse_args()

    queries_path = Path(args.queries_path)
    qrels_path = Path(args.qrels_path)
    if not queries_path.exists() or not qrels_path.exists():
        raise SystemExit(
            "No se encontraron los archivos de evaluación. Corré primero:\n"
            "  python scripts/generate_qrels.py"
        )
    if not config.FAISS_INDEX_PATH.exists():
        raise SystemExit(
            "No se encontró el índice FAISS. Corré primero:\n"
            "  python scripts/build_index.py"
        )

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    qrels = json.loads(qrels_path.read_text(encoding="utf-8"))
    if not queries:
        raise SystemExit(
            "El archivo de consultas de evaluación está vacío. Esto puede pasar si "
            "indexaste un subconjunto muy chico del corpus (--max-docs bajo) y "
            "ningún artista/keyword alcanzó el mínimo de documentos relevantes. "
            "Probá con un --max-docs más alto en scripts/generate_qrels.py."
        )

    print(f"Cargando índice FAISS desde {config.FAISS_INDEX_PATH} ...")
    store = vector_store.VectorStore.load()
    print(f"  -> {store.index.ntotal} documentos indexados")

    use_rerank = not args.no_rerank
    print(f"Evaluando {len(queries)} consultas (reranking={'sí' if use_rerank else 'no'}) ...\n")

    per_query_results = {}
    metric_sums = {f"P@{k}": 0.0 for k in args.k}
    metric_sums.update({f"R@{k}": 0.0 for k in args.k})
    metric_sums.update({f"NDCG@{k}": 0.0 for k in args.k})

    for qid, qtext in queries.items():
        relevant_ids = set(qrels.get(qid, {}).keys())
        max_k = max(args.k)
        retrieved_ids = run_query(store, qtext, top_k=max_k, use_rerank=use_rerank)

        row = {"query": qtext, "n_relevant": len(relevant_ids)}
        for k in args.k:
            p_k = precision_at_k(retrieved_ids, relevant_ids, k)
            r_k = recall_at_k(retrieved_ids, relevant_ids, k)
            ndcg_k = ndcg_at_k(retrieved_ids, relevant_ids, k)
            row[f"P@{k}"] = p_k
            row[f"R@{k}"] = r_k
            row[f"NDCG@{k}"] = ndcg_k
            metric_sums[f"P@{k}"] += p_k
            metric_sums[f"R@{k}"] += r_k
            metric_sums[f"NDCG@{k}"] += ndcg_k
        per_query_results[qid] = row

    n = len(queries)
    macro_avg = {metric: total / n for metric, total in metric_sums.items()} if n else {}

    # --- Reporte por consola -------------------------------------------------
    header = f"{'query':35s}" + "".join(f"{m:>10s}" for m in metric_sums)
    print(header)
    print("-" * len(header))
    for qid, row in per_query_results.items():
        line = f"{row['query'][:34]:35s}" + "".join(
            f"{row[m]:>10.3f}" for m in metric_sums
        )
        print(line)
    print("-" * len(header))
    avg_line = f"{'PROMEDIO':35s}" + "".join(f"{macro_avg[m]:>10.3f}" for m in metric_sums)
    print(avg_line)

    # --- Guardar resultados ---------------------------------------------------
    output = {
        "config": {
            "use_rerank": use_rerank,
            "k_values": args.k,
            "n_queries": n,
            "n_indexed_docs": store.index.ntotal,
        },
        "macro_avg": macro_avg,
        "per_query": per_query_results,
    }
    config.EVAL_RESULTS_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResultados guardados en {config.EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
