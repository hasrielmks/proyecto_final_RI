#!/usr/bin/env python3
"""
scripts/generate_qrels.py
==========================
Genera el conjunto de evaluación (consultas + qrels) requerido por el
enunciado ("El sistema deberá evaluarse utilizando un conjunto de
consultas y documentos relevantes (qrels)").

Como este es un catálogo de productos (no hay juicios de relevancia
humanos preexistentes, tipo TREC), generamos qrels de forma
programática pero fundamentada, con dos estrategias combinadas:

  A) Consultas "por artista/sello" (store): la consulta es el nombre
     del artista/sello, y son relevantes todos los documentos cuyo
     campo `store` coincide (match exacto de campo estructurado,
     no ambiguo). Esto da qrels 100% confiables porque no dependen de
     interpretación semántica.

  B) Consultas "por género/temática" derivadas de categorías/keywords
     presentes en título+descripción (p.ej. "jazz", "christmas",
     "classical"): son relevantes los documentos cuyo título o
     descripción contienen ese keyword (whole-word match). Es un
     proxy razonable de relevancia temática, estándar en RI cuando no
     hay anotación manual (pooling por keyword).

Esto genera un qrels binario {query_id: {doc_id: relevancia(0/1)}} y
un archivo de queries {query_id: texto_query}, listos para
scripts/evaluate.py.

Uso:
    python scripts/generate_qrels.py --n-artist-queries 15 --n-keyword-queries 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src import data_loader

# Vocabulario temático candidato para las consultas de tipo (B). Se filtra
# luego a los que efectivamente tengan suficientes documentos relevantes
# en el corpus.
CANDIDATE_KEYWORDS = [
    "reggae", "opera", "soundtrack", "gospel", "choir", "classical", "folk",
    "acoustic", "symphony", "greatest hits", "guitar", "country", "jazz",
    "christmas", "orchestra", "blues", "piano", "rock", "live",
]

MIN_RELEVANT_DOCS = 3   # una query solo es útil para evaluar si tiene >= N relevantes
MAX_RELEVANT_DOCS_KEYWORD = 500  # evita keywords demasiado genéricos/ruidosos

_GENERIC_ARTIST_NAMES = {
    "various", "various artists", "v.a.", "va", "unknown", "unknown artist", "",
}


def _clean_artist_name(raw_store: str) -> str | None:
    """El campo `store` del corpus suele venir como
    'Nombre Artista (Artist) Format: Audio CD' o similar. Nos quedamos
    solo con el nombre real del artista/sello, descartando entradas
    genéricas (Various, formatos sin artista, etc.) que no sirven como
    consulta de evaluación."""
    name = re.split(r"\s*Format:", raw_store)[0]
    name = re.sub(r"\((?:Artist|Author|Composer|Conductor|Orchestra)\)", "", name)
    name = re.sub(r"\s+", " ", name).strip(" ,")
    if not name or name.lower() in _GENERIC_ARTIST_NAMES:
        return None
    return name


def build_artist_queries(docs: list[data_loader.Document], n: int) -> dict[str, list[str]]:
    """Consultas del tipo 'discos de <artista>' usando el campo store."""
    by_store: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        if not d.store or len(d.store) >= 60:
            continue
        clean = _clean_artist_name(d.store)
        if clean:
            by_store[clean].append(d.doc_id)

    # Priorizamos artistas con un número "evaluable" de documentos:
    # ni tan pocos que no aporten señal, ni tan mezclados con otros artistas
    # en el mismo string que no tenga sentido como consulta.
    candidates = [
        (store, ids) for store, ids in by_store.items() if MIN_RELEVANT_DOCS <= len(ids) <= 200
    ]
    candidates.sort(key=lambda x: -len(x[1]))

    queries = {}
    for store, ids in candidates[:n]:
        queries[store] = ids
    return queries


def build_keyword_queries(docs: list[data_loader.Document], n: int) -> dict[str, list[str]]:
    """Consultas temáticas basadas en presencia de keyword en título/descripción."""
    queries = {}
    for kw in CANDIDATE_KEYWORDS:
        pattern = re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE)
        matches = [
            d.doc_id
            for d in docs
            if pattern.search(d.title) or pattern.search(d.description)
        ]
        if MIN_RELEVANT_DOCS <= len(matches) <= MAX_RELEVANT_DOCS_KEYWORD:
            queries[kw] = matches
        if len(queries) >= n:
            break
    return queries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-docs", type=int, default=config.MAX_DOCS)
    p.add_argument("--n-artist-queries", type=int, default=15)
    p.add_argument("--n-keyword-queries", type=int, default=10)
    args = p.parse_args()

    print(f"Cargando corpus (max_docs={args.max_docs}) ...")
    docs = data_loader.load_corpus(max_docs=args.max_docs)
    print(f"  -> {len(docs)} documentos")

    artist_queries = build_artist_queries(docs, args.n_artist_queries)
    keyword_queries = build_keyword_queries(docs, args.n_keyword_queries)

    queries = {}
    qrels = {}
    qid = 0
    for store, ids in artist_queries.items():
        qid_str = f"artist_{qid}"
        queries[qid_str] = f"discos de {store}"
        qrels[qid_str] = {doc_id: 1 for doc_id in ids}
        qid += 1
    for kw, ids in keyword_queries.items():
        qid_str = f"keyword_{qid}"
        queries[qid_str] = f"música de {kw}"
        qrels[qid_str] = {doc_id: 1 for doc_id in ids}
        qid += 1

    config.EVAL_QUERIES_PATH.write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config.EVAL_QRELS_PATH.write_text(
        json.dumps(qrels, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{len(queries)} consultas de evaluación generadas "
          f"({len(artist_queries)} por artista, {len(keyword_queries)} por keyword).")
    for qid_str, qtext in queries.items():
        print(f"  - [{qid_str}] \"{qtext}\" -> {len(qrels[qid_str])} docs relevantes")
    print(f"\nGuardado en:\n  {config.EVAL_QUERIES_PATH}\n  {config.EVAL_QRELS_PATH}")


if __name__ == "__main__":
    main()
