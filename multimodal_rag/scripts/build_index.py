#!/usr/bin/env python3
"""
scripts/build_index.py
=======================
Pipeline offline de indexado. Se ejecuta UNA vez (o cada vez que cambie
el corpus) para:

  1. Cargar y normalizar el corpus (src/data_loader.py)
  2. Descargar y cachear las imágenes asociadas (src/image_utils.py)
  3. Generar embeddings multimodales CLIP (texto + imagen fusionados)
     (src/embeddings.py)
  4. Indexar los vectores resultantes en FAISS y guardar en disco
     (src/vector_store.py)

Uso:
    python scripts/build_index.py
    MAX_DOCS=5000 python scripts/build_index.py     # para pruebas rápidas
    python scripts/build_index.py --max-docs 5000
    python scripts/build_index.py --skip-images     # solo embeddings de texto

Requiere conexión a internet para:
  - descargar las imágenes del corpus (dominio m.media-amazon.com)
  - descargar los pesos preentrenados de CLIP la primera vez (huggingface.co)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

import config
from src import data_loader, embeddings, image_utils, vector_store


def parse_args():
    p = argparse.ArgumentParser(description="Construye el índice FAISS del corpus multimodal.")
    p.add_argument("--max-docs", type=int, default=config.MAX_DOCS, help="Límite de documentos a indexar (debug/pruebas).")
    p.add_argument("--skip-images", action="store_true", help="No descargar ni embeber imágenes (más rápido, solo texto).")
    p.add_argument("--corpus-path", type=str, default=config.CORPUS_PATH)
    p.add_argument("--embed-batch-size", type=int, default=config.CLIP_BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    print(f"[1/4] Cargando y normalizando corpus desde {args.corpus_path} ...")
    docs = data_loader.load_corpus(args.corpus_path, max_docs=args.max_docs)
    print(f"      -> {len(docs)} documentos normalizados (título válido, sin duplicados).")

    image_paths = {}
    if not args.skip_images:
        print("[2/4] Descargando y cacheando imágenes (esto puede tardar en la primera corrida) ...")
        pairs = [(d.doc_id, d.image_url) for d in docs if d.image_url]
        image_paths = image_utils.download_images_bulk(pairs)
        n_ok = sum(1 for p in image_paths.values() if p is not None)
        print(f"      -> {n_ok}/{len(pairs)} imágenes descargadas correctamente "
              f"({len(pairs) - n_ok} no disponibles; esos documentos se indexan solo con texto).")
    else:
        print("[2/4] --skip-images activo: se omite la descarga de imágenes.")

    print("[3/4] Generando embeddings CLIP (texto + imagen fusionados) e indexando en FAISS ...")
    store = vector_store.VectorStore(dim=config.CLIP_EMBED_DIM)

    batch_size = args.embed_batch_size
    for start in tqdm(range(0, len(docs), batch_size), desc="Embediendo lotes"):
        batch_docs = docs[start : start + batch_size]

        texts = [d.embedding_text() for d in batch_docs]
        text_vecs = embeddings.embed_texts(texts, batch_size=batch_size)

        pil_images = []
        has_image = np.zeros(len(batch_docs), dtype=bool)
        for i, d in enumerate(batch_docs):
            path = image_paths.get(d.doc_id)
            img = image_utils.load_image(path) if path else None
            pil_images.append(img)
            has_image[i] = img is not None

        if has_image.any():
            image_vecs = embeddings.embed_images(pil_images, batch_size=batch_size)
        else:
            image_vecs = np.zeros((len(batch_docs), config.CLIP_EMBED_DIM), dtype="float32")

        fused_vecs = embeddings.fuse_text_image(text_vecs, image_vecs, has_image)

        metadata_batch = []
        for d, has_img in zip(batch_docs, has_image):
            meta = d.to_dict()
            meta["context_text"] = d.context_text()
            meta["has_image_embedding"] = bool(has_img)
            metadata_batch.append(meta)

        store.add(fused_vecs, metadata_batch)

    print(f"[4/4] Guardando índice FAISS ({store.index.ntotal} vectores) en {config.INDEX_DIR} ...")
    store.save()

    embed_info = {
        "clip_model": config.CLIP_MODEL_NAME,
        "clip_pretrained": config.CLIP_PRETRAINED,
        "embed_dim": config.CLIP_EMBED_DIM,
        "text_weight": config.TEXT_WEIGHT,
        "image_weight": config.IMAGE_WEIGHT,
        "n_documents": store.index.ntotal,
        "skip_images": args.skip_images,
        "corpus_path": args.corpus_path,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config.EMBED_INFO_PATH.write_text(json.dumps(embed_info, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\nListo. Índice construido en {elapsed/60:.1f} min.")
    print(f"  - Índice FAISS:  {config.FAISS_INDEX_PATH}")
    print(f"  - Metadata:      {config.DOC_METADATA_PATH}")


if __name__ == "__main__":
    main()
