"""
src/data_loader.py
===================
Carga el corpus multimodal (meta_Digital_Music.jsonl, formato Amazon
Reviews 2023 - Metadata) y lo normaliza a una estructura de documento
única usada por el resto del pipeline (embeddings, indexado, RAG, UI).

Cada línea del JSONL original es un producto musical con campos como
title, description, features, store (artista/sello), details, price,
average_rating, images (lista de variantes thumb/large/hi_res), etc.

Este módulo:
  1. Parsea el JSONL línea a línea (streaming, no carga todo en RAM de golpe
     salvo que se pida explícitamente).
  2. Filtra documentos sin título (inservibles para RI) y descarta
     duplicados por parent_asin.
  3. Asocia correctamente cada imagen (mejor variante disponible) con su
     texto correspondiente.
  4. Construye dos vistas de texto por documento:
       - texto de embedding (corto, para el encoder CLIP, limitado a
         ~77 tokens de contexto)
       - texto de contexto (más largo, el que se le muestra al LLM en el
         paso de generación del RAG)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

import config


@dataclass
class Document:
    """Representación normalizada de un documento del corpus."""

    doc_id: str                      # parent_asin (identificador único)
    title: str
    store: Optional[str] = None      # artista / sello discográfico
    description: str = ""            # description[] concatenado
    features: str = ""               # features[] concatenado
    categories: str = ""
    price: Optional[float] = None
    average_rating: Optional[float] = None
    rating_number: Optional[int] = None
    details: dict = field(default_factory=dict)
    image_url: Optional[str] = None  # mejor URL de imagen disponible
    image_variant: Optional[str] = None

    def embedding_text(self, max_chars: int = None) -> str:
        """Texto corto para el encoder de texto de CLIP (trunca a max_chars)."""
        max_chars = max_chars or config.CLIP_MAX_TEXT_CHARS
        parts = [self.title]
        if self.store:
            parts.append(f"por {self.store}")
        if self.description:
            parts.append(self.description)
        text = ". ".join(p.strip() for p in parts if p and p.strip())
        return text[:max_chars]

    def context_text(self) -> str:
        """Texto largo usado como contexto para el LLM generador."""
        lines = [f"Título: {self.title}"]
        if self.store:
            lines.append(f"Artista/Sello: {self.store}")
        if self.average_rating is not None:
            extra = f" ({self.rating_number} reseñas)" if self.rating_number else ""
            lines.append(f"Calificación promedio: {self.average_rating}/5{extra}")
        if self.price is not None:
            lines.append(f"Precio: ${self.price}")
        if self.categories:
            lines.append(f"Categorías: {self.categories}")
        if self.description:
            lines.append(f"Descripción: {self.description}")
        if self.features:
            lines.append(f"Características: {self.features}")
        interesting_details = {
            k: v
            for k, v in (self.details or {}).items()
            if k in ("Label", "Format", "Number of discs", "Genre", "Artist")
        }
        if interesting_details:
            det = ", ".join(f"{k}: {v}" for k, v in interesting_details.items())
            lines.append(f"Detalles: {det}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Document":
        return Document(**d)


def _clean_ws(text: Optional[str]) -> Optional[str]:
    """Colapsa espacios/tabs/saltos de línea múltiples en uno solo."""
    if not text:
        return text
    return re.sub(r"\s+", " ", text).strip()


def _best_image(images: list) -> tuple[Optional[str], Optional[str]]:
    """Elige la mejor URL de imagen disponible según la prioridad de config."""
    if not images:
        return None, None
    # Preferimos la variante MAIN si existe, si no la primera de la lista.
    main = next((im for im in images if im.get("variant") == "MAIN"), images[0])
    for key in config.IMAGE_VARIANT_PRIORITY:
        url = main.get(key)
        if url:
            return url, key
    # último recurso: cualquier campo no vacío de cualquier variante
    for im in images:
        for key in config.IMAGE_VARIANT_PRIORITY:
            url = im.get(key)
            if url:
                return url, key
    return None, None


def _normalize_record(raw: dict) -> Optional[Document]:
    title = (raw.get("title") or "").strip()
    if not title:
        return None
    doc_id = raw.get("parent_asin")
    if not doc_id:
        return None

    description = " ".join(
        s.strip() for s in (raw.get("description") or []) if s and s.strip()
    )
    features = " ".join(
        s.strip() for s in (raw.get("features") or []) if s and s.strip()
    )
    categories = " > ".join(
        s.strip() for s in (raw.get("categories") or []) if s and s.strip()
    )
    image_url, image_variant = _best_image(raw.get("images") or [])

    return Document(
        doc_id=doc_id,
        title=_clean_ws(title),
        store=_clean_ws(raw.get("store")) or None,
        description=_clean_ws(description),
        features=_clean_ws(features),
        categories=_clean_ws(categories),
        price=raw.get("price"),
        average_rating=raw.get("average_rating"),
        rating_number=raw.get("rating_number"),
        details=raw.get("details") or {},
        image_url=image_url,
        image_variant=image_variant,
    )


def iter_corpus(
    path: str | Path = None, max_docs: int | None = None
) -> Iterator[Document]:
    """Itera documentos normalizados desde el JSONL fuente (streaming)."""
    path = Path(path or config.CORPUS_PATH)
    seen_ids = set()
    n_yielded = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = _normalize_record(raw)
            if doc is None:
                continue
            if doc.doc_id in seen_ids:
                continue
            seen_ids.add(doc.doc_id)
            yield doc
            n_yielded += 1
            if max_docs and n_yielded >= max_docs:
                break


def load_corpus(path: str | Path = None, max_docs: int | None = None) -> list[Document]:
    """Carga el corpus completo en memoria como lista de Document."""
    return list(iter_corpus(path, max_docs))


if __name__ == "__main__":
    # Prueba rápida: cargar los primeros documentos y mostrar un resumen.
    docs = load_corpus(max_docs=5)
    for d in docs:
        print("=" * 60)
        print(d.doc_id, "|", d.title)
        print("img:", d.image_url, f"({d.image_variant})")
        print(d.context_text())
        print("[embedding_text]:", d.embedding_text())
