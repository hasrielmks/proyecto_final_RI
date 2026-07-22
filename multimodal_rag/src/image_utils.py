"""
src/image_utils.py
===================
Descarga y cachea localmente las imágenes del corpus (referenciadas por URL
en meta_Digital_Music.jsonl) para poder generar sus embeddings con CLIP.

Las imágenes se guardan en config.IMAGES_CACHE_DIR usando el doc_id como
nombre de archivo, así que solo se descargan una vez aunque se vuelva a
correr el pipeline. Se usa un ThreadPoolExecutor porque la descarga es
I/O-bound (esperar respuesta HTTP), no CPU-bound.

Nota: si una imagen no se puede descargar (URL rota, timeout, host caído),
el documento simplemente se indexa solo con su embedding de texto — el
sistema sigue funcionando en modo texto-only para ese ítem.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from PIL import Image
from tqdm import tqdm

import config


def _cache_path(doc_id: str, url: str) -> Path:
    # Conservamos la extensión original cuando existe (jpg/gif/png) porque
    # PIL la usa como pista de formato.
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
        ext = ".jpg"
    safe_id = hashlib.md5(doc_id.encode()).hexdigest()[:16]
    return config.IMAGES_CACHE_DIR / f"{safe_id}{ext}"


def download_image(doc_id: str, url: str) -> Optional[Path]:
    """Descarga (con reintentos) una imagen y la deja en caché local.

    Devuelve la ruta local si tuvo éxito (o ya estaba cacheada), None si
    falló tras agotar los reintentos.
    """
    dest = _cache_path(doc_id, url)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Coursework-Bot/1.0)"}
    last_error = None
    for attempt in range(config.IMAGE_DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(
                url, headers=headers, timeout=config.IMAGE_DOWNLOAD_TIMEOUT
            )
            resp.raise_for_status()
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            with open(tmp_path, "wb") as f:
                f.write(resp.content)
            # Validamos que sea una imagen decodificable antes de aceptarla
            with Image.open(tmp_path) as im:
                im.verify()
            tmp_path.rename(dest)
            return dest
        except Exception as e:  # noqa: BLE001 - queremos capturar cualquier fallo de red/IO
            last_error = e
            continue
    return None


def download_images_bulk(doc_id_url_pairs: list[tuple[str, str]]) -> dict[str, Optional[Path]]:
    """Descarga en paralelo una lista de (doc_id, url). Devuelve doc_id -> path|None."""
    results: dict[str, Optional[Path]] = {}
    with ThreadPoolExecutor(max_workers=config.IMAGE_DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(download_image, doc_id, url): doc_id
            for doc_id, url in doc_id_url_pairs
            if url
        }
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Descargando imágenes"
        ):
            doc_id = futures[fut]
            try:
                results[doc_id] = fut.result()
            except Exception:
                results[doc_id] = None
    return results


def load_image(path: Path) -> Optional[Image.Image]:
    """Carga una imagen local como PIL.Image en modo RGB, o None si falla."""
    try:
        img = Image.open(path)
        img = img.convert("RGB")
        return img
    except Exception:
        return None


if __name__ == "__main__":
    # Prueba rápida con un puñado de documentos reales del corpus.
    from data_loader import load_corpus

    docs = load_corpus(max_docs=5)
    pairs = [(d.doc_id, d.image_url) for d in docs if d.image_url]
    results = download_images_bulk(pairs)
    for doc_id, path in results.items():
        print(doc_id, "->", path, "OK" if path else "FALLÓ")
