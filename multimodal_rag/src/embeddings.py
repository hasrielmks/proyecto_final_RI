"""
src/embeddings.py
==================
Wrapper sobre OpenCLIP (ViT-B-32, pesos laion2b_s34b_b79k) para generar
embeddings multimodales de texto e imagen en un mismo espacio vectorial.

Este es el corazón de la "Construcción de representaciones vectoriales"
pedida en el enunciado: un único modelo CLIP embebe tanto las consultas
del usuario (texto) como los documentos del corpus (texto + imagen).

Estrategia de fusión texto+imagen (documento):
    doc_embedding = normalize(
        TEXT_WEIGHT * normalize(text_embedding) +
        IMAGE_WEIGHT * normalize(image_embedding)
    )
Si un documento no tiene imagen (o no se pudo descargar), se usa
únicamente el embedding de texto (fallback automático), y viceversa si
por algún motivo faltara el texto.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image

import config

_model = None
_preprocess = None
_tokenizer = None


def _lazy_load():
    """Carga el modelo CLIP una sola vez (patrón singleton) y lo cachea."""
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED
    )
    model = model.to(config.CLIP_DEVICE)
    model.eval()
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)

    _model, _preprocess, _tokenizer = model, preprocess, tokenizer


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1e-8, norm)
    return vec / norm


@torch.no_grad()
def embed_texts(texts: list[str], batch_size: int = None) -> np.ndarray:
    """Genera embeddings normalizados (L2) para una lista de textos."""
    _lazy_load()
    batch_size = batch_size or config.CLIP_BATCH_SIZE
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        tokens = _tokenizer(batch).to(config.CLIP_DEVICE)
        feats = _model.encode_text(tokens)
        feats = feats.cpu().numpy().astype("float32")
        all_vecs.append(feats)
    vecs = np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, config.CLIP_EMBED_DIM), "float32")
    return _l2_normalize(vecs)


@torch.no_grad()
def embed_images(images: list[Optional[Image.Image]], batch_size: int = None) -> np.ndarray:
    """Genera embeddings normalizados (L2) para una lista de imágenes PIL.

    Las entradas None (imagen faltante/no descargable) producen un vector
    de ceros, que luego se ignora en la fusión texto+imagen.
    """
    _lazy_load()
    batch_size = batch_size or config.CLIP_BATCH_SIZE
    all_vecs = np.zeros((len(images), config.CLIP_EMBED_DIM), dtype="float32")

    valid_idx = [i for i, im in enumerate(images) if im is not None]
    for start in range(0, len(valid_idx), batch_size):
        idx_batch = valid_idx[start : start + batch_size]
        tensors = torch.stack([_preprocess(images[i]) for i in idx_batch]).to(
            config.CLIP_DEVICE
        )
        feats = _model.encode_image(tensors)
        feats = feats.cpu().numpy().astype("float32")
        feats = _l2_normalize(feats)
        for local_i, global_i in enumerate(idx_batch):
            all_vecs[global_i] = feats[local_i]
    return all_vecs


def fuse_text_image(
    text_vecs: np.ndarray, image_vecs: np.ndarray, has_image: np.ndarray
) -> np.ndarray:
    """Combina embeddings de texto e imagen con los pesos de config.py.

    `has_image` es un array booleano que indica qué filas tienen imagen
    real (para no diluir el vector con ceros cuando no la hay).
    """
    text_vecs = _l2_normalize(text_vecs)
    fused = np.empty_like(text_vecs)
    for i in range(len(text_vecs)):
        if has_image[i]:
            v = config.TEXT_WEIGHT * text_vecs[i] + config.IMAGE_WEIGHT * image_vecs[i]
        else:
            v = text_vecs[i]
        fused[i] = v
    return _l2_normalize(fused)


def embed_query(text: str) -> np.ndarray:
    """Embedding de una consulta de usuario (solo texto)."""
    return embed_texts([text])[0]


if __name__ == "__main__":
    # Prueba rápida: embeber un par de textos y verificar dimensión/norma.
    vecs = embed_texts(["jazz album", "classical orchestra music"])
    print("shape:", vecs.shape)
    print("norms:", np.linalg.norm(vecs, axis=1))
    sim = vecs @ vecs.T
    print("similaridad coseno entre ambos textos:\n", sim)
