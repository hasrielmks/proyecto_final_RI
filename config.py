"""
config.py
==========
Configuración centralizada del Sistema de Recuperación de Información
Multimodal con RAG (Asignatura: Recuperación de Información).

Todos los módulos importan sus parámetros desde aquí. No debería ser
necesario tocar otros archivos para ajustar rutas, tamaños de batch,
modelos, etc.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_CACHE_DIR = DATA_DIR / "images_cache"
INDEX_DIR = DATA_DIR / "index"
EVAL_DIR = DATA_DIR / "eval"

for _d in (DATA_DIR, IMAGES_CACHE_DIR, INDEX_DIR, EVAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Corpus fuente (JSONL local, formato Amazon Reviews 2023 - metadata)
CORPUS_PATH = os.environ.get(
    "CORPUS_PATH", str(BASE_DIR / "data" / "meta_Digital_Music.jsonl")
)

# Artefactos generados por scripts/build_index.py
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"          # índice FAISS
DOC_METADATA_PATH = INDEX_DIR / "doc_metadata.jsonl"  # metadata alineada al índice
EMBED_INFO_PATH = INDEX_DIR / "embed_info.json"       # info del modelo usado

# ---------------------------------------------------------------------------
# Límite de corpus (para pruebas rápidas / hardware limitado).
# En producción (máquina del usuario) se recomienda usar None (todo el corpus).
# Se puede sobreescribir con la variable de entorno MAX_DOCS.
# ---------------------------------------------------------------------------
MAX_DOCS = os.environ.get("MAX_DOCS")
MAX_DOCS = int(MAX_DOCS) if MAX_DOCS else None

# ---------------------------------------------------------------------------
# Modelo CLIP (OpenCLIP)
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
CLIP_EMBED_DIM = 512
CLIP_DEVICE = os.environ.get("CLIP_DEVICE", "cpu")  # "cuda" si hay GPU disponible
CLIP_BATCH_SIZE = int(os.environ.get("CLIP_BATCH_SIZE", "32"))

# Los embeddings de texto e imagen se combinan (promedio ponderado) para
# formar la representación final de cada documento del corpus.
TEXT_WEIGHT = 0.6
IMAGE_WEIGHT = 0.4

# Máximo de tokens/caracteres de texto que se usan para el embedding CLIP
# (el text encoder de CLIP trunca a 77 tokens, así que usamos un resumen corto)
CLIP_MAX_TEXT_CHARS = 300

# ---------------------------------------------------------------------------
# Descarga de imágenes
# ---------------------------------------------------------------------------
IMAGE_DOWNLOAD_TIMEOUT = 8
IMAGE_DOWNLOAD_RETRIES = 2
IMAGE_DOWNLOAD_WORKERS = 8
IMAGE_VARIANT_PRIORITY = ("large", "hi_res", "thumb")  # orden de preferencia

# ---------------------------------------------------------------------------
# FAISS
# ---------------------------------------------------------------------------
# IndexFlatIP sobre vectores normalizados == similitud coseno exacta.
# Buena opción para corpus de decenas/cientos de miles de docs.
FAISS_METRIC = "cosine"

# ---------------------------------------------------------------------------
# Recuperación
# ---------------------------------------------------------------------------
TOP_K_RETRIEVAL = 20     # candidatos iniciales recuperados por FAISS
TOP_K_RERANK = 5         # documentos finales tras reranking, usados como contexto
TOP_K_DISPLAY = 5        # evidencias mostradas en la UI

# ---------------------------------------------------------------------------
# Query expansion
# ---------------------------------------------------------------------------
QUERY_EXPANSION_ENABLED_DEFAULT = True
QUERY_EXPANSION_MODE = os.environ.get("QUERY_EXPANSION_MODE", "llm")  # "llm" | "heuristic"
QUERY_EXPANSION_MAX_TERMS = 6

# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------
RERANK_ENABLED_DEFAULT = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------------------------------------------------------------------------
# Memoria conversacional
# ---------------------------------------------------------------------------
MEMORY_ENABLED_DEFAULT = True
MEMORY_MAX_TURNS = 5  # cuántos turnos previos se usan para reformular la consulta

# ---------------------------------------------------------------------------
# Relevance feedback (Rocchio)
# ---------------------------------------------------------------------------
FEEDBACK_ENABLED_DEFAULT = True
ROCCHIO_ALPHA = 1.0   # peso de la consulta original
ROCCHIO_BETA = 0.6    # peso de documentos marcados "me gusta"
ROCCHIO_GAMMA = 0.4   # peso de documentos marcados "no me gusta"

# ---------------------------------------------------------------------------
# Generación (Gemini API)
# ---------------------------------------------------------------------------
# gemini-2.0-flash fue descontinuado (shutdown 1-jun-2026). gemini-2.5-flash
# es la opción estable/económica recomendada a jul-2026; para más capacidad
# (a mayor costo) se puede usar "gemini-3.5-flash" vía variable de entorno.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
GENERATION_MAX_OUTPUT_TOKENS = 700
GENERATION_TEMPERATURE = 0.3

RAG_SYSTEM_PROMPT = """Eres un asistente experto en música que responde preguntas \
usando EXCLUSIVAMENTE la información de contexto proporcionada, la cual proviene \
de un catálogo de productos musicales (álbumes, discos, artistas). Si el contexto \
no contiene información suficiente para responder, dilo explícitamente en vez de \
inventar datos. Responde de forma clara y concisa, y cuando sea relevante menciona \
los títulos/artistas exactos que aparecen en el contexto."""

# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
EVAL_QUERIES_PATH = EVAL_DIR / "eval_queries.json"
EVAL_QRELS_PATH = EVAL_DIR / "qrels.json"
EVAL_RESULTS_PATH = EVAL_DIR / "eval_results.json"
EVAL_K_VALUES = (5, 10)
