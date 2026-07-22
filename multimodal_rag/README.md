# Sistema de Recuperación de Información Multimodal con RAG

Proyecto final — Asignatura: Recuperación de Información.
Corpus: `meta_Digital_Music.jsonl` (metadata de Amazon Digital Music: título,
descripción, artista/sello, precio, rating, imágenes de portada, etc.)

Stack: **OpenCLIP (ViT-B-32)** para embeddings multimodales · **FAISS**
como base de datos vectorial · **Gemini API** para la generación RAG ·
**Streamlit** para la interfaz conversacional.

---

## 1. Arquitectura

```
                         ┌─────────────────────┐
                         │  meta_Digital_Music  │
                         │       .jsonl         │
                         └──────────┬───────────┘
                                    │  scripts/build_index.py
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
            data_loader.py   image_utils.py     embeddings.py
           (normaliza texto)  (descarga+cache)   (CLIP texto+imagen,
                    │               │              fusión ponderada)
                    └───────┬───────┘
                            ▼
                     vector_store.py  ───────►  data/index/faiss.index
                       (FAISS IndexFlatIP)       data/index/doc_metadata.jsonl


   Consulta usuario (app.py / evaluate.py)
        │
        ▼
   memory.py  ──► query_expansion.py ──► embeddings.embed_query ──► vector_store.search
 (condensa la     (LLM o heurístico,        (CLIP, mismo espacio        (Top-K candidatos,
  pregunta usando   amplía términos)          que el corpus)             similitud coseno)
  el historial)                                    ▲
        │                                          │ feedback.py (Rocchio, si hay likes/dislikes)
        ▼
   reranker.py (cross-encoder MS MARCO sobre los Top-K) ──► generator.py (Gemini API)
                                                                    │
                                                                    ▼
                                                        respuesta + evidencias (docs+imágenes+score)
```

### Estructura de archivos

```
config.py                  # configuración centralizada (rutas, hiperparámetros, modelos)
requirements.txt
app.py                     # interfaz Streamlit (chat + evidencias + feedback)
evaluate.py                # evaluación: Precision@k, Recall@k, NDCG@k

src/
  data_loader.py           # carga y normaliza el corpus JSONL
  image_utils.py           # descarga y cachea imágenes del corpus
  embeddings.py             # wrapper OpenCLIP (texto, imagen, fusión)
  vector_store.py           # wrapper FAISS (indexar, buscar, guardar/cargar)
  query_expansion.py        # [excelencia] expansión de consultas (LLM/heurística)
  reranker.py                # [excelencia] reranking con cross-encoder
  memory.py                  # [excelencia] memoria conversacional
  feedback.py                 # [excelencia] relevance feedback (Rocchio)
  generator.py                 # wrapper Gemini API (google-genai SDK)
  rag_pipeline.py               # orquesta todo el flujo RAG (usado por app.py y evaluate.py)

scripts/
  build_index.py             # pipeline offline: corpus -> embeddings -> índice FAISS
  generate_qrels.py           # genera consultas + qrels de evaluación

data/
  meta_Digital_Music.jsonl    # corpus fuente (colocar aquí)
  images_cache/                # imágenes descargadas (se genera automáticamente)
  index/                         # índice FAISS + metadata (se genera automáticamente)
  eval/                           # queries.json + qrels.json + eval_results.json
```

---

## 2. Instalación

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2.1. Corpus

Colocar `meta_Digital_Music.jsonl` en `data/meta_Digital_Music.jsonl`
(o apuntar `CORPUS_PATH` a otra ubicación vía variable de entorno).

### 2.2. Gemini API Key (necesaria para generación, query expansion LLM y memoria)

1. Conseguir una API key gratuita en <https://aistudio.google.com/apikey>
2. Exportarla:

```bash
export GEMINI_API_KEY="tu_api_key_aquí"      # Windows: set GEMINI_API_KEY=...
```

También se puede pegar la key directamente desde la barra lateral de la
app de Streamlit si no se configuró como variable de entorno.

**Sin API key el sistema sigue funcionando**: la recuperación (FAISS +
CLIP + reranking) funciona igual y se muestran las evidencias, pero la
generación de la respuesta final, la expansión de consultas en modo
`"llm"` y la memoria conversacional degradan automáticamente a modo
heurístico / se deshabilitan con un aviso claro, en vez de romper la
aplicación.

---

## 3. Construir el índice

```bash
python scripts/build_index.py
```

Esto descarga las imágenes del corpus (se cachean en
`data/images_cache/`, así que solo se descargan una vez), genera los
embeddings CLIP fusionados (texto + imagen) y construye el índice
FAISS en `data/index/`.

**Nota sobre tiempos**: el corpus completo tiene ~70.500 productos.
Generar embeddings CLIP de ~70k imágenes+textos en CPU puede tardar
varias horas. Opciones:

- **Recomendado para desarrollo/corrección rápida**: indexar un
  subconjunto representativo.
  ```bash
  python scripts/build_index.py --max-docs 8000
  ```
- **Para GPU** (mucho más rápido): setear `CLIP_DEVICE=cuda` (por
  ejemplo en Google Colab con GPU habilitada).
  ```bash
  CLIP_DEVICE=cuda python scripts/build_index.py
  ```
- **Solo texto** (sin descargar/embeber imágenes, más rápido, para
  probar rápidamente el resto del pipeline):
  ```bash
  python scripts/build_index.py --skip-images --max-docs 2000
  ```

---

## 4. Generar consultas de evaluación (qrels)

```bash
python scripts/generate_qrels.py
```

Genera `data/eval/eval_queries.json` y `data/eval/qrels.json` de forma
programática (no hay juicios de relevancia humanos preexistentes para
este catálogo), combinando dos estrategias:

- **Consultas por artista/sello** (ej. "discos de Pink Floyd"): son
  relevantes los documentos cuyo campo estructurado `store` coincide
  exactamente con ese artista — juicio de relevancia no ambiguo.
- **Consultas temáticas por keyword** (ej. "música de jazz"): son
  relevantes los documentos cuyo título o descripción contienen esa
  palabra — proxy estándar de relevancia temática cuando no hay
  anotación manual disponible.

> Si se indexó solo un subconjunto del corpus (`--max-docs`), correr
> `generate_qrels.py` también con `--max-docs` del mismo tamaño para
> que las consultas generadas tengan documentos relevantes presentes
> en el índice.

---

## 5. Evaluar el sistema

```bash
python evaluate.py                # con reranking (configuración por defecto)
python evaluate.py --no-rerank     # solo FAISS + CLIP, sin reranking
python evaluate.py --k 5 10 20
```

Imprime una tabla por consulta y el promedio macro de **Precision@k**,
**Recall@k** y **NDCG@k**, y guarda el detalle completo en
`data/eval/eval_results.json`. Correr con y sin `--no-rerank` permite
comparar cuantitativamente el aporte del reranking para el informe.

---

## 6. Ejecutar la interfaz de chat

```bash
streamlit run app.py
```

Abre automáticamente `http://localhost:8501`. La interfaz permite:

- hacer consultas conversacionales en lenguaje natural;
- ver la respuesta generada por el LLM;
- inspeccionar las **evidencias** usadas (documento, imagen, score de
  recuperación, score de reranking) para verificar la trazabilidad
  entre recuperación y generación;
- activar/desactivar desde la barra lateral: query expansion,
  reranking, memoria conversacional y relevance feedback;
- votar 👍/👎 sobre las evidencias mostradas, lo que ajusta (Rocchio)
  las búsquedas siguientes dentro de la misma sesión.

---

## 7. Funcionalidades de excelencia implementadas

| Funcionalidad | Módulo | Descripción |
|---|---|---|
| Re-ranking (+15) | `src/reranker.py` | Cross-encoder `ms-marco-MiniLM-L-6-v2` sobre los Top-K de FAISS, combinado con el score vectorial original. |
| Query Expansion (+15) | `src/query_expansion.py` | Expansión vía Gemini API (sinónimos/géneros/artistas relacionados), con fallback heurístico sin necesidad de API key. |
| Relevance Feedback (+15) | `src/feedback.py` | Algoritmo de Rocchio sobre los embeddings CLIP a partir de votos 👍/👎 en la sesión. |
| Memoria conversacional (+15) | `src/memory.py` | Condensación de preguntas de seguimiento en consultas autocontenidas usando el historial del chat (patrón "standalone question"). |

---

## 8. Limitaciones conocidas / decisiones de diseño

- El campo `store` del JSONL original viene con ruido (p. ej.
  `"Pink Floyd Format: Audio CD"`); `data_loader.py` y
  `generate_qrels.py` lo limpian antes de usarlo.
- Si la descarga de una imagen falla (URL rota, sin conexión), el
  documento se indexa igual usando solo su embedding de texto
  (fallback automático) — el sistema no descarta documentos por
  imágenes faltantes.
- El modelo Gemini por defecto es `gemini-2.5-flash` (estable/económico
  a la fecha). Se puede cambiar con `GEMINI_MODEL=gemini-3.5-flash`
  para mayor capacidad a mayor costo.
