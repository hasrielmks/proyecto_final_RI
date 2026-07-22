"""
app.py
======
Interfaz web conversacional (Streamlit) del sistema de RI Multimodal
con RAG. Cumple los requisitos de UI del enunciado:

  - consultas conversacionales (chat)
  - visualización de la respuesta del asistente
  - visualización de los documentos e imágenes usados como contexto
    (evidencias), con su score de recuperación
  - además: memoria conversacional, expansión de consultas, reranking
    y relevance feedback (me gusta / no me gusta) configurables desde
    la barra lateral.

Ejecutar con:
    streamlit run app.py
"""

from __future__ import annotations

import os

import streamlit as st

import config
from src import feedback as feedback_mod
from src import generator, memory as memory_mod, rag_pipeline, vector_store

st.set_page_config(
    page_title="Catálogo Musical · RI Multimodal",
    page_icon="🎵",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Estilos mínimos: paleta inspirada en vinilos/fichas de catálogo de tienda
# de discos (crema + tinta + acento vermellón-vinilo), tipografía serif para
# títulos de álbum y monoespaciada para los scores (ficha técnica).
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #12100E; }
    h1, h2, h3 { font-family: Georgia, 'Times New Roman', serif; color: #F2E9DC; }
    p, div, span, li { color: #E4DACB; }
    .evidence-card {
        background-color: #1D1A16;
        border: 1px solid #3A342B;
        border-radius: 6px;
        padding: 0.9rem;
        margin-bottom: 0.7rem;
    }
    .evidence-score {
        font-family: 'Courier New', monospace;
        color: #C97A4A;
        font-size: 0.82rem;
    }
    .evidence-title { font-family: Georgia, serif; font-size: 1.02rem; color: #F2E9DC; }
    .badge {
        display: inline-block; padding: 0.1rem 0.5rem; border-radius: 10px;
        background-color: #2A241C; color: #C97A4A; font-size: 0.72rem;
        margin-right: 0.3rem; font-family: 'Courier New', monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[{"role","content","evidences"}]
if "memory" not in st.session_state:
    st.session_state.memory = memory_mod.ConversationMemory()
if "feedback" not in st.session_state:
    st.session_state.feedback = feedback_mod.FeedbackStore()


@st.cache_resource(show_spinner="Cargando índice FAISS...")
def load_store():
    if not config.FAISS_INDEX_PATH.exists():
        return None
    return vector_store.VectorStore.load()


store = load_store()

# ---------------------------------------------------------------------------
# Barra lateral: configuración y estado del sistema
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎵 RI Multimodal · Configuración")

    api_key_set = bool(os.environ.get(config.GEMINI_API_KEY_ENV_VAR))
    if not api_key_set:
        st.warning(
            f"No se encontró `{config.GEMINI_API_KEY_ENV_VAR}`. El sistema puede "
            "recuperar evidencias pero no generará respuestas del LLM.",
            icon="⚠️",
        )
        manual_key = st.text_input("Pegar GEMINI_API_KEY (solo esta sesión)", type="password")
        if manual_key:
            os.environ[config.GEMINI_API_KEY_ENV_VAR] = manual_key
            st.rerun()
    else:
        st.success("Gemini API configurada.", icon="✅")

    st.divider()
    st.markdown("**Funcionalidades de excelencia**")
    use_query_expansion = st.toggle("Query expansion", value=config.QUERY_EXPANSION_ENABLED_DEFAULT)
    use_reranking = st.toggle("Reranking (cross-encoder)", value=config.RERANK_ENABLED_DEFAULT)
    use_memory = st.toggle("Memoria conversacional", value=config.MEMORY_ENABLED_DEFAULT)
    use_feedback = st.toggle("Relevance feedback (Rocchio)", value=config.FEEDBACK_ENABLED_DEFAULT)

    st.divider()
    top_k_display = st.slider("Evidencias a mostrar (Top-k)", 1, 10, config.TOP_K_DISPLAY)

    st.divider()
    if store is not None:
        st.caption(f"📚 Índice cargado: **{store.index.ntotal}** documentos")
    else:
        st.error(
            "No se encontró el índice FAISS. Corré primero:\n\n"
            "`python scripts/build_index.py`",
            icon="🚫",
        )

    if st.button("🗑️ Reiniciar conversación y feedback"):
        st.session_state.chat_history = []
        st.session_state.memory = memory_mod.ConversationMemory()
        st.session_state.feedback = feedback_mod.FeedbackStore()
        st.rerun()

st.title("🎵 Asistente de Catálogo Musical")
st.caption(
    "Sistema de Recuperación de Información Multimodal con RAG · CLIP + FAISS + Gemini"
)

if store is None:
    st.stop()

pipeline = rag_pipeline.RagPipeline(store)


# ---------------------------------------------------------------------------
# Render de evidencias (documentos + imágenes + score + feedback)
# ---------------------------------------------------------------------------
def render_evidences(evidences: list[dict], turn_key: str) -> None:
    st.markdown("**Evidencias utilizadas**")
    cols = st.columns(min(len(evidences), 3) or 1)
    for i, ev in enumerate(evidences[:top_k_display]):
        col = cols[i % len(cols)]
        with col:
            st.markdown('<div class="evidence-card">', unsafe_allow_html=True)
            if ev.get("image_url"):
                st.image(ev["image_url"], use_container_width=True)
            st.markdown(f'<div class="evidence-title">{ev.get("title","(sin título)")}</div>', unsafe_allow_html=True)
            badges = f'<span class="badge">rank {ev.get("rank","?")}</span>'
            if "final_score" in ev:
                badges += f'<span class="badge">score {ev["final_score"]:.3f}</span>'
            if "rerank_score" in ev:
                badges += f'<span class="badge">rerank {ev["rerank_score"]:.2f}</span>'
            st.markdown(badges, unsafe_allow_html=True)
            if ev.get("store"):
                st.caption(ev["store"][:80])

            vote_key_base = f"{turn_key}_{ev['doc_id']}"
            like_col, dislike_col = st.columns(2)
            current_vote = st.session_state.feedback.vote_for(ev["doc_id"])
            with like_col:
                if st.button(
                    "👍 Me gusta" if current_vote != "like" else "👍 ✓",
                    key=f"like_{vote_key_base}",
                    use_container_width=True,
                ):
                    vec = store.get_vector(ev["doc_id"])
                    if vec is not None:
                        st.session_state.feedback.like(ev["doc_id"], vec)
                    st.rerun()
            with dislike_col:
                if st.button(
                    "👎 No me gusta" if current_vote != "dislike" else "👎 ✓",
                    key=f"dislike_{vote_key_base}",
                    use_container_width=True,
                ):
                    vec = store.get_vector(ev["doc_id"])
                    if vec is not None:
                        st.session_state.feedback.dislike(ev["doc_id"], vec)
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Historial de chat
# ---------------------------------------------------------------------------
for i, turn in enumerate(st.session_state.chat_history):
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("evidences"):
            render_evidences(turn["evidences"], turn_key=f"hist{i}")

# ---------------------------------------------------------------------------
# Input de chat
# ---------------------------------------------------------------------------
user_query = st.chat_input("Preguntá algo sobre el catálogo musical...")
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Buscando y generando respuesta..."):
            response = pipeline.answer(
                user_query,
                memory=st.session_state.memory if use_memory else None,
                session_feedback=st.session_state.feedback if use_feedback else None,
                use_query_expansion=use_query_expansion,
                use_reranking=use_reranking,
                top_k_final=max(top_k_display, config.TOP_K_RERANK),
            )
        st.markdown(response.answer)

        with st.expander("🔍 Detalles de recuperación", expanded=False):
            st.write(f"**Consulta original:** {user_query}")
            if response.used_memory:
                st.write(f"**Consulta tras memoria conversacional:** {response.standalone_query}")
            if response.used_query_expansion:
                st.write(f"**Consulta expandida:** {response.expanded_query}")
            st.write(
                f"Reranking: {'sí' if response.used_reranking else 'no'} · "
                f"Relevance feedback aplicado: {'sí' if response.used_feedback else 'no'}"
            )

        turn_key = f"live{len(st.session_state.chat_history)}"
        if response.evidences:
            render_evidences(response.evidences, turn_key=turn_key)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": response.answer, "evidences": response.evidences}
    )
