"""
src/memory.py
==============
Funcionalidad de excelencia: Memoria conversacional (+15).

Implementa el patrón estándar de "reformulación de consulta condensada"
(query condensation) usado en sistemas RAG conversacionales tipo
LangChain ConversationalRetrievalChain:

  1. Se mantiene un historial de turnos (pregunta usuario + respuesta
     del sistema) en la sesión.
  2. Antes de recuperar documentos para un nuevo turno, si hay
     historial reciente, se le pide al LLM que reformule la pregunta
     del usuario en una consulta autocontenida (standalone), resolviendo
     referencias como "ese artista", "el segundo álbum", "y de rock?".
  3. Esa consulta reformulada es la que se usa para expansión + embedding
     + búsqueda vectorial + reranking, en vez de la pregunta cruda.

Si no hay API key configurada (o falla la llamada), se degrada con
gracia concatenando heurísticamente el turno anterior con el actual,
en vez de romper la conversación.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config
from src import generator


@dataclass
class Turn:
    user_query: str
    standalone_query: str
    answer: str


@dataclass
class ConversationMemory:
    """Memoria de una sesión de chat. Una instancia por usuario/sesión
    (en Streamlit se guarda en st.session_state)."""

    turns: list[Turn] = field(default_factory=list)
    max_turns: int = config.MEMORY_MAX_TURNS

    def add_turn(self, user_query: str, standalone_query: str, answer: str) -> None:
        self.turns.append(Turn(user_query, standalone_query, answer))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def _recent_history_text(self) -> str:
        lines = []
        for t in self.turns:
            lines.append(f"Usuario: {t.user_query}")
            lines.append(f"Asistente: {t.answer}")
        return "\n".join(lines)

    def condense_query(self, new_query: str) -> str:
        """Reformula `new_query` en una consulta autocontenida usando el
        historial de la conversación. Si no hay historial, la devuelve
        sin cambios."""
        if self.is_empty():
            return new_query

        history_text = self._recent_history_text()
        prompt = (
            "Historial de la conversación:\n"
            f"{history_text}\n\n"
            f"Nueva pregunta del usuario: {new_query}\n\n"
            "Reformulá la nueva pregunta como una consulta de búsqueda "
            "autocontenida (standalone), resolviendo cualquier referencia "
            "al historial (pronombres, 'ese álbum', 'y de rock', etc). "
            "Si la nueva pregunta ya es autocontenida, devolvela tal cual. "
            "Responde SOLO con la consulta reformulada, sin explicaciones."
        )
        try:
            standalone = generator.generate_raw(prompt, max_output_tokens=80, temperature=0.0)
            return standalone.strip() or new_query
        except Exception:
            # Fallback heurístico: concatenar el último turno con la nueva
            # pregunta le da al embedding CLIP algo de contexto adicional
            # sin depender de la API.
            last_turn = self.turns[-1]
            return f"{last_turn.user_query}. {new_query}"


if __name__ == "__main__":
    mem = ConversationMemory()
    print("Sin historial:", mem.condense_query("dame discos de Grateful Dead"))
    mem.add_turn(
        "dame discos de Grateful Dead",
        "dame discos de Grateful Dead",
        "Encontré 'Dave's Picks Volume 25' de Grateful Dead.",
    )
    # sin API key, cae al fallback heurístico
    print("Con historial (fallback heurístico):", mem.condense_query("y del mismo artista pero en vivo?"))
