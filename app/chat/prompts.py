"""What the model is told when answering a question.

Kept apart from the code that calls it so the wording can be read, reviewed and
changed without wading through orchestration. Everything here is a pure
function of its arguments, so a test asserts on a string rather than making a
request.

The citation syntax is the load bearing decision. Structured outputs would give
a guaranteed shape, but they cannot be streamed, and an answer over eight
excerpts takes several seconds to write during which the interface would sit
blank. Asking for markers inside ordinary prose keeps the answer streamable and
still leaves something a parser can verify afterwards.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.db.models import Chunk, Message, Role

SYSTEM_PROMPT = """\
You answer questions about corporate annual reports, using only the numbered \
excerpts you are given.

How to cite

Every factual claim needs a citation, written inline as a source number \
followed by a short quotation copied out of that excerpt:

    Shell aims to be a net-zero emissions energy business [3: "net-zero \
emissions energy business by 2050"].

Copy the quotation character for character. It is checked against the source \
text, and a citation that does not match is discarded before the answer is \
shown, so a paraphrase costs you the citation entirely. Keep quotations short, \
a clause or a table row, not a paragraph.

Rules

1. Answer only from the excerpts. If they do not contain the answer, say so \
plainly and say what is missing. That is a useful answer. A guess is not.
2. Never calculate a figure that is not written down. Do not add segments to \
reach a total, convert currencies, or infer a year over year change unless the \
excerpt states it.
3. Always give figures with their unit and their year, because reports state \
several years side by side and a number without a year is not an answer.
4. Note the basis of a figure when it matters. Headcount and full-time \
equivalents are different measures and companies use both.
5. Say which company a figure belongs to. Several reports are indexed at once \
and excerpts from different companies may appear together.
6. Be brief. Answer the question that was asked.
"""


def format_sources(chunks: Sequence[Chunk]) -> str:
    """Render retrieved chunks as numbered excerpts.

    Each carries its company, page and section. The company matters more than
    it looks: excerpts from five different reports can appear in one prompt,
    and an answer that silently mixes two companies' figures is worse than no
    answer.

    Args:
        chunks: The retrieved excerpts, best first.

    Returns:
        The excerpts, numbered from 1 to match the citation syntax.
    """
    return "\n\n".join(
        f"--- source {number} | {chunk.context_header} | page {chunk.page_start}"
        f"{f' to {chunk.page_end}' if chunk.page_end != chunk.page_start else ''}"
        f" ---\n{chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )


def format_history(messages: Sequence[Message], limit: int = 6) -> str:
    """Render recent conversation turns for context.

    Only the last few. A long thread crowds out the excerpts, and the excerpts
    are what the answer has to come from.

    Args:
        messages: The conversation so far, oldest first.
        limit: How many recent turns to include.

    Returns:
        The turns as labelled lines, or an empty string when there are none.
    """
    recent = messages[-limit:]
    if not recent:
        return ""
    lines = "\n".join(
        f"{'User' if message.role is Role.USER else 'Assistant'}: {message.content}"
        for message in recent
    )
    return f"Earlier in this conversation:\n\n{lines}\n\n"


def build_prompt(
    question: str, chunks: Sequence[Chunk], history: Sequence[Message] = ()
) -> str:
    """Assemble the user message for one question.

    Args:
        question: What was asked.
        chunks: The retrieved excerpts, best first.
        history: Earlier turns in this conversation, oldest first.

    Returns:
        The prompt.
    """
    if not chunks:
        return (
            f"{format_history(history)}"
            f"Question: {question}\n\n"
            f"No excerpts were found for this question. Say so, and suggest "
            f"how the question might be reworded."
        )
    return (
        f"{format_history(history)}"
        f"Sources:\n\n{format_sources(chunks)}\n\n"
        f"Question: {question}"
    )


def build_title_prompt(question: str) -> str:
    """Ask for a short label for a conversation.

    Args:
        question: The first question in the thread.

    Returns:
        The prompt.
    """
    return (
        f"Give a title of at most six words for a conversation that starts "
        f"with this question. Reply with the title alone, no quotes, no "
        f"punctuation at the end.\n\n{question}"
    )
