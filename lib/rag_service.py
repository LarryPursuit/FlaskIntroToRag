from __future__ import annotations

import re
from typing import Any

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "the",
    "their",
    "to",
    "use",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
    "you",
    "your",
}


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
NO_CONTEXT_MESSAGE = "No relevant context was found in the approved company documents."


def tokenize(text: str) -> set[str]:
    """Convert text into a set of searchable lowercase tokens."""
    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    cleaned_tokens = (token.strip("'") for token in raw_tokens)
    return {
        token
        for token in cleaned_tokens
        if len(token) > 1 and token not in STOPWORDS
    }


def document_search_text(document: dict[str, Any]) -> str:
    """Combine searchable document fields into one text value."""
    return " ".join(
        [
            document.get("title", ""),
            document.get("category", ""),
            " ".join(document.get("tags", [])),
            document.get("text", ""),
        ]
    )


def score_document(query: str, document: dict[str, Any]) -> dict[str, Any]:
    """Score a document using keyword overlap, with a small boost for title matches."""
    query_tokens = tokenize(query)
    document_tokens = tokenize(document_search_text(document))
    title_tokens = tokenize(document.get("title", ""))

    matched_terms = sorted(query_tokens & document_tokens)
    title_boost = 0.5 * len(query_tokens & title_tokens)

    return {
        "document": document,
        "score": len(matched_terms) + title_boost,
        "matched_terms": matched_terms,
    }


def retrieve_context(
    query: str,
    documents: list[dict[str, Any]],
    limit: int = 2,
    minimum_score: float = 1.0,
) -> list[dict[str, Any]]:
    """Select the most relevant documents for the query."""
    scored = [score_document(query, document) for document in documents]
    relevant = [match for match in scored if match["score"] >= minimum_score]
    ranked = sorted(relevant, key=lambda match: match["score"], reverse=True)
    return ranked[:limit]


def format_context(context_matches: list[dict[str, Any]]) -> str:
    """Format retrieved documents into a context block for the prompt."""
    if not context_matches:
        return NO_CONTEXT_MESSAGE

    blocks = [
        "\n".join(
            [
                f"Source ID: {match['document']['id']}",
                f"Title: {match['document']['title']}",
                f"Category: {match['document']['category']}",
                f"Content: {match['document']['text']}",
            ]
        )
        for match in context_matches
    ]
    return "\n\n---\n\n".join(blocks)


def build_prompt(query: str, context_matches: list[dict[str, Any]]) -> str:
    """Build a structured prompt with instructions, context, question, and requirements."""
    return f"""Instructions:
You are an internal company assistant answering employee and developer questions.
Use only the information in the Context section below.
Do not invent policies, numbers, steps, or other details that are not supported by the context.
If the context does not contain enough information, say so clearly.

Context:
{format_context(context_matches)}

Question:
{query}

Response requirements:
- Answer in a few clear, concise sentences.
- Base every statement on the provided context.
- Cite the Source ID(s) you used at the end of the answer.
"""


def source_metadata(match: dict[str, Any]) -> dict[str, str]:
    """Return source information that is safe to expose in the API response."""
    document = match["document"]
    return {"id": document["id"], "title": document["title"]}
