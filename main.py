"""Streamlit UI for Swiss AI Office Assistant ch (multilingual)

- Chat UI with Streamlit
- Retrieval via Chroma using OpenAI embeddings
- Answers via OpenAI chat models
- Multilingual small talk and responses in user language (EN/DE/FR/IT)
"""
from __future__ import annotations

import os
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from retriver import get_vectorstore, ensure_index_built

APP_TITLE = "Swiss AI Office Assistant ch"
FOOTER_TEXT = "Built with OpenAI API + LangChain + Streamlit"

DEFAULT_MODEL_CANDIDATES = [
    os.getenv("OPENAI_MODEL", "gpt-5-nano"),
    "gpt-5-mini",
    "gpt-4o-mini",
]

SYSTEM_PROMPT = (
    "You are Swiss AI Office Assistant, an internal HR bot for emloyees.\n"
    "Answer clearly, politely, and in a professional corporate tone.\n"
    "If You're unsure, say so and siggest contacting HR."
)

FALLBACK_BY_LANG = {
    "en": "I'm not fully sure - please contact HR at hr@company.ch",
    "de": "Ich bin nicht ganz sicher – bitte wenden Sie sich an HR unter hr@company.ch.",
    "fr": "Je ne suis pas totalement sûr — veuillez contacter les RH à hr@company.ch.",
    "it": "Non sono del tutto sicuro — contatti le Risorse Umane a hr@company.ch.",
}

LANG_DISPLAY = {
    "en": "English",
    "de": "German (Swiss professional tone)",
    "fr": "French (Swiss professional tone)",
    "it": "Italian (Swiss professional tone)",
}


def detect_lang(text: str) -> str:
    t = (text or "").lower()
    # Keyword hints
    if any(w in t for w in ["grüezi", "gruezi", "hallo", "guten", "danke", "bitte", "tschüss", "servus", "hoi", "salü", "sali"]):
        return "de"
    if any(w in t for w in ["bonjour", "salut", "bonsoir", "merci", "s'il vous plaît", "svp"]):
        return "fr"
    if any(w in t for w in ["ciao", "buongiorno", "buonasera", "grazie", "per favore"]):
        return "it"
    # Character hints
    if any(ch in t for ch in "äöüß"):  # German charset
        return "de"
    if any(ch in t for ch in "éèêàçùûîôëïœ"):  # French charset
        return "fr"
    if any(ch in t for ch in "àèéìîòóù" ):  # Italian charset (rough)
        return "it"
    return "en"


def small_talk_response(text: str, lang: str) -> str | None:
    t = (text or "").strip().lower()
    # Greetings per lang
    greet_map = {
        "en": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"],
        "de": ["grüezi", "gruezi", "hallo", "guten morgen", "guten tag", "guten abend", "hoi", "salü", "sali"],
        "fr": ["bonjour", "salut", "bonsoir"],
        "it": ["ciao", "buongiorno", "buonasera"],
    }
    thanks_map = {
        "en": ["thanks", "thank you", "thx", "ty"],
        "de": ["danke", "dankeschön", "merci"],
        "fr": ["merci"],
        "it": ["grazie"],
    }
    bye_map = {
        "en": ["bye", "goodbye", "see you", "see ya"],
        "de": ["tschüss", "auf wiedersehen"],
        "fr": ["au revoir"],
        "it": ["arrivederci", "ciao"],
    }

    def starts_with_any(words: list[str]) -> bool:
        return any(t.startswith(w) or t == w for w in words)

    if starts_with_any(greet_map.get(lang, [])):
        return {
            "en": "Hello — how can I help you today?",
            "de": "Grüezi — wie kann ich Ihnen helfen?",
            "fr": "Bonjour — comment puis-je vous aider?",
            "it": "Buongiorno — come posso aiutarla?",
        }.get(lang)
    if any(w in t for w in thanks_map.get(lang, [])):
        return {
            "en": "You're welcome. How else can I assist?",
            "de": "Gern geschehen. Womit kann ich sonst helfen?",
            "fr": "Je vous en prie. Puis-je vous aider autrement?",
            "it": "Prego. In cos'altro posso aiutarla?",
        }.get(lang)
    if starts_with_any(bye_map.get(lang, [])):
        return {
            "en": "Goodbye. If anything comes up, feel free to ask.",
            "de": "Auf Wiedersehen. Bei Fragen melden Sie sich gerne.",
            "fr": "Au revoir. En cas de besoin, n'hésitez pas à me solliciter.",
            "it": "Arrivederci. Se serve altro, sono a disposizione.",
        }.get(lang)
    return None


def call_chat_model(model: str, system_prompt: str, user_content: str) -> str:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return resp.choices[0].message.content or ""


def init_page() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🇨🇭", layout="centered")
    st.title(APP_TITLE)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore_ready" not in st.session_state:
        st.session_state.vectorstore_ready = False


def pick_available_model() -> Tuple[str | None, str | None]:
    for m in DEFAULT_MODEL_CANDIDATES:
        if not m:
            continue
        return m, None
    return None, "No model configured. Please set OPENAI_MODEL or use defaults."


def render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def format_context_block(chunks: List[Tuple[str, str]]) -> str:
    if not chunks:
        return ""
    lines = ["Context (top relevant excerpts):"]
    for i, (src, txt) in enumerate(chunks, start=1):
        lines.append(f"[{i}] Source: {src}\n{txt}")
    return "\n\n".join(lines)


def main() -> None:
    load_dotenv(override=True)
    init_page()
    init_state()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning(
            "OPENAI_API_KEY is not set. Load from .env or environment to enable answers.",
            icon="⚠️",
        )

    with st.spinner("Loading knowledge base..."):
        vs = get_vectorstore()
        if not vs:
            try:
                ensure_index_built()
            except Exception as e:
                st.info(f"Index build pending: {e}")
            vs = get_vectorstore()
        st.session_state.vectorstore_ready = vs is not None

    render_chat_history()

    user_input = st.chat_input("Ask a question about HR or IT policies...")
    if not user_input:
        st.markdown(
            f"<div style='text-align:center; color:#6b7280; font-size:0.9rem; margin-top:2rem'>{FOOTER_TEXT}</div>",
            unsafe_allow_html=True,
        )
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            lang = detect_lang(user_input)

            # Small talk in user's language
            canned = small_talk_response(user_input, lang)
            if canned:
                st.markdown(canned)
                st.session_state.messages.append({"role": "assistant", "content": canned})
                return

            if not st.session_state.vectorstore_ready:
                st.error("Vector store not ready. Please try again.")
                return

            try:
                docs_scores = vs.similarity_search_with_relevance_scores(user_input, k=3)
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                return

            relevant_pairs: List[Tuple[str, str]] = []
            max_score = 0.0
            for doc, score in docs_scores:
                max_score = max(max_score, float(score or 0.0))
                source = doc.metadata.get("source") or doc.metadata.get("file_path") or "unknown"
                relevant_pairs.append((os.path.basename(source), doc.page_content.strip()))

            RELEVANCE_THRESHOLD = 0.2
            if max_score < RELEVANCE_THRESHOLD:
                st.markdown(FALLBACK_BY_LANG.get(lang, FALLBACK_BY_LANG["en"]))
                st.session_state.messages.append({"role": "assistant", "content": FALLBACK_BY_LANG.get(lang, FALLBACK_BY_LANG["en"])})
                return

            context_block = format_context_block(relevant_pairs)
            user_block = (
                "Using the provided context excerpts, answer the user's question.\n"
                "- Be concise, polite, and professional.\n"
                "- If the context is insufficient to answer confidently, say you are not fully sure and suggest contacting HR at hr@company.ch.\n"
                f"- Respond in {LANG_DISPLAY.get(lang, 'English')} appropriate for Switzerland.\n\n"
                f"{context_block}\n\n"
                f"User question: {user_input}"
            )

            model_name, warn = pick_available_model()

            response_text = None
            last_error = None
            if not api_key:
                response_text = FALLBACK_BY_LANG.get(lang, FALLBACK_BY_LANG["en"])  # cannot call model
            else:
                for candidate in [m for m in DEFAULT_MODEL_CANDIDATES if m]:
                    try:
                        response_text = call_chat_model(candidate, SYSTEM_PROMPT, user_block)
                        break
                    except Exception as e:
                        last_error = e
                        continue

            if response_text is None:
                err_msg = f"Model call failed. {last_error!s}" if last_error else "Model call failed."
                st.error(err_msg)
                response_text = FALLBACK_BY_LANG.get(lang, FALLBACK_BY_LANG["en"]) 

            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

    st.markdown(
        f"<div style='text-align:center; color:#6b7280; font-size:0.9rem; margin-top:2rem'>{FOOTER_TEXT}</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
