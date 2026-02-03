import streamlit as st
from rag.ingest import build_index_from_pdf
from rag.generator import generate_answer

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    layout="wide"
)

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------
defaults = {
    "index": None,
    "chunks": None,
    "chat_history": [],
    "debug_mode": False,
    "last_context": None,
    "last_answer": None,
    "last_confidence": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------------------------------
# Helper: confidence label + explanation
# --------------------------------------------------
def confidence_label(score: int):
    if score >= 70:
        return "High", "Answer is strongly supported by the document content."
    elif score >= 40:
        return "Medium", "Answer is supported, but matched via partial or keyword-based retrieval."
    else:
        return "Low", "Answer relevance is weak or inferred from limited context."

# --------------------------------------------------
# Sidebar UI
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 📄 Upload PDF")

    uploaded = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    if uploaded:
        with st.status("Indexing document...", expanded=True):
            index, chunks = build_index_from_pdf(uploaded)

        st.session_state.index = index
        st.session_state.chunks = chunks
        st.success("Document indexed successfully")

    st.divider()

    # Debug toggle
    st.session_state.debug_mode = st.toggle("Debug mode")

    st.divider()

    # Clear chat
    if st.button("🧹 Clear chat"):
        st.session_state.chat_history = []
        st.session_state.last_context = None
        st.session_state.last_answer = None
        st.session_state.last_confidence = None
        st.experimental_rerun()

    # Reset everything
    if st.button("♻️ Reset document & cache"):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.experimental_rerun()

# --------------------------------------------------
# Main UI
# --------------------------------------------------
st.markdown("## 🧠 Enterprise RAG Assistant")
st.caption("Secure · Transparent · Retrieval-Augmented")

# --------------------------------------------------
# Chat history
# --------------------------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("confidence") is not None:
            label, explanation = confidence_label(msg["confidence"])
            st.caption(f"Confidence: **{label}**")

            with st.expander("Why this confidence?"):
                st.markdown(explanation)

# --------------------------------------------------
# Chat input
# --------------------------------------------------
query = st.chat_input("Ask a question about the document...")

if query:
    # User message
    st.session_state.chat_history.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Guard: no document uploaded
    if st.session_state.index is None:
        with st.chat_message("assistant"):
            st.markdown("Please upload a PDF first.")
        st.stop()

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = generate_answer(
                query=query,
                index=st.session_state.index,
                chunks=st.session_state.chunks,
                session_state=st.session_state,
                debug=st.session_state.debug_mode
            )

            label, explanation = confidence_label(result["confidence"])

            st.markdown(result["answer"])
            st.caption(f"Confidence: **{label}**")

            with st.expander("Why this confidence?"):
                st.markdown(explanation)

            if st.session_state.debug_mode and result.get("debug"):
                with st.expander("Debug info"):
                    st.json(result["debug"])

    # Save assistant message
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "confidence": result["confidence"]
        }
    )
