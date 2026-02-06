import os
from pathlib import Path

import streamlit as st

from rag import RAGSystem


def ensure_upload_dir() -> Path:
    """Create and return the upload directory path."""
    upload_dir = Path("uploaded_docs")
    upload_dir.mkdir(exist_ok=True)
    return upload_dir


def build_rag_system(
    uploaded_files,
    chunk_size: int,
    chunk_overlap: int,
    k: int,
    temperature: float,
    model_embedding: str,
    model_llm: str,
    preamble: str,
):
    """
    Instantiate a RAGSystem from uploaded files and user-selected parameters.

    This helper does NOT expose db_name or vector DB controls in the UI; it just
    builds a fresh RAGSystem for the current set of PDFs and parameters.
    """
    if not uploaded_files:
        return None

    upload_dir = ensure_upload_dir()
    file_paths: list[str] = []

    # Save uploaded PDFs to disk so PyPDFLoader can read them
    for uploaded in uploaded_files:
        filename = uploaded.name
        dest_path = upload_dir / filename
        with open(dest_path, "wb") as f:
            f.write(uploaded.getbuffer())
        file_paths.append(str(dest_path))

    # Prompt template: user-controlled preamble + fixed context/question section
    prompt_template = f"""{preamble.strip()}

Context:
{{context}}

Question:
{{question}}"""

    kwargs = {
        "file_path": file_paths if len(file_paths) > 1 else file_paths[0],
        "prompt_template": prompt_template,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "search_kwargs": {"k": k},
        "temperature": temperature,
        "model_embedding": model_embedding,
        "model_llm": model_llm,
    }

    rag = RAGSystem(**kwargs)
    return rag


def main():
    st.set_page_config(page_title="RAG PDF Assistant", layout="wide")
    st.title("📄 RAG PDF Assistant")
    st.write(
        "Upload one or more PDF files, tune the Retrieval-Augmented Generation (RAG) "
        "parameters, and ask questions grounded in those documents."
    )

    with st.sidebar:
        st.header("RAG Parameters")

        chunk_size = st.slider("Chunk size", min_value=100, max_value=10000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=10000, value=100, step=50)
        k = st.slider("Top-k retrieved chunks", min_value=1, max_value=20, value=3, step=1)
        temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

        st.subheader("Models")
        model_embedding = st.text_input("Embedding model", value="gemini-embedding-001")
        model_llm = st.text_input("Chat model", value="gemini-2.5-flash")

    # Editable prompt preamble (part before Context / Question)
    default_preamble = """You are a helpful assistant.
You answer questions strictly based on the content of the provided documents.
If the documents do not contain the answer, say that you cannot find it."""
    preamble = st.text_area(
        "System prompt / instructions (preamble before Context and Question):",
        value=default_preamble,
        height=120,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload one or more PDFs. They'll be embedded into a Chroma vector store.",
    )

    st.markdown("---")
    st.header("Ask Questions")

    question = st.text_area("Your question about the uploaded document(s):", height=100)
    ask_button = st.button("Get Answer")

    if ask_button:
        if not uploaded_files:
            st.error("Please upload at least one PDF before asking a question.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Building knowledge base and generating answer..."):
                rag = build_rag_system(
                    uploaded_files=uploaded_files,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    k=k,
                    temperature=temperature,
                    model_embedding=model_embedding,
                    model_llm=model_llm,
                    preamble=preamble,
                )
                if rag is None:
                    st.error("Failed to initialize RAG system. Please check logs and configuration.")
                    return
                answer = rag.ask(question.strip())
            if answer is None:
                st.error("Failed to generate an answer. Please check logs and configuration.")
            else:
                st.subheader("Answer")
                st.write(answer)


if __name__ == "__main__":
    main()

