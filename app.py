"""Beginner-friendly Streamlit RAG app for chatting with one uploaded PDF or CSV."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ingest import load_documents, split_documents

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


load_dotenv()

st.set_page_config(page_title="Chat with Your Data", page_icon="📄")


def check_app_password() -> bool:
    """Require a shared password before the app can be used."""
    expected_password = os.getenv("APP_PASSWORD")

    if not expected_password:
        return True

    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False

    if st.session_state.is_authenticated:
        return True

    st.title("Chat with Your Data")
    st.write("Enter the app password to continue.")

    entered_password = st.text_input("App password", type="password")

    if st.button("Unlock App"):
        if entered_password == expected_password:
            st.session_state.is_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    st.info("Please reach out to Adrian to get the app password.")
    return False


def save_uploaded_file(uploaded_file) -> str:
    """Save the uploaded file to a temporary file and return its path."""
    suffix = Path(uploaded_file.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def build_vectorstore_from_file(file_path: str):
    """Load the file, split it, and build an in-memory FAISS index."""
    documents = load_documents(file_path)
    if not documents:
        raise ValueError("No readable content was found in the uploaded file.")

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("The file was loaded, but no text chunks were created.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(documents), len(chunks)


def retrieve_context(vectorstore, question: str, k: int = 4):
    """Fetch the top-k most relevant chunks for a user question."""
    return vectorstore.similarity_search(question, k=k)


def get_source_label(doc) -> str:
    """Return a human-friendly label for PDF pages or CSV rows."""
    if "page" in doc.metadata:
        return f"Page {doc.metadata['page'] + 1}"
    if "row" in doc.metadata:
        return f"Row {doc.metadata['row']}"
    return "Source"


def build_context_text(documents) -> str:
    """Turn retrieved chunks into one prompt-friendly context block."""
    context_parts = []
    for index, doc in enumerate(documents, start=1):
        source_label = get_source_label(doc)
        context_parts.append(
            f"Source {index} | {source_label}\n{doc.page_content.strip()}"
        )
    return "\n\n".join(context_parts)


def answer_question(question: str, retrieved_docs):
    """Ask the LLM to answer only from the retrieved file context."""
    prompt = ChatPromptTemplate.from_template(
        """You are a careful assistant answering questions about an uploaded document.

Use only the context below.
If the answer is not clearly in the context, say:
"I couldn't find the answer in the uploaded document."

Give a short, beginner-friendly answer. Do not make up facts.

Question:
{question}

Context:
{context}
"""
    )

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    chain = prompt | llm
    response = chain.invoke(
        {
            "question": question,
            "context": build_context_text(retrieved_docs),
        }
    )
    return response.content


def show_sources(retrieved_docs):
    """Display the retrieved source chunks and their page numbers or row numbers."""
    st.subheader("Source Excerpts")
    for index, doc in enumerate(retrieved_docs, start=1):
        source_label = get_source_label(doc)
        excerpt = doc.page_content.strip().replace("\n", " ")
        excerpt = excerpt[:400] + "..." if len(excerpt) > 400 else excerpt

        st.markdown(f"**Source {index} | {source_label}**")
        st.write(excerpt)


def main():
    if not check_app_password():
        return

    st.title("Chat with Your Data")
    st.write(
        "Upload one PDF or CSV, ask a question, and get an answer grounded in the file."
    )

    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "OPENAI_API_KEY is missing. Add it to your .env file before running the app."
        )
        st.stop()

    uploaded_file = st.file_uploader("Upload a PDF or CSV", type=["pdf", "csv"])

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
        st.session_state.file_name = None

    if uploaded_file is not None:
        is_new_file = st.session_state.file_name != uploaded_file.name

        if is_new_file:
            try:
                with st.spinner("Reading the file and building the search index..."):
                    temp_file_path = save_uploaded_file(uploaded_file)
                    vectorstore, document_count, chunk_count = build_vectorstore_from_file(
                        temp_file_path
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.file_name = uploaded_file.name

                st.success(
                    f"Indexed '{uploaded_file.name}' with {document_count} source sections and {chunk_count} chunks."
                )
            except Exception as error:
                st.session_state.vectorstore = None
                st.error(f"Could not process the file: {error}")

    question = st.text_input("Ask a question about the uploaded file")

    if st.button("Get Answer"):
        if st.session_state.vectorstore is None:
            st.warning("Please upload a PDF or CSV first.")
            st.stop()

        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        try:
            with st.spinner("Searching the file and writing an answer..."):
                retrieved_docs = retrieve_context(st.session_state.vectorstore, question)
                answer = answer_question(question, retrieved_docs)

            st.subheader("Answer")
            st.write(answer)
            show_sources(retrieved_docs)
        except Exception as error:
            st.error(f"Something went wrong while answering the question: {error}")


if __name__ == "__main__":
    main()
