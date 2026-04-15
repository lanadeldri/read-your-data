"""Simple ingestion script for a local FAISS vector store.

Usage:
    python ingest.py /path/to/file.pdf
    python ingest.py /path/to/file.csv

This script:
1. Loads a PDF or CSV file
2. Splits it into smaller chunks
3. Creates OpenAI embeddings
4. Saves a FAISS index locally
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_INDEX_DIR = "faiss_index"
SUPPORTED_EXTENSIONS = {".pdf", ".csv"}


def load_pdf_documents(pdf_path: str):
    """Load all pages from a PDF file."""
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def load_csv_documents(csv_path: str):
    """Turn each CSV row into a searchable document."""
    documents = []

    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)

        if not reader.fieldnames:
            raise ValueError("The CSV file must include a header row.")

        for row_number, row in enumerate(reader, start=1):
            values = []
            for column_name, value in row.items():
                cleaned_value = (value or "").strip()
                if cleaned_value:
                    values.append(f"{column_name}: {cleaned_value}")

            if not values:
                continue

            documents.append(
                Document(
                    page_content=f"Row {row_number}\n" + "\n".join(values),
                    metadata={
                        "source": os.path.basename(csv_path),
                        "row": row_number,
                        "type": "csv",
                    },
                )
            )

    return documents


def load_documents(file_path: str):
    """Load documents based on the uploaded file type."""
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return load_pdf_documents(file_path)
    if extension == ".csv":
        return load_csv_documents(file_path)

    raise ValueError("Unsupported file type. Please use a PDF or CSV file.")


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Split documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(documents)


def build_vectorstore(chunks):
    """Create a FAISS vector store from document chunks."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


def save_vectorstore(vectorstore, output_dir: str):
    """Save the FAISS index to disk."""
    vectorstore.save_local(output_dir)


def ingest_file(file_path: str, output_dir: str = DEFAULT_INDEX_DIR):
    """Main ingestion flow used by the CLI and reusable from other files."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = Path(file_path).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file type. Please use a PDF or CSV file.")

    documents = load_documents(file_path)
    if not documents:
        raise ValueError("No readable content was loaded from the file.")

    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("No text chunks were created from the file.")

    vectorstore = build_vectorstore(chunks)
    save_vectorstore(vectorstore, output_dir)

    return {
        "documents_loaded": len(documents),
        "chunks_created": len(chunks),
        "output_dir": output_dir,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a local FAISS index from a PDF or CSV file."
    )
    parser.add_argument("file_path", help="Path to the PDF or CSV file")
    parser.add_argument(
        "--output",
        default=DEFAULT_INDEX_DIR,
        help=f"Directory to save the FAISS index (default: {DEFAULT_INDEX_DIR})",
    )
    return parser.parse_args()


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is missing. Add it to your .env file before running ingest.py."
        )

    args = parse_args()
    file_path = str(Path(args.file_path).expanduser())

    result = ingest_file(file_path, args.output)

    print("Ingestion complete.")
    print(f"Source sections loaded: {result['documents_loaded']}")
    print(f"Chunks created: {result['chunks_created']}")
    print(f"FAISS index saved to: {result['output_dir']}")


if __name__ == "__main__":
    main()
