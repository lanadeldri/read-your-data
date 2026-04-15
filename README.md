# Chat with Your Data

A beginner-friendly RAG (Retrieval-Augmented Generation) app built with Python, Streamlit, LangChain, OpenAI, FAISS, and PyPDF.

The app lets you:

- upload one PDF or CSV
- ask questions about it
- retrieve the most relevant chunks
- answer using only the document context
- view source excerpts with page numbers or row numbers

## Project Files

- `app.py` - Streamlit user interface for PDF and CSV uploads
- `ingest.py` - command-line script to create a local FAISS index from a PDF or CSV
- `requirements.txt` - Python dependencies
- `.env.example` - example environment variable file

## 1. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Add Your OpenAI API Key

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Open `.env` and add your key:

```env
OPENAI_API_KEY=your_openai_api_key_here
APP_PASSWORD=choose_a_shared_password
```

`APP_PASSWORD` is optional for local testing, but recommended if other people will access the app.

## 4. Run `ingest.py`

You can build a local FAISS index from a PDF file with:

```bash
python ingest.py /path/to/your-file.pdf
```

You can also build an index from a CSV file with:

```bash
python ingest.py /path/to/your-file.csv
```

By default, this saves the index in a folder named `faiss_index`.

You can choose a different output folder too:

```bash
python ingest.py /path/to/your-file.csv --output my_index
```

## 5. Run the Streamlit App

Start the app with:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal, usually:

```text
http://localhost:8501
```

If `APP_PASSWORD` is set in `.env`, users must enter that password before they can use the app.

## How the App Works

1. You upload one PDF or CSV in the Streamlit app.
2. PDF text is extracted with PyPDF, or CSV rows are converted into searchable text.
3. The content is split into chunks.
4. OpenAI embeddings are created for the chunks.
5. FAISS stores those embeddings for local search.
6. When you ask a question, the app retrieves the top 4 matching chunks.
7. The model answers only from those retrieved chunks.
8. The app shows the source excerpts and page numbers or row numbers below the answer.

## Test with a Sample File

### Sample PDF

1. Run:

```bash
streamlit run app.py
```

2. Upload a short PDF in the browser.
3. Ask a question that is clearly answered in the document.
4. Ask another question that is not covered by the PDF to test the fallback message.

### Sample CSV

Use a simple CSV like this:

```csv
name,role,location
Alice,Designer,Chicago
Ben,Engineer,Austin
Cara,Recruiter,New York
```

Try questions like:

- "Who is the recruiter?"
- "Which person is in Austin?"
- "What city is Alice in?"
- "What is David's role?"  
  This one should trigger the fallback if David is not in the CSV.

Expected result:

- the app gives a short answer grounded in the uploaded file
- if the answer is missing from the context, it says it could not find the answer in the uploaded document
- source excerpts and page numbers or row numbers appear below the answer

## Notes

- This is a simple MVP designed for local development.
- It uses local FAISS storage instead of a database server.
- It does not include agents, authentication, Docker, or deployment setup.
- A basic password gate is available through the `APP_PASSWORD` environment variable.
- Keep your `.env` file private so the OpenAI API key stays on the machine running the app.
