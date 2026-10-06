# AI Research Paper Assistant

A simple local AI-based Research Paper Assistant built using:

- Streamlit
- Ollama
- Llama 3.2:3B
- RAG
- ChromaDB
- Sentence Transformers
- PyPDF

The application allows users to upload a research paper in PDF format and interact with it using a local AI model.

---

## Features

- Upload research papers in PDF format
- Extract text from PDF files
- Split paper content into chunks
- Generate embeddings using Sentence Transformers
- Store embeddings in ChromaDB
- Ask questions about the uploaded paper
- Retrieve relevant sections using RAG
- Generate answers using Ollama
- Display supporting page and chunk information
- Generate:
  - Paper Summary
  - Methodology
  - Key Results
  - Keywords
  - Research Gaps
  - Literature Review
- Deep Analysis tools:
  - Dataset Analysis
  - Results Analysis
  - Limitations
  - Future Work
  - Strengths and Weaknesses
  - Research Gaps
- View indexed source chunks
- Chat with the research paper

---

## Project Architecture

```text
Research Paper PDF
        ↓
PDF Text Extraction
        ↓
Text Chunking
        ↓
Sentence Transformer
        ↓
Embeddings
        ↓
ChromaDB
        ↓
User Question
        ↓
Relevant Chunk Retrieval
        ↓
Ollama - Llama 3.2:3B
        ↓
AI Generated Answer
        ↓
Supporting Sources
```
