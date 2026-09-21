# 📄 DocAnalyzer AI Pro

<p align="center">
  <b>AI-powered document analysis, semantic search, and Retrieval-Augmented Generation (RAG) system built with LangChain, LangGraph, Google Gemini, and Streamlit.</b>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python"></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.58%2B-red?style=for-the-badge&logo=streamlit" alt="Streamlit"></a>
  <a href="https://www.langchain.com/"><img src="https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge" alt="LangChain"></a>
  <a href="https://langchain-ai.github.io/langgraph/"><img src="https://img.shields.io/badge/LangGraph-Workflow-purple?style=for-the-badge" alt="LangGraph"></a>
  <a href="https://ai.google.dev/"><img src="https://img.shields.io/badge/Google%20Gemini-LLM-orange?style=for-the-badge&logo=google" alt="Gemini"></a>
</p>

---

## 🚀 Overview

**DocAnalyzer AI Pro** is an AI-powered document search and analysis application designed to combine **semantic search, Retrieval-Augmented Generation (RAG), and multimodal document understanding**.

Users can upload PDF, TXT, Markdown, and image-based documents, then interact with the content through a conversational interface.

The application uses **LangChain** to build the RAG components and **LangGraph** to orchestrate the overall question-answering workflow.

### Core Workflow

```text
                    Document
                       │
             ┌─────────┴─────────┐
             │                   │
        Text Document       Scanned / Image
             │                   │
        Text Extraction      Gemini Vision
             │                   │
             └─────────┬─────────┘
                       ↓
                 Text Chunks
                       ↓
              Gemini Embeddings
                       ↓
                 FAISS Index
                       ↓
                  User Query
                       ↓
              LangGraph Workflow
                       ↓
                 Retrieval
                       ↓
                Context Builder
                       ↓
                 Gemini LLM
                       ↓
                    Answer
```

---

## ✨ Key Features

### 🧠 Retrieval-Augmented Generation

* Document-based question answering using RAG.
* Semantic search using **Gemini Embeddings**.
* Vector similarity search with **FAISS**.
* Context-aware answer generation using Google Gemini.
* Configurable **Top-K retrieval**.
* Document metadata and source information are preserved throughout the pipeline.

### 🔗 LangChain Integration

LangChain is used as the core framework for building the RAG pipeline:

* `Document` objects for standardized document representation.
* `RecursiveCharacterTextSplitter` for document chunking.
* `GoogleGenerativeAIEmbeddings` for vector embeddings.
* FAISS vector store for similarity search.
* Retriever abstraction for relevant context retrieval.
* `ChatGoogleGenerativeAI` for LLM integration.

### 🔄 LangGraph Workflow

LangGraph is used to orchestrate the RAG workflow as a state-based graph.

```text
User Query
    ↓
Query Analysis
    ↓
Document Retrieval
    ↓
Context Construction
    ↓
Answer Generation
    ↓
Final Response
```

This architecture makes the application easier to extend with additional routing logic, validation steps, tools, or agent-based workflows.

### 🔎 Semantic Search

The system converts document chunks and user queries into vector representations and retrieves semantically relevant content instead of relying only on exact keyword matching.

### 🖼️ Multimodal Document Understanding

The application supports image-based content and scanned PDF pages.

* Detects pages with limited or missing text extraction.
* Extracts images from PDF documents.
* Sends image content to Gemini Vision for analysis.
* Supports document images, diagrams, charts, and other visual content.

### 💬 Interactive AI Chat

* Conversational document question answering.
* Markdown-formatted AI responses.
* Chat history maintained using Streamlit session state.
* Context inspection for retrieved document chunks.
* Configurable model parameters such as temperature and token limits.

### 🎨 Streamlit Interface

* Clean and responsive Streamlit interface.
* Document upload and preview.
* Configurable AI settings through the sidebar.
* Chat-based interaction.
* Retrieved-context inspection.

---

## 🛠️ Tech Stack

| Category               | Technology        |
| ---------------------- | ----------------- |
| Programming Language   | Python            |
| Web Framework          | Streamlit         |
| RAG Framework          | LangChain         |
| Workflow Orchestration | LangGraph         |
| LLM                    | Google Gemini     |
| Embeddings             | Gemini Embeddings |
| Vector Database        | FAISS             |
| Document Processing    | PyPDF             |
| Image Processing       | Pillow            |
| Numerical Computing    | NumPy             |
| Version Control        | Git / GitHub      |

---

## 🤖 AI Models

The application uses Google Gemini models for both text generation and embeddings.

### Generation

```text
Gemini 3.6 Flash
```

### Embeddings

```text
gemini-embedding-001
```

The model configuration can be adjusted according to API availability and application requirements.

---

## 📂 Project Structure

```text
TimKiemNoiDung/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml          # Local secrets - not committed
│
└── uploads/                  # Uploaded documents / temporary files
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/trongquy1811/TimKiemNoiDung.git
cd TimKiemNoiDung
```

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration

The application requires a **Google Gemini API key**.

Create:

```text
.streamlit/secrets.toml
```

with:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
```

The secrets file is intentionally excluded from Git using `.gitignore`.

> **Security:** Never commit API keys, credentials, or other sensitive information to a public repository.

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 💡 Usage

### 1. Upload Documents

Upload supported files such as:

* PDF
* TXT
* Markdown
* PNG
* JPG / JPEG

### 2. Process Documents

The application extracts the document content and divides it into smaller chunks for retrieval.

For image-based content, Gemini Vision can be used to analyze visual information.

### 3. Ask Questions

Example queries:

```text
Summarize the main content of this document.
```

```text
What are the key points discussed in the document?
```

```text
Find all information related to the contract.
```

```text
Explain the information shown in this image.
```

### 4. Inspect Retrieved Context

The application provides access to the retrieved document context so users can inspect the information used by the RAG pipeline to generate the answer.

---

## 🧩 Architecture

The application separates the RAG process into several logical stages:

```text
┌──────────────────────┐
│    Document Input    │
│ PDF / TXT / MD/Image │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Document Processing │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│     Text Chunking    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│ Gemini Embeddings    │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│     FAISS Search     │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│     LangGraph        │
│   RAG Orchestration  │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│    Gemini LLM        │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│    Final Answer      │
└──────────────────────┘
```

---

## 🎯 Project Objectives

This project was developed to explore practical applications of:

* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Large Language Models (LLMs)
* Multimodal AI
* Document Intelligence
* LangChain
* LangGraph
* Vector Search

The project also provides hands-on experience in designing an end-to-end AI application from document ingestion and retrieval to LLM-based answer generation.

---

## 🔮 Future Improvements

Potential extensions include:

* Conversational memory
* Hybrid search combining semantic and keyword retrieval
* Reranking models
* Query rewriting
* Agentic RAG
* Tool calling
* Multi-agent workflows
* OCR pipelines for complex scanned documents
* Persistent vector databases
* Evaluation metrics for RAG quality

---

## 👨‍💻 Author

**Nguyen Trong Quy**

Computer Science | AI Engineer

GitHub: [@trongquy1811](https://github.com/trongquy1811)

---

## 📄 License

This project is intended for educational and portfolio purposes.
