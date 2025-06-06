# RAG-ResearchPaperAssistant

A Streamlit-based chatbot assistant for research papers using Retrieval-Augmented Generation (RAG). Upload PDFs or search arXiv, ask questions, and get answers grounded in the documents.

---

## Features

- **PDF Upload & QA:** Upload one or more research papers in PDF format and ask questions about their content.
- **arXiv Integration:** Search and summarize papers directly from arXiv.
- **Retrieval-Augmented Generation:** Combines vector search (FAISS) with LLMs for accurate, context-aware answers.
- **Chat Interface:** Interactive chat for both PDF and arXiv-based queries.
- **Source Attribution:** Answers include references to the source document and page.
- **Duplicate Chunk Filtering:** Ensures unique context for each answer.
- **Dockerized:** Easily deployable with Docker and AWS ECR/EC2.
- **Environment Variable Support:** Securely manage API keys and secrets via `.env` files.

---

## Specifications

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Backend:** Python 3.11+
- **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss)
- **Embeddings:** [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- **LLM Integration:** [Groq API](https://console.groq.com/)
- **PDF Parsing:** [PyPDF2](https://pypi.org/project/PyPDF2/)
- **arXiv Loader:** [LangChain ArxivLoader](https://python.langchain.com/docs/integrations/document_loaders/arxiv)
- **Deployment:** Docker, AWS ECR, AWS EC2

---

## Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/ShlokPrajapati/RAG_ResearchPaperAssistant.git
cd rag-researchpaperassistant
```

### 2. Install dependencies

```sh
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
```

### 4. Run locally

```sh
streamlit run app.py
```

---

## Docker Usage

### Build the image

```sh
docker build -t rag-researchpaperassistant:latest .
```

### Run the container

```sh
docker run --env-file .env -p 8501:8501 rag-researchpaperassistant:latest
```

---

## Deploy to AWS

- Push your Docker image to AWS ECR.
- Deploy on EC2 (see project documentation or ask for detailed steps).

---

## Security

- **Never commit your `.env` file or API keys to public repositories.**
- Use `.gitignore` and `.dockerignore` to keep secrets and unnecessary files out of version control and Docker images.

---

## License

MIT License

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [arXiv](https://arxiv.org/)