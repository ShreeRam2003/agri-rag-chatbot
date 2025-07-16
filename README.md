[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15976381.svg)](https://doi.org/10.5281/zenodo.15976381)

# 🌾 Agricultural Advisory Chatbot using Retrieval-Augmented Generation (RAG)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline designed to provide real-time, context-rich answers to agricultural queries. It leverages dense retrieval with ChromaDB and generative responses using LLMs (via Ollama) to deliver knowledge-grounded recommendations based on *Package of Practices* (PoP) documents for crops like Maize, Sweet Potato, Ragi, Cotton, and Groundnut.

---

## 📘 Abstract

This study introduces a RAG-based system combining FAISS-powered vector search and transformer-based LLMs for agricultural question answering. Preprocessed PoP documents are semantically chunked and embedded using Amazon Titan (`BedrockEmbeddings`) and indexed in ChromaDB. On receiving a query, relevant chunks are retrieved and passed into a prompt that is answered using an LLM (e.g., Mistral, LLaMA2, Phi3). This supports use cases like:

- Crop management recommendations  
- Fertilizer and irrigation schedules  
- Disease diagnosis  
- Climate-smart agriculture  

---

## 🗂️ Repository Structure

```bash
.
├── data/                     # Directory for PDF documents (PoP files)
├── chroma_db/                # ChromaDB persistence directory
├── get_embedding_function.py # Embedding setup using Bedrock & Amazon Titan
├── populate_database.py      # Preprocesses and adds documents to vector DB
├── query_data.py             # Main RAG pipeline and evaluation metrics
├── requirements.txt          # Python dependencies
└── README.md                 # You're here!
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ShreeRam2003/agri-rag-chatbot.git
cd agri-rag-chatbot
```

### 2. Set up virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure you have:
- AWS credentials configured for Bedrock
- Ollama installed and running ([https://ollama.com](https://ollama.com))

### 4. Download and prepare data

Place PoP PDFs inside the `data/` directory.

To ingest data into the vector database:

```bash
python populate_database.py --reset
```

---

## 🚀 Run a Query

To get an answer from the system:

```bash
python query_data.py "What are the main planting seasons for maize?"
```

You will receive:
- ✅ A generated response
- 📚 Top-5 retrieved chunks
- 📈 Evaluation metrics against a ground truth dataset

---

## 📊 Evaluation Metrics

The following are computed for each query (if ground truth is available):

### 🔍 Retrieval Evaluation:
- **Precision@K**  
- **Recall@K**  
- **Mean Reciprocal Rank (MRR)**  
- **Normalized Discounted Cumulative Gain (NDCG)**

### 🧠 Generation Evaluation:
- **BLEU Score**  
- **ROUGE-1** and **ROUGE-L**  
- **BERTScore (F1)**  

Results are logged to `metrics_log.csv`.

---

## 🧠 Models Used

- **Embedding Model**: `amazon.titan-embed-text-v2:0` via AWS Bedrock  
- **LLMs Supported** via [Ollama](https://ollama.com):
  - Mistral
  - LLaMA2
  - Nous-Hermes
  - Phi3

You can configure the model in `query_data.py`:

```python
model = OllamaLLM(model="mistral")  # or "llama2", "phi3"
```

---

## 📚 Sample Query Output

**Query**:  
```
how many irrigations does cotton need?
```

**Generated Answer**:  
```
For irrigated cotton crops, irrigate the plants once every two weeks, with copious irrigation during flowering being essential to ensure good pod setting and high fiber quality.
```

**Evaluation Metrics**:
```
Precision@K: 1.00
Recall@K: 5
MRR: 1.00
NDCG: 1.00
BLEU: 0.87
ROUGE-1: 0.88
BERTScore F1: 0.91
```

---

## 🧪 Custom Ground Truth

To add or update ground truth for new queries, modify the `GROUND_TRUTH` dictionary in `query_data.py`.
