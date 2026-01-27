# RAG Documentation Assistant

A Retrieval-Augmented Generation (RAG) system for querying PDF documents using Google's Gemini AI models. This project implements a complete RAG pipeline that enables intelligent question-answering over any PDF document.

This system was tested using the "A Short Guide to the EU" document (`a_short_guide_to_eu.pdf`), demonstrating how RAG can provide accurate, document-grounded answers.

> **Note:** This project was developed entirely using free quotas of Google Gemini APIs, with no monetary cost. All development and testing were done within the free tier limits provided by Google.

## Table of Contents

- [What is RAG?](#what-is-rag)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [RAG vs Standard LLM Comparison](#rag-vs-standard-llm-comparison)
- [Future Updates](#future-updates)

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is an advanced AI technique that combines the power of information retrieval with large language models to provide accurate, context-aware answers.

### How RAG Works:

1. **Document Processing**: PDF documents are loaded and split into smaller, manageable chunks
2. **Embedding Generation**: Each chunk is converted into a vector embedding using an embedding model
3. **Vector Storage**: Embeddings are stored in a vector database (ChromaDB) for fast similarity search
4. **Query Processing**: When you ask a question:
   - Your question is converted to an embedding
   - The system searches for the most relevant document chunks using vector similarity
   - The retrieved chunks are used as context
   - The LLM generates an answer based on this context

### Why RAG is Better:

- **Accuracy**: Answers are grounded in actual document content, reducing hallucinations
- **Up-to-date**: Can work with documents that weren't in the LLM's training data
- **Transparency**: Answers are based on retrievable source material
- **Efficiency**: Only relevant context is sent to the LLM, reducing token usage

---

## Features

### Core Functionality

- **PDF Document Processing**: Automatically loads and processes PDF files using PyPDFLoader
- **Intelligent Text Chunking**: Uses RecursiveCharacterTextSplitter to split documents while preserving context
- **Vector Embeddings**: Leverages Google's Gemini embedding model (`gemini-embedding-001`) for high-quality vector representations
- **Persistent Vector Database**: ChromaDB stores embeddings on disk, allowing reuse across sessions without re-embedding
- **Semantic Search**: Finds the most relevant document chunks using vector similarity search
- **LLM Integration**: Uses Google's Gemini 2.5 Flash model for fast, accurate answer generation

### Advanced Features

- **Automatic Database Naming**: Database name is automatically derived from the PDF filename (e.g., `hello.pdf` → `hello_db`)
- **Configurable Chunking**: Adjustable chunk size (100-10,000 characters) and overlap for optimal performance
- **Caching System**: Implements comprehensive caching for documents, chunks, embeddings, vectorstore, retriever, and LLM chain to improve performance
- **Type Safety**: Built with Pydantic for automatic validation and type checking
- **Customizable Prompts**: Flexible prompt templates with `{context}` and `{question}` placeholders
- **Search Configuration**: Configurable search parameters (top-k retrieval, search type)
- **Temperature Control**: Adjustable temperature parameter for controlling response randomness
- **Comprehensive Logging**: Detailed logging system with configurable log levels

### Technical Highlights

- **Pydantic Model Validation**: Automatic parameter validation with constraints
- **Lazy Loading**: Components are initialized only when needed
- **Error Handling**: Robust error handling with informative error messages
- **LangChain Integration**: Built on LangChain for modularity and extensibility

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Google API Key for Gemini models

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd rag-eu-documentation-assistant
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

**How to get a Google API Key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

### Step 4: Prepare Your PDF Document

Place your PDF file in the project directory. For example:
- `a_short_guide_to_eu.pdf`
- `document.pdf`
- Any other PDF file you want to query

### Step 5: Run the Test Script

```bash
python test.py
```

The first run will:
- Load and process your PDF
- Split it into chunks
- Generate embeddings
- Create a ChromaDB database
- Answer your questions

Subsequent runs will reuse the existing database, making queries much faster!

---

## Usage

### Basic Usage

```python
from rag import RAGSystem

# Initialize the RAG system
rag = RAGSystem(
    file_path="a_short_guide_to_eu.pdf",
    prompt_template="""You are a virtual assistant for the European Union.
    Your goal is to assist users who wants to know about the European Union.
    You must use the relevant documentation given to you to answer user queries.
    You can only answer questions about the European Union. 

    Context: {context}
    Question: {question}"""
)

# Ask questions
answer = rag.ask("What is the EU doing to improve things where I live?")
```

### Advanced Configuration

```python
rag = RAGSystem(
    file_path="document.pdf",
    prompt_template="Context: {context}\nQuestion: {question}",
    chunk_size=2000,                    # Larger chunks for more context
    chunk_overlap=200,                 # More overlap between chunks
    db_name="custom_db",               # Custom database name (optional)
    model_embedding="gemini-embedding-001",
    model_llm="gemini-2.5-flash",
    search_kwargs={"k": 5},            # Retrieve top 5 chunks
    temperature=0.1                    # More deterministic responses
)

answer = rag.ask("Your question here")
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|----------|-------------|
| `file_path` | str/Path | Required | Path to the PDF file |
| `prompt_template` | str | Required | Template with `{context}` and `{question}` |
| `db_name` | str | Auto-generated | Database directory name (auto: `filename_db`) |
| `chunk_size` | int | 1000 | Chunk size in characters (100-10000) |
| `chunk_overlap` | int | 100 | Overlap between chunks (≥0) |
| `model_embedding` | str | "gemini-embedding-001" | Embedding model name |
| `model_llm` | str | "gemini-2.5-flash" | LLM model name |
| `search_type` | str | "similarity" | Search type ("similarity", "mmr", etc.) |
| `search_kwargs` | dict | `{"k": 3}` | Search parameters |
| `temperature` | float | 0.0 | LLM temperature (0.0-1.0) |

---

## RAG vs Standard LLM Comparison

This section provides a side-by-side comparison of responses from a standard LLM and the RAG system to the same question. This comparison was conducted using the "A Short Guide to the EU" document (`a_short_guide_to_eu.pdf`) as the source material. This is a visual comparison only - no quantitative measurements or metrics are used.

### Test Question

**Question:** "What is the EU doing to improve things where I live?"

### Output Comparison

**Standard LLM Response:**

The standard LLM response below is generated directly by Google Gemini 2.5 Flash without any document context. This output is based solely on the LLM's training data and may contain correct information, but it could also include hallucinations or generic responses that may not be accurate or specific to the actual document content.

```
To give you a precise answer, I'd need to know **where you live** (e.g., your country, region, or even city).

However, the European Union impacts the daily lives of its citizens in countless ways, often through national and regional governments, but also through direct funding and overarching policies.

Here are the general ways the EU improves things at a local level, which likely apply where you live:

1.  **Regional Development and Cohesion Funds:**
    *   **Infrastructure:** The EU invests heavily in improving transport links (roads, railways, ports, airports), energy networks, and broadband internet access. Look for signs on new bridges, roads, or public buildings that mention EU funding (e.g., "Co-funded by the European Union").
    *   **Job Creation & Training:** Funds like the European Social Fund Plus (ESF+) support training programs, help people find jobs, promote social inclusion, and assist small and medium-sized enterprises (SMEs) in growing and creating employment.
    *   **Urban Regeneration:** Many cities receive EU funding for revitalizing deprived areas, improving public spaces, and developing sustainable urban transport.
    *   **Environmental Projects:** Investments in renewable energy, waste management, water treatment, and protecting biodiversity.

2.  **Environmental Protection:**
    *   **Cleaner Air and Water:** EU directives set standards for air quality, drinking water, and bathing water, leading to cleaner environments.
    *   **Waste Management:** EU policies promote recycling and reduce landfill waste.
    *   **Climate Action:** The EU sets ambitious climate targets, driving investments in green technologies and sustainable practices locally.

3.  **Consumer Rights and Safety:**
    *   **Product Safety:** EU standards ensure that products sold in your area (from toys to electronics) meet high safety requirements.
    *   **Food Safety:** Strict regulations on food production, labeling, and hygiene protect your health.
    *   **Data Protection (GDPR):** Your personal data is protected by some of the strongest privacy laws in the world.
    *   **Traveler Rights:** If you fly or take a train, EU rules protect your rights in case of delays or cancellations.        

4.  **Research and Innovation:**
    *   **Horizon Europe:** This program funds research projects at universities, research institutions, and companies in your area, leading to scientific breakthroughs, new technologies, and local job creation.

5.  **Education and Youth:**
    *   **Erasmus+:** This program allows students, teachers, and young people from your area to study, train, or volunteer abroad, bringing back new skills and perspectives.
    *   **Youth Employment Initiatives:** Specific programs aim to reduce youth unemployment in regions where it's a particular problem.

6.  **Agriculture and Rural Development:**
    *   **Common Agricultural Policy (CAP):** Supports farmers in your region, ensuring a stable food supply, promoting sustainable farming practices, and supporting rural economies.

7.  **Digital Transformation:**
    *   **Broadband Access:** Initiatives to expand high-speed internet access, especially in rural or underserved areas.        
    *   **Digital Skills:** Programs to improve digital literacy among the population.

**How to find out specifically for your area:**

*   **Look for signs:** Many public works projects funded by the EU will have a sign indicating the funding source.
*   **Check your national/regional government websites:** They often have sections detailing EU-funded projects in your area.    
*   **Visit the website of the EU Representation in your country:** They usually provide information on EU activities and projects.
*   **Search online:** Try searching for "EU funded projects [your city/region]" or "European Regional Development Fund [your city/region]".

**If you tell me your country or region, I can try to provide more specific examples relevant to your area!**
```

**RAG System Response:**

The RAG system response below is generated by the same Google Gemini 2.5 Flash model, but with retrieved context from the actual document (`a_short_guide_to_eu.pdf`). This response is grounded in the document content and can be verified by checking page 20, 21 and 22 of the PDF file. The information provided is directly based on the document's content.

```
The EU invests locally in towns and regions to boost jobs and the economy and improve quality of life. It aims to make a difference in five key areas:
*   Helping people to get into employment and education, or helping to include them in society.
*   Supporting small and medium-sized businesses.
*   Fostering research and innovation.
*   Protecting and improving the environment.
*   Modernising transport and energy to fight climate change.

Between 2014 and 2020, the EU invested over €460 billion in its regions. Examples of projects include building a new emergency hospital in Romania, improving road connections in Greece, providing faster internet connections in Sicily, and creating green playgrounds in Paris.
```

**Note:** You can verify the RAG response by checking page 20, 21 and 22 of `a_short_guide_to_eu.pdf`, where this information is located.

---

## Future Updates

### Planned Enhancements

> **Note:** These are potential enhancements that may be worked on as time permits. Contributions from the community are welcome! If you're interested in implementing any of these features, please feel free to submit a Pull Request.

1. **Multi-Document Support**
   - Ability to process and query multiple PDF documents simultaneously
   - Cross-document reference and comparison capabilities

2. **Web Interface**
   - Interactive web UI using Streamlit or Gradio
   - Real-time question-answering interface
   - Document upload and management

3. **Advanced Retrieval Methods**
   - Hybrid search (combining keyword and semantic search)
   - Re-ranking of retrieved chunks
   - Query expansion and reformulation

4. **Citation and Source Tracking**
   - Automatic citation of source chunks
   - Page number references
   - Confidence scores for answers

5. **Multi-Model Support**
   - Support for OpenAI, Anthropic, and other LLM providers
   - Embedding model selection (OpenAI, Cohere, etc.)
   - Model comparison tools

6. **Performance Optimizations**
   - Batch processing for multiple questions
   - Async/await support for concurrent queries
   - Streaming responses for real-time answers

7. **Enhanced Chunking Strategies**
   - Semantic chunking (chunk by topic/meaning)
   - Document structure-aware chunking (respect headers, sections)
   - Custom chunking strategies

8. **Vector Database Options**
   - Support for alternative vector databases (Pinecone, Weaviate, Qdrant, FAISS, etc.)
   - Easy switching between different vector database backends
   - Database-specific optimizations

9. **Multiple Document Types**
   - Support for additional document formats (.txt, .html, .docx, .md, etc.)
   - Format-specific loaders and processors
   - Unified interface for different document types

10. **API Endpoint**
    - RESTful API for integration with other applications
    - Webhook support
    - Rate limiting and authentication

---

## Project Structure

```
rag-eu-documentation-assistant/
│
├── rag.py                 # Main RAG system implementation
├── test.py                # Example usage script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore rules
│
└── [pdf_file]_db/        # ChromaDB database (auto-generated)
    └── ...               # Vector database files
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Powered by [Google Gemini](https://deepmind.google/technologies/gemini/) models
- Developed using free quotas of Google Gemini APIs - no monetary cost incurred during development

---

## Contact

Feel free to reach out to me on LinkedIn for any questions, suggestions, or collaborations:

[Saltuk Bugra Karacan](https://www.linkedin.com/in/sbkaracan)

For more information or support, please open an issue on the repository or contact me via LinkedIn.
