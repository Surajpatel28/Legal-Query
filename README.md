# LegalQuery

An AI-powered legal information assistant that provides intelligent search and question-answering capabilities for the Bharatiya Nyaya Sanhita (BNS), India's new criminal code.

## Features

- Semantic search across all BNS sections
- Conversational interface with chat history
- Smart section filtering and reranking
- Real-time streaming responses
- Simple, easy-to-understand legal explanations
- Context-aware follow-up questions

## Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq (Llama 3.1 8B)
- **Embeddings:** HuggingFace (BAAI/bge-base-en-v1.5)
- **Vector Database:** Qdrant Cloud
- **Reranker:** CrossEncoder (BAAI/bge-reranker-base)
- **Framework:** LangChain

## Prerequisites

- Python 3.8 or higher
- Qdrant Cloud account (free tier available)
- Groq API key (free tier available)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LegalQuery.git
cd LegalQuery
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` file with your credentials:
```env
QDRANT_CLOUD_URL=your_qdrant_cloud_url
QDRANT_CLOUD_API_KEY=your_qdrant_api_key
COLLECTION_NAME=bns_sections
GROQ_MODEL_NAME=llama-3.1-70b-versatile
GROQ_API_KEY=your_groq_api_key
```

## Getting API Keys

### Qdrant Cloud
1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a cluster (free tier available)
3. Get your cluster URL and API key

### Groq
1. Sign up at [Groq](https://console.groq.com/)
2. Generate an API key
3. Free tier includes generous limits

## Usage

### Running the Application

```bash
streamlit run app.py
```

Or use the batch file (Windows):
```bash
run.bat
```

The app will open in your browser at `http://localhost:8501`

### Data Preparation (Optional)

If you want to scrape and process BNS data from scratch:

1. Open `notebooks/data_preparation.ipynb`
2. Run all cells to scrape, clean, and upload data to Qdrant
3. This is only needed once or when updating data

## Project Structure

```
LegalQuery/
├── src/
│   ├── __init__.py          # System initialization
│   ├── config.py            # Environment configuration
│   ├── database.py          # Qdrant and embeddings setup
│   ├── retrieval.py         # Retrieval and reranking logic
│   └── chain.py             # RAG chain configuration
├── notebooks/
│   ├── data_preparation.ipynb  # Data scraping and processing
│   └── rag.ipynb               # RAG experimentation
├── data/
│   ├── raw/                    # Scraped data
│   └── processed/              # Cleaned and chunked data
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## How It Works

1. **User Query:** User asks a question about BNS
2. **Section Detection:** System checks if a specific section is mentioned
3. **Retrieval:** Searches vector database for relevant sections
4. **Reranking:** CrossEncoder reranks results for best match
5. **Context Building:** Formats retrieved sections with chat history
6. **LLM Generation:** Groq LLM generates a clear, simple answer
7. **Streaming:** Response appears word-by-word to the user

## Example Queries

- "What is the punishment for theft?"
- "Explain Section 103"
- "What happens if someone attempts to murder?"
- "Laws about harassment"

## Disclaimer

This application provides legal information for **educational purposes only**. It is not a substitute for professional legal advice. Always consult a qualified lawyer for specific legal matters.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BNS data sourced from [devgan.in](https://devgan.in/bns/)
- Built with LangChain, Streamlit, and Groq
- Embeddings by HuggingFace

## Contact

For issues, questions, or suggestions, please open an issue on GitHub.
