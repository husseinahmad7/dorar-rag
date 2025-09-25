# Dorar Hadith RAG System

A comprehensive Islamic Hadith search and retrieval system built with Django, featuring advanced AI capabilities including semantic search, RAG (Retrieval-Augmented Generation), and intelligent agentic search.

## 🌟 Features

### Core Functionality
- **Multiple Search Types**: Database search, semantic search, RAG, and agentic RAG
- **Arabic Text Processing**: Automatic removal of diacritics (tashkeel) for better search
- **Book Filtering**: Search within specific Hadith collections
- **Pagination**: Efficient handling of large result sets
- **REST API**: Complete API endpoints for all search functionality

### AI-Powered Search
- **Semantic Search**: Vector-based similarity search using embeddings
- **RAG Search**: Context-aware answer generation with source citations
- **Agentic RAG**: Intelligent multi-agent search with reasoning capabilities
- **Internet Integration**: Optional web search for enhanced context

### Vector Database
- **ChromaDB Integration**: Persistent vector storage
- **Dual Embedding Support**: Google Gemini API or local Gemma models
- **Automatic Embeddings**: Background processing for new hadiths

### Web Interface
- **Modern UI**: Clean, responsive design
- **Real-time Streaming**: Live progress updates for agentic search
- **Search History**: Track and revisit previous searches
- **Source Citations**: Detailed hadith references and narration chains

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- pipenv (recommended) or pip
- SQLite (default) or PostgreSQL

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/husseinahmad7/dorar-rag.git
   cd dorar-rag
   ```

2. **Install dependencies**
   ```bash
   pipenv install
   # or
   pip install -r requirements.txt
   ```

3. **Environment setup**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Database setup**
   ```bash
   pipenv run python manage.py migrate
   ```

5. **Load hadith data**
   ```bash
   pipenv run python manage.py load_hadiths data/hadiths.json
   ```

6. **Generate embeddings** (optional, for semantic search)
   ```bash
   pipenv run python manage.py update_embeddings
   ```

7. **Start the server**
   ```bash
   pipenv run python manage.py runserver
   ```

Visit `http://localhost:8000` to access the web interface.

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
# Django Settings
DEBUG=True
SECRET_KEY=your-secret-key-here

# AI Service API Keys
GEMINI_API_KEY=your-gemini-api-key
TAVILY_API_KEY=your-tavily-api-key

# Embedding Configuration
DEFAULT_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_OUTPUT_DIMENSIONALITY=3072

# LLM Configuration
DEFAULT_LLM_MODEL=gemini-2.5-flash-lite

# Vector Database
CHROMA_DB_PATH=chroma_db
VECTOR_DB_ENABLED=True

# Agentic RAG Settings
AGENTIC_RAG_ENABLED=True
MAX_SUBAGENTS=3
MEMORY_BUFFER_SIZE=10
AGENT_MAX_RUNTIME_SECONDS=60

# Local Embeddings (optional)
USE_LOCAL_EMBEDDINGS=False
LOCAL_EMBEDDING_URL=http://localhost:12434/engines/llama.cpp/v1/embeddings
LOCAL_EMBEDDING_MODEL=ai/embeddinggemma
LOCAL_EMBEDDING_DIMENSIONALITY=768
```

### Local Embeddings Setup

For privacy and cost savings, you can use local Gemma models instead of Google Gemini:

1. **Install llama.cpp** with embedding support
2. **Download a Gemma embedding model**
3. **Start the server**:
   ```bash
   ./server -m path/to/gemma-model.gguf --port 12434 --embedding
   ```
4. **Enable in environment**:
   ```bash
   USE_LOCAL_EMBEDDINGS=True
   ```

## 📖 Usage

### Web Interface

Navigate to `http://localhost:8000` and use the search interface with multiple search types:

- **Database Search**: Traditional text search with filters
- **Semantic Search**: AI-powered similarity search
- **RAG Search**: Generate answers with source citations
- **Agentic Search**: Intelligent multi-step reasoning

### API Endpoints

#### Search Hadiths
```bash
GET /api/hadiths/?q=query&search_mode=contains&book=Sahih%20Bukhari
```

#### Semantic Search
```bash
GET /api/semantic-search/?q=query&n=10
```

#### RAG Search
```bash
GET /api/rag-search/?q=query&n=5
```

#### Agentic Search
```bash
GET /api/agentic-search/?q=query&use_internet=true&max_subagents=3
```

#### Streaming Agentic Search
```bash
GET /api/agentic-search/stream/?q=query
```

## 🛠️ Management Commands

### Data Management
```bash
# Load hadiths from JSON
python manage.py load_hadiths data/hadiths.json

# Update text without diacritics
python manage.py update_tashkeel

# Populate data with embeddings
python manage.py populate_data
```

### Embedding Management
```bash
# Update embeddings for all hadiths
python manage.py update_embeddings

# Update with specific model
python manage.py update_embeddings_with_model gemini-embedding-001

# Rebuild search embeddings
python manage.py rebuild_search_embeddings
```

### Vector Database
```bash
# Reset ChromaDB
python manage.py reset_chroma

# Create fresh ChromaDB instance
python manage.py create_fresh_chroma
```

### Testing
```bash
# Test semantic search
python manage.py test_semantic_search

# Test RAG functionality
python manage.py test_rag

# Test agentic RAG
python manage.py test_agentic_rag
```

## 🏗️ Architecture

### Core Components

- **Hadith Model**: Django model for storing hadith data
- **VectorStoreService**: Manages embeddings and vector operations
- **HadithService**: Business logic for search operations
- **AgenticRAG**: Multi-agent reasoning system
- **Repository Layer**: Data access abstraction

### Search Pipeline

1. **Query Processing**: Text preprocessing and filtering
2. **Retrieval**: Database or vector-based retrieval
3. **Ranking**: Similarity scoring and relevance ranking
4. **Generation**: Answer synthesis (for RAG/agentic)
5. **Formatting**: Result presentation with citations

## 🔧 Development

### Project Structure
```
dorar-beta/
├── Dorar/                 # Django project settings
├── Hadith/               # Main application
│   ├── models.py         # Database models
│   ├── views.py          # HTTP endpoints
│   ├── services/         # Business logic
│   ├── repositories/     # Data access
│   ├── management/       # Django commands
│   └── templates/        # HTML templates
├── scraper/              # Data scraping tools
├── debug/                # Testing utilities
├── data/                 # Hadith data files
└── chroma_db/            # Vector database
```

### Testing

Run the test suite:
```bash
python manage.py test
```

Run specific service tests:
```bash
python manage.py test Hadith.tests.test_agentic_rag
```

### Code Quality

Follow Django style guide and Python best practices:
- Use meaningful variable names
- Add docstrings to functions
- Handle exceptions appropriately
- Write tests for new features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Islamic scholars and institutions for hadith collections
- Google for Gemini AI models
- ChromaDB for vector database
- DeepAgents for agentic AI framework
- Tavily for web search integration

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation in `/docs`
- Review the debug logs for troubleshooting

---

**Note**: This system is designed for educational and research purposes. Always verify information with qualified Islamic scholars for religious guidance.
