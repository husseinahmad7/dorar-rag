# Local Embedding Integration with Gemma

This document describes the local embedding functionality that allows you to use a local Gemma model instead of Google's Gemini API for generating embeddings.

## Overview

The system now supports both Google Gemini embeddings and local Gemma embeddings through a configurable interface. You can switch between them using environment variables without changing any code.

## Features

- ✅ **LocalEmbeddingFunction**: Compatible with ChromaDB's embedding function interface
- ✅ **Seamless Integration**: Works with existing VectorStoreService without code changes
- ✅ **System-level Configuration**: Enable/disable via environment variables
- ✅ **Automatic Fallback**: Graceful handling when local server is unavailable
- ✅ **Dimension Detection**: Automatically detects and adjusts to local model dimensions
- ✅ **Performance Monitoring**: Comprehensive logging and error handling

## Configuration

### Environment Variables

Set these environment variables to enable local embeddings:

```bash
# Enable local embeddings
USE_LOCAL_EMBEDDINGS=True

# Local embedding server configuration
LOCAL_EMBEDDING_URL=http://localhost:12434/engines/llama.cpp/v1/embeddings
LOCAL_EMBEDDING_MODEL=ai/embeddinggemma
LOCAL_EMBEDDING_DIMENSIONALITY=768
```

### Settings Integration

The configuration is automatically loaded from Django settings:

```python
# In Dorar/settings.py
USE_LOCAL_EMBEDDINGS = os.getenv('USE_LOCAL_EMBEDDINGS', 'False').lower() in ('1', 'true', 'yes')
LOCAL_EMBEDDING_URL = os.getenv('LOCAL_EMBEDDING_URL', 'http://localhost:12434/engines/llama.cpp/v1/embeddings')
LOCAL_EMBEDDING_MODEL = os.getenv('LOCAL_EMBEDDING_MODEL', 'ai/embeddinggemma')
LOCAL_EMBEDDING_DIMENSIONALITY = int(os.getenv('LOCAL_EMBEDDING_DIMENSIONALITY', '768'))
```

## Usage

### Basic Usage

```python
from Hadith.services.vector_store import VectorStoreService

# Initialize vector store (automatically uses local embeddings if configured)
vector_store = VectorStoreService()

# Generate embeddings (works the same regardless of backend)
embedding = vector_store.generate_embedding("النص العربي للاختبار")
```

### Direct LocalEmbeddingFunction Usage

```python
from Hadith.services.vector_store import LocalEmbeddingFunction

# Initialize local embedding function
local_embedding = LocalEmbeddingFunction(
    base_url="http://localhost:12434",
    model_name="ai/embeddinggemma"
)

# Generate single embedding
embedding = local_embedding.embed_query("نص للاختبار")

# Generate batch embeddings
embeddings = local_embedding.embed_documents([
    "النص الأول",
    "النص الثاني",
    "النص الثالث"
])
```

## Local Server Requirements

### llama.cpp Server Setup

1. **Install llama.cpp** with embedding support
2. **Download Gemma embedding model** (e.g., `ai/embeddinggemma`)
3. **Start the server**:
   ```bash
   ./server -m path/to/gemma-model.gguf --port 12434 --embedding
   ```

### API Endpoint

The local server should provide an OpenAI-compatible embedding endpoint:

```
POST http://localhost:12434/engines/llama.cpp/v1/embeddings
Content-Type: application/json

{
    "model": "ai/embeddinggemma",
    "input": ["text to embed"]
}
```

Expected response:
```json
{
    "data": [
        {
            "embedding": [0.1, 0.2, 0.3, ...],
            "index": 0
        }
    ]
}
```

## Testing

### Test Scripts

1. **Basic functionality test**:
   ```bash
   python test_local_mod.py
   ```

2. **Integration test**:
   ```bash
   python demo_local_embeddings.py
   ```

3. **Comprehensive test**:
   ```bash
   python test_local_embeddings.py
   ```

### Manual Testing

```python
# Test local server connectivity
import requests

response = requests.post(
    "http://localhost:12434/engines/llama.cpp/v1/embeddings",
    json={"model": "ai/embeddinggemma", "input": ["test"]},
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    embedding = response.json()["data"][0]["embedding"]
    print(f"Embedding dimension: {len(embedding)}")
```

## Benefits

### Cost Savings
- No API costs for embedding generation
- Unlimited usage without rate limits

### Privacy & Security
- All data stays local
- No external API calls
- Full control over the embedding process

### Performance
- No network latency
- Consistent response times
- No dependency on external services

### Customization
- Use custom-trained models
- Fine-tune for specific domains
- Control model parameters

## Troubleshooting

### Common Issues

1. **Connection refused**:
   - Check if local server is running
   - Verify port and URL configuration
   - Check firewall settings

2. **Dimension mismatch**:
   - The system automatically detects dimensions
   - Update LOCAL_EMBEDDING_DIMENSIONALITY if needed

3. **Model not found**:
   - Verify model name in LOCAL_EMBEDDING_MODEL
   - Check if model is loaded in llama.cpp server

4. **Timeout errors**:
   - Increase timeout in LocalEmbeddingFunction
   - Check server performance and resources

### Debug Mode

Enable debug logging to see detailed information:

```python
import logging
logging.getLogger('Hadith.services.vector_store').setLevel(logging.DEBUG)
```

## Migration

### From Gemini to Local

1. Set `USE_LOCAL_EMBEDDINGS=True`
2. Start local embedding server
3. Restart Django application
4. Existing embeddings remain compatible

### From Local to Gemini

1. Set `USE_LOCAL_EMBEDDINGS=False`
2. Ensure `GEMINI_API_KEY` is set
3. Restart Django application

## Performance Comparison

| Aspect | Local Gemma | Gemini API |
|--------|-------------|------------|
| Cost | Free | Pay per token |
| Latency | ~50-200ms | ~200-500ms |
| Privacy | Full | Limited |
| Scalability | Hardware limited | API limited |
| Customization | Full control | Limited |

## Future Enhancements

- Support for multiple local models
- Model switching without restart
- Embedding caching
- Performance metrics
- Health monitoring dashboard
