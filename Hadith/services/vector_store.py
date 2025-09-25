"""
Vector store service for embeddings and semantic search using Django conventions.

This module provides vector storage and semantic search capabilities using
ChromaDB and Google's gemini-embedding-001 model or local Gemma embeddings,
following Django best practices.
"""
import re, unicodedata
import logging
import requests
from typing import List, Dict, Any, Optional, Union

from django.conf import settings
from google import genai
from google.genai.types import EmbedContentConfig
import chromadb
import chromadb.utils.embedding_functions as embedding_functions

# Optional imports for LangChain integration
try:
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_chroma import Chroma
    langchain_available = True
except ImportError as e:
    langchain_available = False
    logging.warning(f"LangChain not fully available: {e}. RAG with LangChain will not be available.")

# Configure logging
logger = logging.getLogger(__name__)


class LocalEmbeddingFunction:
    """
    Local embedding function that uses a local Gemma model via llama.cpp server.

    This class provides an interface compatible with ChromaDB's embedding functions
    while using a local embedding model instead of external APIs.
    """

    def __init__(self, base_url: str, model_name: str):
        """
        Initialize the local embedding function.

        Args:
            base_url (str): Base URL of the local embedding server
            model_name (str): Name of the embedding model to use
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        logger.info(f"Initialized LocalEmbeddingFunction with URL: {self.base_url}, Model: {self.model_name}")

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of texts to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        try:
            headers = {'Content-Type': 'application/json'}
            data = {
                'model': self.model_name,
                'input': texts
            }

            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]

            logger.debug(f"Generated {len(embeddings)} embeddings using local model")
            return embeddings

        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting local embeddings: {e}")
            raise Exception(f"Local embedding service unavailable: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing local embedding response: {e}")
            raise Exception(f"Invalid response from local embedding service: {e}")

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        """
        Embed documents for storage in vector database.

        Args:
            docs (List[str]): List of documents to embed

        Returns:
            List[List[float]]: List of embedding vectors
        """
        return self(docs)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query for search.

        Args:
            query (str): Query text to embed

        Returns:
            List[float]: Embedding vector
        """
        # Add prompt prefix for search queries
        prefixed_query = f"task: search result | query: {query}"
        return self([prefixed_query])[0]
        #return self([query])[0]



class VectorStoreService:
    """
    Service for vector embeddings and semantic search using Django conventions.
    
    This service provides vector storage and semantic search capabilities using
    ChromaDB and Google's gemini-embedding-001 model. It follows Django best
    practices for configuration, logging, and error handling.
    
    Attributes:
        COLLECTION_NAME (str): Name of the ChromaDB collection
        EMBEDDING_TASK_DOCUMENT (str): Task type for document embeddings
        EMBEDDING_TASK_QUERY (str): Task type for query embeddings
    """
    
    # Class-level constants
    COLLECTION_NAME = "hadith_collection"
    EMBEDDING_TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
    EMBEDDING_TASK_QUERY = "RETRIEVAL_QUERY"
    
    @staticmethod
    def remove_tashkeel(text: str) -> str:
        """
        Remove Arabic diacritics (tashkeel) from text.
        
        Args:
            text (str): The Arabic text to process
            
        Returns:
            str: Text with tashkeel removed
        """
        # Arabic diacritics Unicode ranges
        tashkeel = re.compile(r'[\u064b-\u0652\u0670]')
        return re.sub(tashkeel, '', text)
        # text = unicodedata.normalize("NFKC", text)
        # text = re.sub(r'[\u064b-\u065f\u0670]', '', text)          # tashkīl
        # text = re.sub(r'[\u0640]', '', text)                       # kashida
        # text = re.sub(r'[آأإ]', 'ا', text)                         # hamza on alif
        # text = re.sub(r'[ة]', 'ه', text)                           # tāʾ marbūṭa → hāʾ
        # text = re.sub(r'[ى]', 'ي', text)                           # alif maksūra → yāʾ
        # text = re.sub(r'[^\u0600-\u06FF\s\d]', ' ', text)          # keep only Arabic
        # text = re.sub(r'\s+', ' ', text).strip()
        # return text


    def __init__(self):
        """
        Initialize the vector store service.
        
        Sets up the Gemini client, ChromaDB client, and LangChain integration
        based on Django settings configuration.
        """
        # Configuration from Django settings
        self.gemini_api_key = settings.GEMINI_API_KEY
        self.chroma_path = settings.CHROMA_DB_PATH
        self.embedding_model = settings.DEFAULT_EMBEDDING_MODEL

        self.llm_model = settings.DEFAULT_LLM_MODEL
        self.embedding_dimensionality = settings.EMBEDDING_OUTPUT_DIMENSIONALITY

        # Local embedding configuration
        self.use_local_embeddings = getattr(settings, 'USE_LOCAL_EMBEDDINGS', False)
        self.local_embedding_url = getattr(settings, 'LOCAL_EMBEDDING_URL', 'http://localhost:12434/engines/llama.cpp/v1/embeddings')
        self.local_embedding_model = getattr(settings, 'LOCAL_EMBEDDING_MODEL', 'ai/embeddinggemma')
        self.local_embedding_dimensionality = getattr(settings, 'LOCAL_EMBEDDING_DIMENSIONALITY', 768)

        # Client instances
        self.genai_client = None
        self.chroma_client = None
        self.local_embedding_function = None
        
        # LangChain integration
        self.langchain_llm = None
        self.langchain_retriever = None
        self.use_langchain = langchain_available
        
        # Availability flag
        self.is_available = False

        # Check if vector database is globally enabled
        if not getattr(settings, 'VECTOR_DB_ENABLED', True):
            logger.info("Vector database is globally disabled via VECTOR_DB_ENABLED setting")
            return

        # Initialize based on embedding preference
        if self.use_local_embeddings:
            try:
                self._initialize_local_embeddings()
                self._initialize_clients()
                self._initialize_langchain()
                self.is_available = True
                logger.info("Vector store service initialized successfully with local embeddings")
            except Exception as e:
                logger.error(f"Error initializing vector store service with local embeddings: {e}")
                self.is_available = False
        elif self.gemini_api_key:
            try:
                self._initialize_clients()
                self._initialize_langchain()
                self.is_available = True
                logger.info("Vector store service initialized successfully with Gemini embeddings")
            except Exception as e:
                logger.error(f"Error initializing vector store service: {e}")
                self.is_available = False
        else:
            logger.warning("Neither local embeddings enabled nor GEMINI_API_KEY found. Vector store service not available.")

    def _initialize_local_embeddings(self) -> None:
        """
        Initialize the local embedding function.

        Raises:
            Exception: If local embedding service is not available
        """
        try:
            # Test connection to local embedding service
            test_response = requests.get(
                self.local_embedding_url.replace('/embeddings', '/health'),
                timeout=5
            )
            logger.info("Local embedding service health check passed")
        except requests.exceptions.RequestException:
            # If health check fails, try a simple embedding test
            try:
                self.local_embedding_function = LocalEmbeddingFunction(
                    self.local_embedding_url.rsplit('/', 2)[0],  # Remove /engines/llama.cpp/v1/embeddings
                    self.local_embedding_model
                )
                # Test with a simple text
                test_embedding = self.local_embedding_function.embed_query("test")
                if len(test_embedding) != self.local_embedding_dimensionality:
                    logger.warning(f"Local embedding dimensionality mismatch: expected {self.local_embedding_dimensionality}, got {len(test_embedding)}")
                    # Update dimensionality to match actual output
                    self.embedding_dimensionality = len(test_embedding)
                    self.local_embedding_dimensionality = len(test_embedding)
                logger.info(f"Local embedding function initialized successfully with dimensionality {len(test_embedding)}")
            except Exception as e:
                raise Exception(f"Local embedding service test failed: {e}")

        # Initialize the local embedding function if not already done
        if not self.local_embedding_function:
            self.local_embedding_function = LocalEmbeddingFunction(
                self.local_embedding_url.rsplit('/', 2)[0],  # Remove /engines/llama.cpp/v1/embeddings
                self.local_embedding_model
            )

    def _initialize_clients(self) -> None:
        """
        Initialize the Gemini and ChromaDB clients.

        Raises:
            Exception: If client initialization fails
        """
        # Initialize the Gemini client only if not using local embeddings
        if not self.use_local_embeddings and self.gemini_api_key:
            self.genai_client = genai.Client(api_key=self.gemini_api_key)
            logger.info(f"Using Gemini embedding model: {self.embedding_model}")
        elif self.use_local_embeddings:
            logger.info(f"Using local embedding model: {self.local_embedding_model}")

        logger.info(f"Using LLM model: {self.llm_model}")

        # Initialize ChromaDB with PersistentClient to store data on disk
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        logger.info(f"ChromaDB initialized at: {self.chroma_path}")
    
    def _initialize_langchain(self) -> None:
        """
        Initialize LangChain integration if available.
        
        Sets up the LangChain LLM and retriever for advanced RAG capabilities.
        """
        if not self.use_langchain:
            return
            
        try:
            self.langchain_llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                google_api_key=self.gemini_api_key,
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=1024
            )
            
            # Initialize LangChain retriever
            self._init_langchain_retriever()
            logger.info("LangChain integration initialized successfully")
        except Exception as e:
            logger.warning(f"LangChain initialization failed: {e}")
            self.use_langchain = False

    def _init_langchain_retriever(self) -> None:
        """
        Initialize LangChain retriever with ChromaDB.
        
        Creates a custom embedding function that uses our generate_embedding method
        and sets up a LangChain retriever for advanced RAG capabilities.
        """
        if not self.use_langchain:
            return

        try:
            # Create a custom embedding function that uses our generate_embedding method
            class CustomEmbeddingFunction:
                def __init__(self, embedding_function):
                    self.embedding_function = embedding_function

                def embed_documents(self, texts):
                    return [self.embedding_function(text, use_without_tashkeel=True) for text in texts]

                def embed_query(self, text):
                    return self.embedding_function(text, use_without_tashkeel=True, is_query=True)

            # Check if collection exists
            try:
                native_collection = self.chroma_client.get_collection(self.COLLECTION_NAME)
                logger.info(f"Found existing collection with {native_collection.count()} documents")
            except Exception as e:
                logger.warning(f"Error getting collection: {e}")
                return

            # Create LangChain Chroma instance using the existing ChromaDB collection
            langchain_db = Chroma(
                collection_name=self.COLLECTION_NAME,
                persist_directory=self.chroma_path,
                embedding_function=CustomEmbeddingFunction(self.generate_embedding)
            )

            # Create retriever
            self.langchain_retriever = langchain_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            logger.info("LangChain retriever initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LangChain retriever: {e}")

    def generate_embedding(self, text: str, use_without_tashkeel: bool = True, is_query: bool = False) -> List[float]:
        """
        Generate embedding for text using either local Gemma model or Gemini API.

        Args:
            text (str): The text to embed
            use_without_tashkeel (bool): Whether to remove tashkeel before embedding
            is_query (bool): Whether this is a query (affects embedding task type)

        Returns:
            List[float]: The embedding vector

        Raises:
            ValueError: If the vector store service is not available
            Exception: If embedding generation fails
        """
        if not self.is_available:
            raise ValueError("Vector store service not available")

        # Preprocess text
        text_to_embed = self._preprocess_text(text, use_without_tashkeel)

        # Auto-detect if this is a query if not explicitly specified
        if not is_query:
            is_query = self._is_question(text_to_embed)

        # Generate embedding using local or remote service
        if self.use_local_embeddings and self.local_embedding_function:
            try:
                logger.debug(f"Generating local embedding for {'query' if is_query else 'document'}: {text_to_embed[:50]}...")
                embedding = self.local_embedding_function.embed_query(text_to_embed)
                logger.debug(f"Generated local embedding with {len(embedding)} dimensions")
                return embedding
            except Exception as e:
                logger.error(f"Error generating local embedding: {e}")
                raise Exception(f"Local embedding generation failed: {e}")
        else:
            # Generate embedding using Google Gemini API
            try:
                task_type = self.EMBEDDING_TASK_QUERY if is_query else self.EMBEDDING_TASK_DOCUMENT
                print(task_type, text_to_embed)
                logger.debug(f"Generating embedding with model: {self.embedding_model}, task_type: {task_type}")

                # Use EmbedContentConfig for proper configuration
                config = EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.embedding_dimensionality
                )

                result = self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=[text_to_embed],  # Contents should be a list
                    config=config
                )

                # Extract the embedding values
                embedding = result.embeddings[0].values
                logger.debug(f"Generated embedding with {len(embedding)} dimensions")
                return embedding

            except Exception as e:
                logger.error(f"Error generating embeddings with model {self.embedding_model}: {e}")
                return self._handle_embedding_fallback(text_to_embed, is_query)
    
    def _preprocess_text(self, text: str, use_without_tashkeel: bool) -> str:
        """
        Preprocess text for embedding generation.
        
        Args:
            text (str): The text to preprocess
            use_without_tashkeel (bool): Whether to remove tashkeel
            
        Returns:
            str: Preprocessed text
        """
        if use_without_tashkeel:
            try:
                # Try to use the static method from this class
                return self.remove_tashkeel(text)
            except Exception:
                # Fallback to Hadith model method if available
                try:
                    from Hadith.models import Hadith
                    return Hadith.remove_tashkeel(text)
                except (ImportError, AttributeError):
                    logger.warning("Could not remove tashkeel, using original text")
                    return text
        return text
    
    def _is_question(self, text: str) -> bool:
        """
        Detect if text is a question based on Arabic question words and punctuation.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            bool: True if text appears to be a question
        """
        question_indicators = ['ما', 'ماذا', 'لماذا', 'كيف', 'متى', 'أين', 'من', 'هل', 'أي']
        return '?' in text or any(q in text for q in question_indicators)
    
    def _handle_embedding_fallback(self, text: str, is_query: bool) -> List[float]:
        """
        Handle embedding generation fallback scenarios.
        
        Args:
            text (str): The text to embed
            is_query (bool): Whether this is a query
            
        Returns:
            List[float]: Fallback embedding vector
            
        Raises:
            Exception: If all fallback attempts fail
        """
        # Try with a fallback model
        fallback_model = "embedding-001"
        if self.embedding_model != fallback_model:
            logger.warning(f"Trying fallback embedding model: {fallback_model}")
            try:
                task_type = self.EMBEDDING_TASK_QUERY if is_query else self.EMBEDDING_TASK_DOCUMENT
                result = self.genai_client.models.embed_content(
                    model=fallback_model,
                    contents=[text],  # Contents should be a list
                    config=EmbedContentConfig(task_type=task_type)
                )
                return result.embeddings[0].values
            except Exception as e:
                logger.error(f"Error with fallback model: {e}")

        # Final fallback - this should only be used in development/testing
        if settings.DEBUG:
            import numpy as np
            logger.warning("Returning random embedding as fallback. This should only be used for testing!")
            return list(np.random.rand(self.embedding_dimensionality))
        else:
            raise Exception("Embedding generation failed and no fallback available in production")

    def _get_or_create_collection(self):
        """
        Get or create the ChromaDB collection.

        Returns:
            chromadb.Collection: The ChromaDB collection
            
        Raises:
            Exception: If collection creation/retrieval fails
        """
        try:
            # Try to get the existing collection
            collection = self.chroma_client.get_collection(self.COLLECTION_NAME)
            logger.info(f"Found existing collection with {collection.count()} documents")
            return collection
        except Exception as e:
            # Collection doesn't exist yet, create it
            logger.info(f"Creating new collection: {e}")

            # Determine embedding function and metadata based on configuration
            if self.use_local_embeddings and self.local_embedding_function:
                embedding_function = self.local_embedding_function
                model_description = f"local {self.local_embedding_model}"
                dimensionality = self.local_embedding_dimensionality
            else:
                embedding_function = None  # Use default Gemini embeddings
                model_description = self.embedding_model
                dimensionality = self.embedding_dimensionality

            return self.chroma_client.create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=embedding_function,
                metadata={
                    "description": f"Collection of hadith embeddings using {model_description}",
                    "model": model_description,
                    "embedding_dimensionality": dimensionality,
                    "created_by": "VectorStoreService",
                    "use_local_embeddings": self.use_local_embeddings
                }
            )

    def semantic_search(self, query: str, n_results: int = 10, is_question: Optional[bool] = None,
                       book_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search using gemini-embedding-001.

        Args:
            query (str): The search query
            n_results (int): Number of results to return
            is_question (bool, optional): Force query to be treated as a question
            book_filter (List[str], optional): List of book names to filter results by

        Returns:
            List[Dict[str, Any]]: List of search results with metadata

        Raises:
            ValueError: If the vector store service is not available
        """
        logger.info(f'Semantic search for query: "{query}" (is_question={is_question})')
        
        if not self.is_available:
            raise ValueError("Vector store service not available")

        # Generate query embedding - always treat search queries as questions for better results
        query_embedding = self.generate_embedding(query, use_without_tashkeel=True, is_query=True)
        logger.debug(f"Generated embedding with is_query=True (overriding is_question={is_question})")

        # Get or create collection
        collection = self._get_or_create_collection()

        # Check collection status
        try:
            count = collection.count()
            logger.info(f"Collection has {count} documents")
            if count == 0:
                logger.warning("Collection is empty! Run update_embeddings to populate it.")
                return []
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")

        # Execute search
        try:
            # If book filter is provided, we need to get more results and filter them
            search_limit = min(n_results * 3, 300) if book_filter else min(n_results, 100)

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=search_limit,
                include=['metadatas', 'documents', 'distances']
            )
            logger.info(f"Query returned {len(results['documents'][0])} results")
        except Exception as e:
            logger.error(f"Error during ChromaDB query: {e}")
            raise

        # Format and return results
        formatted_results = self._format_search_results(results)

        # Apply book filter if provided
        if book_filter:
            filtered_results = []
            for result in formatted_results:
                if result.get('book_name') in book_filter:
                    filtered_results.append(result)
                    if len(filtered_results) >= n_results:
                        break
            logger.info(f"Filtered results from {len(formatted_results)} to {len(filtered_results)} based on book filter")
            return filtered_results

        return formatted_results[:n_results]
    
    def _format_search_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Format ChromaDB search results into a standardized format.
        
        Args:
            results (Dict[str, Any]): Raw results from ChromaDB
            
        Returns:
            List[Dict[str, Any]]: Formatted search results
        """
        hadiths = []
        for metadata, document, distance in zip(
            results['metadatas'][0],
            results['documents'][0],
            results['distances'][0]
        ):
            # Process narrators data
            narrators = self._parse_narrators(metadata.get('narrators', []))
            
            # Create formatted source and narration chain
            formatted_source = self._create_formatted_source(metadata)
            narration_chain = self._create_narration_chain(narrators, metadata.get('exporter'))

            # Add hadith to results
            hadiths.append({
                'id': metadata.get('id'),
                'text': document,
                'book_name': metadata.get('book_name'),
                'volume': metadata.get('volume'),
                'page_number': metadata.get('page_number'),
                'chapter': metadata.get('chapter'),
                'exporter': metadata.get('exporter'),
                'narrators': narrators,
                'url': metadata.get('url'),
                'get_formatted_source': formatted_source,
                'get_narration_chain': narration_chain,
                'similarity_score': (1 - distance) * 100  # Convert distance to percentage
            })

        return hadiths
    
    def _parse_narrators(self, narrators_data: Union[str, List[str]]) -> List[str]:
        """
        Parse narrators data from metadata.
        
        Args:
            narrators_data (Union[str, List[str]]): Raw narrators data
            
        Returns:
            List[str]: Parsed list of narrators
        """
        if isinstance(narrators_data, str):
            try:
                import json
                return json.loads(narrators_data.replace("'", '"'))
            except Exception as e:
                logger.warning(f"Error parsing narrators: {e}")
                return []
        return narrators_data if isinstance(narrators_data, list) else []
    
    def _create_formatted_source(self, metadata: Dict[str, Any]) -> str:
        """
        Create a formatted source string from metadata.
        
        Args:
            metadata (Dict[str, Any]): Hadith metadata
            
        Returns:
            str: Formatted source string
        """
        book_name = metadata.get('book_name', '')
        volume = metadata.get('volume', '')
        page_number = metadata.get('page_number', '')
        chapter = metadata.get('chapter', '')
        
        formatted_source = book_name
        if volume:
            formatted_source += f", الجزء {volume}"
        if page_number:
            formatted_source += f", الصفحة {page_number}"
        if chapter:
            formatted_source += f", الباب: {chapter}"
            
        return formatted_source
    
    def _create_narration_chain(self, narrators: List[str], exporter: Optional[str]) -> str:
        """
        Create a narration chain string from narrators and exporter.
        
        Args:
            narrators (List[str]): List of narrators
            exporter (Optional[str]): The hadith exporter
            
        Returns:
            str: Formatted narration chain
        """
        narration_chain = ""
        
        if narrators:
            # Join narrators with 'عن'
            narrators_chain = " عن ".join(narrators[:-1]) if len(narrators) > 1 else ""
            
            # Add last narrator if available
            if len(narrators) > 0:
                if narrators_chain:
                    narrators_chain += " عن " + narrators[-1]
                else:
                    narrators_chain = narrators[-1]
            
            # Add exporter if available
            if exporter:
                narration_chain = f"{narrators_chain} قال {exporter}"
            else:
                narration_chain = narrators_chain
        elif exporter:
            narration_chain = f"قال {exporter}"
            
        return narration_chain

    def store_embedding(self, hadith_id: int, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Store embedding in vector store using gemini-embedding-001.

        Args:
            hadith_id (int): The hadith ID
            text (str): The text to embed
            metadata (Dict[str, Any]): Metadata for the document

        Returns:
            bool: True if successful

        Raises:
            ValueError: If the vector store service is not available
            Exception: If storage fails
        """
        if not self.is_available:
            raise ValueError("Vector store service not available")

        try:
            # Get or create collection
            collection = self._get_or_create_collection()

            # Prepare metadata - ensure all values are strings
            processed_metadata = {k: str(v) for k, v in metadata.items() if v is not None}

            # Generate embedding for document storage
            embedding = self.generate_embedding(text, use_without_tashkeel=True, is_query=False)

            # Store in ChromaDB with embeddings
            collection.upsert(
                documents=[text],
                metadatas=[processed_metadata],
                ids=[str(hadith_id)],
                embeddings=[embedding]
            )

            # Update the hadith model to mark it as embedded
            self._update_hadith_embedded_status(hadith_id)
            
            logger.info(f"Successfully stored embedding for hadith {hadith_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embedding in ChromaDB: {e}")
            raise
    
    def _update_hadith_embedded_status(self, hadith_id: int) -> None:
        """
        Update the hadith model to mark it as embedded.
        
        Args:
            hadith_id (int): The hadith ID to update
        """
        try:
            from Hadith.models import Hadith
            hadith = Hadith.objects.get(hadith_id=hadith_id)
            hadith.is_embedded = True
            hadith.save(update_fields=['is_embedded'])
            logger.debug(f"Marked hadith {hadith_id} as embedded")
        except Exception as e:
            logger.warning(f"Error updating hadith model: {e}")

    def generate_rag_response(self, query: str, context_texts: Optional[List[str]] = None, 
                            max_tokens: int = 1024, use_langchain: Optional[bool] = None) -> str:
        """
        Generate a response using RAG (Retrieval-Augmented Generation).

        Args:
            query (str): The user query
            context_texts (Optional[List[str]]): List of context texts. If None and use_langchain=True,
                                               will use LangChain retriever to get contexts
            max_tokens (int): Maximum tokens for response
            use_langchain (Optional[bool]): Whether to use LangChain for RAG

        Returns:
            str: The generated response

        Raises:
            ValueError: If the vector store service is not available
        """
        if not self.is_available:
            raise ValueError("Vector store service not available")

        # Determine whether to use LangChain
        if use_langchain is None:
            use_langchain = self.use_langchain

        # If LangChain is requested but not available, fall back to standard method
        if use_langchain and not self.use_langchain:
            logger.warning("LangChain requested but not available. Falling back to standard RAG.")
            use_langchain = False

        if use_langchain:
            # If context_texts is None, use LangChain retriever to get contexts
            if context_texts is None and self.langchain_retriever is not None:
                context_texts = self._retrieve_langchain_contexts(query)
            return self._generate_langchain_rag_response(query, context_texts, max_tokens)
        else:
            # If context_texts is None, use semantic search to get contexts
            if context_texts is None:
                search_results = self.semantic_search(query, n_results=5, is_question=True)
                context_texts = [result['text'] for result in search_results]
            return self._generate_standard_rag_response(query, context_texts, max_tokens)
    
    def _retrieve_langchain_contexts(self, query: str) -> List[str]:
        """
        Retrieve context texts using LangChain retriever.
        
        Args:
            query (str): The search query
            
        Returns:
            List[str]: List of context texts
        """
        try:
            # Use LangChain retriever to get relevant documents
            docs = self.langchain_retriever.invoke(query)
            context_texts = [doc.page_content for doc in docs]
            logger.info(f"Retrieved {len(context_texts)} documents using LangChain retriever")
            return context_texts
        except Exception as e:
            logger.error(f"Error retrieving documents with LangChain: {e}")
            # Fall back to semantic search if retriever fails
            search_results = self.semantic_search(query, n_results=5, is_question=True)
            return [result['text'] for result in search_results]

    def _generate_standard_rag_response(self, query: str, context_texts: List[str], max_tokens: int = 1024) -> str:
        """
        Generate a response using standard RAG (without LangChain).

        Args:
            query (str): The user query
            context_texts (List[str]): List of context texts
            max_tokens (int): Maximum tokens for response

        Returns:
            str: The generated response
        """
        # Prepare context
        context = "\n\n".join([f"Hadith: {text}" for text in context_texts])

        # Prepare prompt
        prompt = f"""
        You are an AI assistant specialized in Islamic hadiths.
        Answer the following question based on the provided hadiths.
        If the answer cannot be found in the provided hadiths, say so clearly.
        Do not make up information. Cite the hadiths you use in your answer.

        Context hadiths:
        {context}

        Question: {query}

        Answer:
        """

        # Generate response using the Gemini API
        try:
            response = self.genai_client.models.generate_content(
                model=self.llm_model,
                contents=[prompt],  # Contents should be a list
                config={
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            # Fallback with basic parameters
            try:
                response = self.genai_client.models.generate_content(
                    model=self.llm_model,
                    contents=[prompt]  # Contents should be a list
                )
                return response.text
            except Exception as e2:
                logger.error(f"Error with fallback generation: {e2}")
                raise

    def get_recommended_hadiths(self, hadith_id: int, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get recommended hadiths similar to the given hadith.

        Args:
            hadith_id (int): The hadith ID to find recommendations for
            n_results (int): Number of recommendations to return

        Returns:
            List[Dict[str, Any]]: List of recommended hadiths

        Raises:
            ValueError: If the vector store service is not available
        """
        if not self.is_available:
            raise ValueError("Vector store service not available")

        try:
            # Get the hadith text
            from Hadith.models import Hadith
            hadith = Hadith.objects.get(hadith_id=hadith_id)
            text = hadith.text

            # Get or create collection
            collection = self._get_or_create_collection()

            # Generate embedding for document similarity (not query)
            embedding = self.generate_embedding(text, use_without_tashkeel=True, is_query=False)

            # Execute the query
            results = collection.query(
                query_embeddings=[embedding],
                n_results=min(n_results + 1, 100),  # +1 to account for the query hadith itself
                include=['metadatas', 'documents', 'distances']
            )

            # Filter out the query hadith itself and format results
            return self._filter_and_format_recommendations(results, hadith_id, n_results)
            
        except Exception as e:
            logger.error(f"Error getting recommended hadiths: {e}")
            return []
    
    def _filter_and_format_recommendations(self, results: Dict[str, Any], 
                                         hadith_id: int, n_results: int) -> List[Dict[str, Any]]:
        """
        Filter out the query hadith and format recommendation results.
        
        Args:
            results (Dict[str, Any]): Raw results from ChromaDB
            hadith_id (int): The original hadith ID to filter out
            n_results (int): Number of results to return
            
        Returns:
            List[Dict[str, Any]]: Filtered and formatted recommendations
        """
        # Filter out the query hadith itself
        filtered_results = {
            'metadatas': [],
            'documents': [],
            'distances': []
        }

        for i, id_val in enumerate(results['ids'][0]):
            if id_val != str(hadith_id):
                filtered_results['metadatas'].append(results['metadatas'][0][i])
                filtered_results['documents'].append(results['documents'][0][i])
                filtered_results['distances'].append(results['distances'][0][i])

        # Format results using the same method as semantic search
        formatted_results = []
        for i in range(min(n_results, len(filtered_results['metadatas']))):
            metadata = filtered_results['metadatas'][i]
            document = filtered_results['documents'][i]
            distance = filtered_results['distances'][i]

            # Process narrators and create formatted strings
            narrators = self._parse_narrators(metadata.get('narrators', []))
            formatted_source = self._create_formatted_source(metadata)
            narration_chain = self._create_narration_chain(narrators, metadata.get('exporter'))

            # Add hadith to results
            formatted_results.append({
                'id': metadata.get('id'),
                'text': document,
                'book_name': metadata.get('book_name'),
                'volume': metadata.get('volume'),
                'page_number': metadata.get('page_number'),
                'chapter': metadata.get('chapter'),
                'exporter': metadata.get('exporter'),
                'narrators': narrators,
                'url': metadata.get('url'),
                'get_formatted_source': formatted_source,
                'get_narration_chain': narration_chain,
                'similarity_score': (1 - distance) * 100
            })

        return formatted_results

    def _generate_langchain_rag_response(self, query: str, context_texts: List[str], 
                                       max_tokens: Optional[int] = None) -> str:
        """
        Generate a response using LangChain RAG.

        Args:
            query (str): The user query
            context_texts (List[str]): List of context texts
            max_tokens (Optional[int]): Not used in this implementation

        Returns:
            str: The generated response

        Raises:
            ValueError: If LangChain LLM is not initialized
        """
        if not self.langchain_llm:
            raise ValueError("LangChain LLM not initialized")

        # Prepare context
        context = "\n\n".join([f"Hadith: {text}" for text in context_texts])

        # Create prompt template
        template = """
        You are an AI assistant specialized in Islamic hadiths.
        Answer the following question based on the provided hadiths.
        If the answer cannot be found in the provided hadiths, say so clearly.
        Do not make up information. Cite the hadiths you use in your answer.

        Context hadiths:
        {context}

        Question: {question}

        Answer:
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Format the prompt
        formatted_prompt = prompt.format(context=context, question=query)

        # Generate response directly
        try:
            response = self.langchain_llm.invoke(formatted_prompt)
            
            # Extract the text content
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Error generating LangChain RAG response: {e}")
            raise
