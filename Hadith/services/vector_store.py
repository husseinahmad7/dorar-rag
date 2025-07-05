"""
Vector store service for embeddings and semantic search
"""
import os
import re
from google import genai
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv

# Optional imports for LangChain integration
try:
    from langchain.prompts import PromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    try:
        # Try to import from langchain_chroma (new package)
        # from langchain_chroma import Chroma # not compatable with chromadb1.0.5
        pass
    except ImportError:
        # Fall back to community version
        from langchain_community.vectorstores import Chroma
    # from langchain_core.documents import Document
    langchain_available = True
except ImportError as e:
    langchain_available = False
    print(f"LangChain not fully available: {e}. RAG with LangChain will not be available.")

# Load environment variables
load_dotenv(override=True)

class VectorStoreService:
    """Service for vector embeddings and semantic search"""

    @staticmethod
    def remove_tashkeel(text):
        """Remove Arabic diacritics (tashkeel) from text"""
        # Arabic diacritics Unicode ranges
        tashkeel = re.compile(r'[\u064b-\u0652\u0670]')
        return re.sub(tashkeel, '', text)

    def __init__(self):
        """Initialize the vector store service"""
        # API keys and configuration
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.jinaai_api_key = os.getenv('JINAAI_API_KEY')

        # ChromaDB configuration
        # Change this path if you want to use a different ChromaDB location
        self.chroma_path = os.getenv('CHROMA_DB_PATH', "./chroma_db")

        # Client instances
        self.genai_client = None
        self.chroma_client = None
        self.jina_ef = None  # Jina embedding function

        # Model names - easily configurable
        # You can change these models by setting environment variables
        self.embedding_model = os.getenv('EMBEDDING_MODEL', "jina-embeddings-v3")
        self.llm_model = os.getenv('LLM_MODEL', 'gemini-1.5-flash-002')

        # LangChain integration
        self.langchain_llm = None
        self.langchain_retriever = None
        self.use_langchain = langchain_available

        # Availability flag
        self.is_available = False

        # Flag to indicate if we're using Jina embeddings
        self.use_jina = self.embedding_model.startswith('jina')

        # Initialize if API keys are available
        if self.gemini_api_key and self.jinaai_api_key:
            try:
                # Initialize the Gemini client for LLM
                self.genai_client = genai.Client(api_key=self.gemini_api_key)
                print(f"Using LLM model: {self.llm_model}")

                # Initialize Jina embedding function if using Jina
                if self.use_jina:
                    if self.jinaai_api_key:
                        print(f"Using Jina embedding model: {self.embedding_model}")
                        # Initialize Jina embedding function with default task type
                        # We'll specify the task type when generating embeddings
                        self.jina_ef = embedding_functions.JinaEmbeddingFunction(
                            api_key=self.jinaai_api_key,
                            model_name=self.embedding_model
                        )
                    else:
                        print("WARNING: JINAAI_API_KEY not found but Jina embedding model selected")
                        self.is_available = False
                        return
                else:
                    print(f"Using Google embedding model: {self.embedding_model}")

                # Initialize ChromaDB with PersistentClient to store data on disk
                self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)

                # Initialize LangChain LLM if available
                if self.use_langchain:
                    self.langchain_llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash-002",
                        google_api_key=self.gemini_api_key,
                        temperature=0.2,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=1024
                    )

                    # Initialize LangChain retriever
                    self._init_langchain_retriever()

                # Set available flag based on models
                self.is_available = True
                print("Vector store service initialized successfully")
            except Exception as e:
                print(f"Error initializing vector store service: {e}")
                self.is_available = False
        else:
            print("Gemini API key or Jina API key not found. Vector store service not available.")

    def _init_langchain_retriever(self):
        """
        Initialize LangChain retriever with ChromaDB
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
                native_collection = self.chroma_client.get_collection("hadith_collection")
                print(f"Found existing collection with {native_collection.count()} documents")
            except Exception as e:
                print(f"Error getting collection: {e}")
                return

            # Create LangChain Chroma instance using the existing ChromaDB collection
            langchain_db = Chroma(
                collection_name="hadith_collection",
                persist_directory=self.chroma_path,
                embedding_function=CustomEmbeddingFunction(self.generate_embedding)
            )

            # Create retriever
            self.langchain_retriever = langchain_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            print("LangChain retriever initialized successfully")
        except Exception as e:
            print(f"Error initializing LangChain retriever: {e}")

    def generate_embedding(self, text, use_without_tashkeel=True, is_query=False):
        """
        Generate embedding for text

        Args:
            text (str): The text to embed
            use_without_tashkeel (bool): Whether to remove tashkeel before embedding
            is_query (bool): Whether this is a query (affects embedding task type)

        Returns:
            list: The embedding vector
        """
        if not self.is_available:
            raise ValueError("Vector store service not available")

        # Remove tashkeel if requested
        if use_without_tashkeel and hasattr(self, 'remove_tashkeel'):
            text_to_embed = self.remove_tashkeel(text)
        else:
            # Import the function from Hadith model if available
            try:
                from Hadith.models import Hadith
                text_to_embed = Hadith.remove_tashkeel(text)
            except (ImportError, AttributeError):
                # Fallback to original text if function not available
                text_to_embed = text

        # Detect if this is a question if not explicitly specified
        if not is_query:
            is_query = '?' in text_to_embed or any(q in text_to_embed for q in ['ما', 'ماذا', 'لماذا', 'كيف', 'متى', 'أين', 'من', 'هل'])

        # Check if we're using Jina embeddings
        if self.use_jina and self.jina_ef:
            try:
                # Log which model we're using
                print(f"Generating embedding with Jina model: {self.embedding_model}")

                # Generate embedding using Jina with the appropriate task type
                # Jina supports different task types for different use cases
                task_type = "retrieval.query" if is_query else "retrieval.passage"

                # Create a custom embedding function with the specific task type
                jina_ef_with_task = embedding_functions.JinaEmbeddingFunction(
                    api_key=self.jinaai_api_key,
                    model_name=self.embedding_model,
                    task=task_type
                )

                # Generate the embedding with the appropriate task type
                embedding = jina_ef_with_task([text_to_embed])[0]
                print(f"Generated Jina embedding with task_type: {task_type}")

                return embedding
            except Exception as e:
                print(f"Error generating embeddings with Jina model {self.embedding_model}: {e}")
        else:
            # Use Google API to generate embeddings
            try:
                # Log which model and task type we're using
                task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
                print(f"Generating embedding with Google model: {self.embedding_model}, task_type: {task_type}")

                result = self.genai_client.models.embed_content(
                    model=self.embedding_model,
                    contents=text_to_embed,
                    config={
                        "task_type": task_type
                    }
                )

                # Extract the embedding values
                return result.embeddings[0].values
            except Exception as e:
                print(f"Error generating embeddings with Google model {self.embedding_model}: {e}")

                # Try with a fallback model if the primary model fails
                fallback_model = "embedding-001"
                if self.embedding_model != fallback_model:
                    print(f"Trying fallback Google embedding model: {fallback_model}")
                    try:
                        result = self.genai_client.models.embed_content(
                            model=fallback_model,
                            contents=text_to_embed,
                            config={
                                "task_type": "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
                            }
                        )
                        return result.embeddings[0].values
                    except Exception as e2:
                        print(f"Error with fallback model: {e2}")

                # As a last resort, return a random embedding (for testing only)
                import numpy as np
                print("WARNING: Returning random embedding as fallback. This should only be used for testing!")
                return list(np.random.rand(768))

    def _get_or_create_collection(self):
        """
        Get or create the ChromaDB collection

        Returns:
            chromadb.Collection: The ChromaDB collection
        """
        try:
            # Try to get the collection
            if self.use_jina and self.jina_ef:
                # Use Jina embedding function with ChromaDB
                # For collection retrieval, we use the retrieval.passage task type
                # as we're working with stored documents
                jina_ef_passage = embedding_functions.JinaEmbeddingFunction(
                    api_key=self.jinaai_api_key,
                    model_name=self.embedding_model,
                    task="retrieval.passage"
                )
                collection = self.chroma_client.get_collection(
                    name="hadith_collection",
                    embedding_function=jina_ef_passage
                )
            else:
                # Use default embedding function
                collection = self.chroma_client.get_collection("hadith_collection")
            print(f"Found existing collection with {collection.count()} documents")
            return collection
        except Exception as e:
            # Collection doesn't exist yet or there was an error
            print(f"Creating new collection: {e}")
            if self.use_jina and self.jina_ef:
                # Use Jina embedding function with ChromaDB
                # For collection creation, we use the retrieval.passage task type
                # as we're storing documents
                jina_ef_passage = embedding_functions.JinaEmbeddingFunction(
                    api_key=self.jinaai_api_key,
                    model_name=self.embedding_model,
                    task="retrieval.passage"
                )
                return self.chroma_client.create_collection(
                    name="hadith_collection",
                    metadata={
                        "description": "Collection of hadith embeddings with Jina",
                        "model": self.embedding_model,
                        "task_type": "retrieval.passage"
                    },
                    embedding_function=jina_ef_passage
                )
            else:
                # Use default embedding function
                return self.chroma_client.create_collection(
                    name="hadith_collection",
                    metadata={"description": "Collection of hadith embeddings"}
                )

    def semantic_search(self, query, n_results=10, is_question=None):
        """
        Perform semantic search

        Args:
            query (str): The search query
            n_results (int): Number of results to return
            is_question (bool, optional): Force query to be treated as a question

        Returns:
            list: List of search results
        """
        print(f'Semantic search for query: "{query}" (is_question={is_question})')
        if not self.is_available:
            raise ValueError("Vector store service not available")

        # If using Jina with ChromaDB's embedding function, we don't need to generate embeddings
        # ChromaDB will use the embedding function automatically
        query_embedding = None
        if not (self.use_jina and self.jina_ef):
            # Always treat search queries as questions for better results
            # This is a key change - we're forcing is_query=True for all semantic searches
            query_embedding = self.generate_embedding(query, use_without_tashkeel=True, is_query=True)
            print(f"Generated embedding with is_query=True (overriding is_question={is_question})")

        # Get or create collection
        collection = self._get_or_create_collection()

        # Print collection info for debugging
        try:
            count = collection.count()
            print(f"Collection has {count} documents")
            if count == 0:
                print("WARNING: Collection is empty! Run update_embeddings to populate it.")
                return []
        except Exception as e:
            print(f"Error getting collection count: {e}")

        # Search in ChromaDB
        try:
            if self.use_jina and self.jina_ef:
                # When using Jina embedding function, we need to create a query-specific function
                # with the retrieval.query task type
                jina_ef_query = embedding_functions.JinaEmbeddingFunction(
                    api_key=self.jinaai_api_key,
                    model_name=self.embedding_model,
                    task="retrieval.query"  # Use retrieval.query for search queries
                )

                # Set the embedding function for the query
                collection.embedding_function = jina_ef_query

                # Execute the query
                results = collection.query(
                    query_texts=[query],  # Use query_texts instead of query_embeddings
                    n_results=min(n_results, 100),  # Limit to 100 max results
                    include=['metadatas', 'documents', 'distances']
                )
            else:
                # When using manual embeddings, we provide query_embeddings
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(n_results, 100),  # Limit to 100 max results
                    include=['metadatas', 'documents', 'distances']
                )
            print(f"Query returned {len(results['documents'][0])} results")
        except Exception as e:
            print(f"Error during ChromaDB query: {e}")
            raise

        # Format results
        hadiths = []
        for metadata, document, distance in zip(
            results['metadatas'][0],
            results['documents'][0],
            results['distances'][0]
        ):
            # Process narrators - it might be stored as a string representation of a list
            narrators_data = metadata.get('narrators', [])
            if isinstance(narrators_data, str):
                try:
                    import json
                    narrators = json.loads(narrators_data.replace("'", '"'))
                except Exception as e:
                    print(f"Error parsing narrators: {e}")
                    narrators = []
            else:
                narrators = narrators_data

            # Create formatted source
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

            # Create narration chain
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
                exporter = metadata.get('exporter')
                if exporter:
                    narration_chain = f"{narrators_chain} قال {exporter}"
                else:
                    narration_chain = narrators_chain
            elif metadata.get('exporter'):
                narration_chain = f"قال {metadata.get('exporter')}"

            # Add hadith to results
            hadiths.append({
                'id': metadata.get('id'),
                'text': document,
                'book_name': metadata.get('book_name'),
                'volume': metadata.get('volume'),
                'page_number': metadata.get('page_number'),
                'chapter': metadata.get('chapter'),  # Add chapter field
                'exporter': metadata.get('exporter'),
                'narrators': narrators,
                'url': metadata.get('url'),
                'get_formatted_source': formatted_source,  # Add formatted source
                'get_narration_chain': narration_chain,    # Add narration chain
                'similarity_score': (1 - distance) * 100  # Convert distance to percentage
            })

        return hadiths

    def store_embedding(self, hadith_id, text, metadata):
        """
        Store embedding in vector store

        Args:
            hadith_id (int): The hadith ID
            text (str): The text to embed
            metadata (dict): Metadata for the document

        Returns:
            bool: True if successful
        """
        if not self.is_available:
            raise ValueError("Vector store service not available")

        try:
            # Get or create collection
            collection = self._get_or_create_collection()

            # Prepare metadata - ensure all values are strings
            processed_metadata = {k: str(v) for k, v in metadata.items() if v is not None}

            # If using Jina with ChromaDB's embedding function, we need to use the correct task type
            # For storing documents, we use the retrieval.passage task type
            if self.use_jina and self.jina_ef:
                # Create a document-specific embedding function with retrieval.passage task type
                jina_ef_passage = embedding_functions.JinaEmbeddingFunction(
                    api_key=self.jinaai_api_key,
                    model_name=self.embedding_model,
                    task="retrieval.passage"  # Use retrieval.passage for document storage
                )

                # Set the embedding function for document storage
                collection.embedding_function = jina_ef_passage

                # Store in ChromaDB without providing embeddings
                collection.upsert(
                    documents=[text],
                    metadatas=[processed_metadata],
                    ids=[str(hadith_id)]
                )
            else:
                # Generate embedding manually
                embedding = self.generate_embedding(text)

                # Store in ChromaDB with embeddings
                collection.upsert(
                    documents=[text],
                    metadatas=[processed_metadata],
                    ids=[str(hadith_id)],
                    embeddings=[embedding]
                )

            # Update the hadith model to mark it as embedded
            try:
                from Hadith.models import Hadith
                hadith = Hadith.objects.get(hadith_id=hadith_id)
                hadith.is_embedded = True
                hadith.save(update_fields=['is_embedded'])
                print(f"Successfully stored embedding for hadith {hadith_id} and marked as embedded")
            except Exception as e:
                print(f"Error updating hadith model: {e}")

            return True
        except Exception as e:
            print(f"Error storing embedding in ChromaDB: {e}")
            raise

    def generate_rag_response(self, query, context_texts=None, max_tokens=1024, use_langchain=None):
        """
        Generate a response using RAG

        Args:
            query (str): The user query
            context_texts (list, optional): List of context texts. If None and use_langchain=True,
                                           will use LangChain retriever to get contexts
            max_tokens (int): Maximum tokens for response
            use_langchain (bool): Whether to use LangChain for RAG

        Returns:
            str: The generated response
        """
        if not self.is_available:
            raise ValueError("Vector store service not available")

        # Determine whether to use LangChain
        if use_langchain is None:
            use_langchain = self.use_langchain

        # If LangChain is requested but not available, fall back to standard method
        if use_langchain and not self.use_langchain:
            print("LangChain requested but not available. Falling back to standard RAG.")
            use_langchain = False

        if use_langchain:
            # If context_texts is None, use LangChain retriever to get contexts
            if context_texts is None and self.langchain_retriever is not None:
                try:
                    # Use LangChain retriever to get relevant documents
                    # Using invoke instead of get_relevant_documents (which is deprecated)
                    docs = self.langchain_retriever.invoke(query)
                    context_texts = [doc.page_content for doc in docs]
                    print(f"Retrieved {len(context_texts)} documents using LangChain retriever")
                except Exception as e:
                    print(f"Error retrieving documents with LangChain: {e}")
                    # Fall back to semantic search if retriever fails
                    if not context_texts:
                        search_results = self.semantic_search(query, n_results=5, is_question=True)
                        context_texts = [result['text'] for result in search_results]

            return self._generate_langchain_rag_response(query, context_texts, max_tokens)
        else:
            # If context_texts is None, use semantic search to get contexts
            if context_texts is None:
                search_results = self.semantic_search(query, n_results=5, is_question=True)
                context_texts = [result['text'] for result in search_results]

            return self._generate_standard_rag_response(query, context_texts, max_tokens)

    def _generate_standard_rag_response(self, query, context_texts, max_tokens=1024):
        """
        Generate a response using standard RAG (without LangChain)

        Args:
            query (str): The user query
            context_texts (list): List of context texts
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

        # Generate response using the new API
        try:
            # First try with generation_config parameter
            response = self.genai_client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config={
                    # "max_output_tokens": max_tokens,
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "top_k": 40
                }
            )
        except TypeError as e:
            print(f"Error with individual parameters: {e}")
            # try with just the basic parameters
            response = self.genai_client.models.generate_content(
                model=self.llm_model,
                contents=prompt
            )

        return response.text

    def get_recommended_hadiths(self, hadith_id, n_results=5):
        """
        Get recommended hadiths similar to the given hadith using text-matching task type

        Args:
            hadith_id (int): The hadith ID to find recommendations for
            n_results (int): Number of recommendations to return

        Returns:
            list: List of recommended hadiths
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

            # For recommendations, we use the text-matching task type
            if self.use_jina and self.jina_ef:
                # Create a text-matching embedding function
                jina_ef_matching = embedding_functions.JinaEmbeddingFunction(
                    api_key=self.jinaai_api_key,
                    model_name=self.embedding_model,
                    task_type="text-matching"  # Use text-matching for recommendations
                )

                # Set the embedding function for recommendations
                collection.embedding_function = jina_ef_matching

                # Execute the query
                results = collection.query(
                    query_texts=[text],
                    n_results=min(n_results + 1, 100),  # +1 to account for the query hadith itself
                    include=['metadatas', 'documents', 'distances']
                )
            else:
                # Generate embedding with text-matching task type
                # For Google embeddings, we don't have a text-matching option
                embedding = self.generate_embedding(text, is_query=False)

                # Execute the query
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=min(n_results + 1, 100),
                    include=['metadatas', 'documents', 'distances']
                )

            # Filter out the query hadith itself
            filtered_results = {
                'ids': [],
                'metadatas': [],
                'documents': [],
                'distances': []
            }

            for i, id_val in enumerate(results['ids'][0]):
                if id_val != str(hadith_id):
                    filtered_results['ids'].append(id_val)
                    filtered_results['metadatas'].append(results['metadatas'][0][i])
                    filtered_results['documents'].append(results['documents'][0][i])
                    filtered_results['distances'].append(results['distances'][0][i])

            # Format results
            hadiths = []
            for i in range(min(n_results, len(filtered_results['ids']))):
                metadata = filtered_results['metadatas'][i]
                document = filtered_results['documents'][i]
                distance = filtered_results['distances'][i]

                # Process narrators
                narrators_data = metadata.get('narrators', [])
                if isinstance(narrators_data, str):
                    try:
                        import json
                        narrators = json.loads(narrators_data.replace("'", '"'))
                    except Exception as e:
                        print(f"Error parsing narrators: {e}")
                        narrators = []
                else:
                    narrators = narrators_data

                # Create formatted source
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

                # Create narration chain
                narration_chain = ""
                if narrators:
                    narrators_chain = " عن ".join(narrators[:-1]) if len(narrators) > 1 else ""
                    if len(narrators) > 0:
                        if narrators_chain:
                            narrators_chain += " عن " + narrators[-1]
                        else:
                            narrators_chain = narrators[-1]
                    exporter = metadata.get('exporter')
                    if exporter:
                        narration_chain = f"{narrators_chain} قال {exporter}"
                    else:
                        narration_chain = narrators_chain
                elif metadata.get('exporter'):
                    narration_chain = f"قال {metadata.get('exporter')}"

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
                    'similarity_score': (1 - distance) * 100
                })

            return hadiths
        except Exception as e:
            print(f"Error getting recommended hadiths: {e}")
            return []

    def _generate_langchain_rag_response(self, query, context_texts, max_tokens=None):
        """
        Generate a response using LangChain RAG

        Args:
            query (str): The user query
            context_texts (list): List of context texts
            max_tokens (int, optional): Not used in this implementation

        Returns:
            str: The generated response
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
        response = self.langchain_llm.invoke(formatted_prompt)

        # Extract the text content
        if hasattr(response, 'content'):
            return response.content
        else:
            return str(response)
