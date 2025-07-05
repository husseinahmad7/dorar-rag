# hadith_app/management/commands/load_hadiths.py
from django.core.management.base import BaseCommand
import json
from Hadith.models import Hadith
import os
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Optional imports for embedding generation
try:
    import chromadb
    import chromadb.utils.embedding_functions as embedding_functions
    from google import genai
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv(override=True)

    # Get API keys and configuration
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    jinaai_api_key = os.getenv('JINAAI_API_KEY')
    embedding_model = os.getenv('EMBEDDING_MODEL', 'jina-embeddings-v3')

    # Initialize embedding function based on model
    embedding_function = None
    embedding_available = False

    # Default task type for document storage
    default_task_type = "retrieval.passage"

    # Check if using Jina embeddings
    if embedding_model.startswith('jina') and jinaai_api_key:
        print(f"Using Jina embedding model: {embedding_model} with task_type={default_task_type}")
        embedding_function = embedding_functions.JinaEmbeddingFunction(
            api_key=jinaai_api_key,
            model_name=embedding_model,
            task=default_task_type  # Use retrieval.passage for document storage
        )
        embedding_available = True
    # Check if using Google embeddings
    elif gemini_api_key:
        print(f"Using Google embedding model: {embedding_model}")
        genai_client = genai.Client(api_key=gemini_api_key)
        embedding_available = True
    else:
        print("No embedding API keys found. Embeddings will be skipped.")
        embedding_available = False

    # Initialize ChromaDB if available
    chroma_path = os.getenv('CHROMA_DB_PATH', "./chroma_db")
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    try:
        # Try to get or create collection with embedding function if available
        if embedding_function:
            collection = chroma_client.get_or_create_collection(
                name="hadith_collection",
                embedding_function=embedding_function
            )
        else:
            collection = chroma_client.get_or_create_collection("hadith_collection")
        chroma_available = True
    except ValueError as e:
        print(f"Error with ChromaDB collection: {e}")
        # Collection doesn't exist yet or there was an error
        chroma_available = False
except Exception as e:
    print(f"Embedding or ChromaDB not available: {e}")
    embedding_available = False
    chroma_available = False


class Command(BaseCommand):
    help = 'Load hadiths from JSON file and create embeddings'

    def add_arguments(self, parser):
        parser.add_argument('json_file', type=str, help='Path to the JSON file containing hadiths')
        parser.add_argument('--skip-embeddings', action='store_true', help='Skip generating embeddings')
        parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
        parser.add_argument('--embedding-model', type=str, help='Embedding model to use (e.g., jina-embeddings-v3, text-multilingual-embedding-002)')
        parser.add_argument('--task-type', type=str, default='retrieval.passage',
                          choices=['retrieval.passage', 'retrieval.query', 'text-matching', 'separation', 'classification'],
                          help='Task type for Jina embeddings (default: retrieval.passage)')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    def handle(self, **options):
        json_file = options['json_file']
        skip_embeddings = options['skip_embeddings']
        batch_size = options['batch_size']
        custom_embedding_model = options.get('embedding_model')
        task_type = options.get('task_type', 'retrieval.passage')
        verbose = options.get('verbose', False)

        # Configure logging level
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # Log the task type
        self.stdout.write(self.style.SUCCESS(f'Using task type: {task_type}'))

        # Override embedding model if specified
        global embedding_model, embedding_function, embedding_available, chroma_available, collection, jinaai_api_key, gemini_api_key, genai_client

        if custom_embedding_model:
            self.stdout.write(self.style.SUCCESS(f'Using custom embedding model: {custom_embedding_model}'))
            embedding_model = custom_embedding_model

            # Reinitialize embedding function based on the new model
            if embedding_model.startswith('jina') and jinaai_api_key:
                self.stdout.write(self.style.SUCCESS(f'Using Jina embedding model: {embedding_model} with task_type={task_type}'))
                embedding_function = embedding_functions.JinaEmbeddingFunction(
                    api_key=jinaai_api_key,
                    model_name=embedding_model,
                    task=task_type  # Use the specified task type
                )
                embedding_available = True
            elif gemini_api_key:
                # No need to reinitialize genai_client, just update the model name
                embedding_available = True
            else:
                embedding_available = False

            # Reinitialize ChromaDB collection with the new embedding function
            if embedding_available and embedding_function:
                try:
                    collection = chroma_client.get_or_create_collection(
                        name="hadith_collection",
                        embedding_function=embedding_function
                    )
                    chroma_available = True
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'Error reinitializing ChromaDB collection: {e}'))
                    chroma_available = False

        if not skip_embeddings and (not embedding_available or not chroma_available):
            self.stdout.write(self.style.WARNING('Embedding generation is not available. Will skip embeddings.'))
            skip_embeddings = True

        self.stdout.write(self.style.SUCCESS('Reading JSON file...'))

        with open(json_file, 'r', encoding='utf-8') as f:
            hadiths = json.load(f)

        self.stdout.write(self.style.SUCCESS(f'Found {len(hadiths)} hadiths'))

        # Process in batches
        total = len(hadiths)
        processed = 0
        start_time = time.time()

        for i in range(0, total, batch_size):
            batch = hadiths[i:i+batch_size]
            self.process_batch(batch, skip_embeddings, task_type)
            processed += len(batch)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0

            self.stdout.write(self.style.SUCCESS(
                f'Processed {processed}/{total} hadiths ({processed/total*100:.1f}%) - '
                f'Rate: {rate:.1f} hadiths/sec - ETA: {eta/60:.1f} minutes'
            ))

        self.stdout.write(self.style.SUCCESS(f'Successfully processed {processed} hadiths'))

    def process_batch(self, batch, skip_embeddings, task_type='retrieval.passage'):
        """Process a batch of hadiths and optionally generate embeddings

        Args:
            batch: List of hadith dictionaries to process
            skip_embeddings: Whether to skip embedding generation
            task_type: Task type for Jina embeddings (default: retrieval.passage)
        """
        for hadith_data in batch:
            try:
                # Extract fields from hadith_data
                hadith_id = hadith_data['id']

                # Get text and generate text without tashkeel
                text = hadith_data['text']
                text_without_tashkeel = Hadith.remove_tashkeel(text)

                # Create or update Hadith model
                hadith, created = Hadith.objects.update_or_create(
                    hadith_id=hadith_id,
                    defaults={
                        'text': text,
                        'text_without_tashkeel': text_without_tashkeel,
                        'book_name': hadith_data.get('book_name', ''),
                        'volume': hadith_data.get('volume', ''),
                        'chapter': hadith_data.get('chapter', ''),
                        'page_number': hadith_data.get('page', ''),
                        'exporter': hadith_data.get('exporter', ''),
                        'narrators': hadith_data.get('narrators', []),
                        'url': hadith_data.get('url', f"https://hadith.inoor.ir/ar/hadith/{hadith_id}"),
                        'is_embedded': False  # Will be set to True when embedded
                    }
                )

                # Generate and store embedding if enabled
                if not skip_embeddings and embedding_available and chroma_available:
                    try:
                        # Prepare metadata - ensure all values are strings
                        metadata = {k: str(v) for k, v in hadith_data.items() if v is not None}

                        # Use text without tashkeel for better search results
                        text_to_embed = text_without_tashkeel

                        # If using Jina embedding function with ChromaDB
                        if embedding_function:
                            # Create a custom embedding function with the specified task type
                            try:
                                # Create a task-specific embedding function
                                custom_ef = embedding_functions.JinaEmbeddingFunction(
                                    api_key=jinaai_api_key,
                                    model_name=embedding_model,
                                    task=task_type
                                )

                                # Log the task type being used
                                logger.info(f"Using task_type={task_type} for hadith {hadith_id}")

                                # Store in ChromaDB with the custom embedding function
                                collection.embedding_function = custom_ef
                                collection.upsert(
                                    documents=[text_to_embed],
                                    metadatas=[metadata],
                                    ids=[str(hadith_id)]
                                )

                                # For model storage, we need to generate the embedding manually
                                # This is just for reference, as ChromaDB handles the embeddings
                                embedding = custom_ef([text_to_embed])[0]

                                # Update the hadith model to mark it as embedded
                                hadith.is_embedded = True
                                hadith.save(update_fields=['is_embedded'])

                                self.stdout.write(self.style.SUCCESS(
                                    f'Stored embedding for hadith {hadith_id} with task_type={task_type}'
                                ))
                            except Exception as e:
                                self.stdout.write(self.style.WARNING(f'Could not store embedding in model for hadith {hadith_id}: {e}'))
                                logger.error(f"Jina embedding error for hadith {hadith_id}: {e}")

                        # If using Google embeddings
                        elif gemini_api_key:
                            # Generate embedding using Google
                            # Map Jina task types to Google task types
                            google_task_type = "RETRIEVAL_DOCUMENT"  # Default for document storage
                            if task_type == "retrieval.query":
                                google_task_type = "RETRIEVAL_QUERY"

                            # Log the task type being used
                            logger.info(f"Using Google task_type={google_task_type} for hadith {hadith_id}")

                            result = genai_client.models.embed_content(
                                model=embedding_model,
                                contents=text_to_embed,
                                config={
                                    "task_type": google_task_type
                                }
                            )

                            # Extract the embedding values
                            embedding = result.embeddings[0].values

                            # Update the hadith model to mark it as embedded
                            hadith.is_embedded = True
                            hadith.save(update_fields=['is_embedded'])

                            # Store in ChromaDB
                            collection.upsert(
                                documents=[text_to_embed],
                                metadatas=[metadata],
                                ids=[str(hadith_id)],
                                embeddings=[embedding]
                            )
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f'Error generating embedding for hadith {hadith_id}: {e}'))

                # Log success
                action = 'Created' if created else 'Updated'
                self.stdout.write(self.style.SUCCESS(f'{action} hadith {hadith_id}'))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error processing hadith {hadith_data.get("id")}: {e}'))