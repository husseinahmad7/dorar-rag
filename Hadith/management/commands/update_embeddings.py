# Hadith/management/commands/update_embeddings.py
from django.core.management.base import BaseCommand
from Hadith.models import Hadith
from Hadith.services.vector_store import VectorStoreService
import time
import logging

# Configure logging
logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = 'Update embeddings for all hadiths using the latest embedding model'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
        parser.add_argument('--force', action='store_true', help='Force update all embeddings even if they exist')
        parser.add_argument('--task-type', type=str, default='retrieval.passage',
                          choices=['retrieval.passage', 'retrieval.query', 'text-matching', 'separation', 'classification'],
                          help='Task type for Jina embeddings (default: retrieval.passage)')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    def handle(self, **options):
        batch_size = options['batch_size']
        force = options['force']
        task_type = options['task_type']
        verbose = options['verbose']

        # Configure logging level
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # Initialize vector store service
        try:
            vector_store = VectorStoreService()

            if not vector_store.is_available:
                self.stdout.write(self.style.ERROR('Vector store service not available. Check your API key.'))
                return

            # Log embedding model information
            self.stdout.write(self.style.SUCCESS(f'Using embedding model: {vector_store.embedding_model}'))
            self.stdout.write(self.style.SUCCESS(f'Using task type: {task_type}'))

            # Check if using Jina
            if not vector_store.use_jina:
                self.stdout.write(self.style.WARNING('Not using Jina embeddings. Task type will be ignored.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error initializing vector store service: {e}'))
            return

        # Get hadiths that need updating
        if force:
            hadiths_to_update = Hadith.objects.all()
            self.stdout.write(self.style.SUCCESS('Forcing update of all embeddings.'))
        else:
            hadiths_to_update = Hadith.objects.filter(is_embedded=False)
            self.stdout.write(self.style.SUCCESS('Updating hadiths not yet embedded.'))

        total = hadiths_to_update.count()

        if total == 0:
            self.stdout.write(self.style.SUCCESS('No hadiths need updating.'))
            return

        self.stdout.write(self.style.SUCCESS(f'Found {total} hadiths that need updating.'))

        # Process in batches
        processed = 0
        start_time = time.time()

        for i in range(0, total, batch_size):
            batch = hadiths_to_update[i:i+batch_size]
            self.process_batch(batch, vector_store, task_type=task_type)
            processed += len(batch)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0

            self.stdout.write(self.style.SUCCESS(
                f'Processed {processed}/{total} hadiths ({processed/total*100:.1f}%) - '
                f'Rate: {rate:.1f} hadiths/sec - ETA: {eta/60:.1f} minutes'
            ))

        self.stdout.write(self.style.SUCCESS(f'Successfully updated {processed} hadiths'))

    def process_batch(self, batch, vector_store, task_type='retrieval.passage'):
        """Process a batch of hadiths to update their embeddings

        Args:
            batch: List of Hadith objects to process
            vector_store: VectorStoreService instance
            task_type: Task type for Jina embeddings (default: retrieval.passage)
        """
        import chromadb.utils.embedding_functions as embedding_functions

        # Create a custom embedding function with the specified task type if using Jina
        jina_ef_custom = None
        if vector_store.use_jina and vector_store.jinaai_api_key:
            try:
                jina_ef_custom = embedding_functions.JinaEmbeddingFunction(
                    api_key=vector_store.jinaai_api_key,
                    model_name=vector_store.embedding_model,
                    task=task_type
                )
                logger.info(f"Created custom Jina embedding function with task_type={task_type}")
            except Exception as e:
                logger.error(f"Failed to create custom Jina embedding function: {e}")

        # Get or create collection with the custom embedding function
        try:
            if jina_ef_custom:
                collection = vector_store.chroma_client.get_collection(
                    name="hadith_collection",
                    embedding_function=jina_ef_custom
                )
                logger.info(f"Using collection with custom embedding function (task_type={task_type})")
            else:
                collection = vector_store._get_or_create_collection()
                logger.info("Using default collection")
        except Exception as e:
            logger.error(f"Error getting collection: {e}")
            collection = None

        for hadith in batch:
            try:
                # Get text without tashkeel
                text_without_tashkeel = hadith.text_without_tashkeel or Hadith.remove_tashkeel(hadith.text)

                # Prepare metadata
                metadata = {
                    'id': hadith.hadith_id,
                    'book_name': hadith.book_name,
                    'volume': hadith.volume,
                    'chapter': hadith.chapter,
                    'page_number': hadith.page_number,
                    'exporter': hadith.exporter,
                    'narrators': hadith.narrators,
                    'url': hadith.url
                }

                # Store in ChromaDB
                try:
                    # If we have a custom collection with the right task type, use it directly
                    if collection and jina_ef_custom:
                        # Process metadata - ensure all values are strings
                        processed_metadata = {k: str(v) for k, v in metadata.items() if v is not None}

                        # Store directly in ChromaDB
                        collection.upsert(
                            documents=[text_without_tashkeel],
                            metadatas=[processed_metadata],
                            ids=[str(hadith.hadith_id)]
                        )

                        # Mark hadith as embedded
                        hadith.is_embedded = True
                        hadith.save(update_fields=['is_embedded'])

                        self.stdout.write(self.style.SUCCESS(
                            f'Updated embedding for hadith {hadith.hadith_id} with task_type={task_type}'
                        ))
                    else:
                        # Use the standard method which will handle embedding generation
                        vector_store.store_embedding(hadith.hadith_id, text_without_tashkeel, metadata)
                        self.stdout.write(self.style.SUCCESS(f'Updated embedding for hadith {hadith.hadith_id}'))
                except Exception as e:
                    # If ChromaDB fails, the hadith won't be marked as embedded
                    self.stdout.write(self.style.WARNING(
                        f'ChromaDB storage failed for hadith {hadith.hadith_id}: {e}'
                    ))
                    logger.error(f"ChromaDB error for hadith {hadith.hadith_id}: {e}")
                    # Continue with the next hadith
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error updating embedding for hadith {hadith.hadith_id}: {e}'))
                logger.error(f"Processing error for hadith {hadith.hadith_id}: {e}")
