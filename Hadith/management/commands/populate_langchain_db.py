# Hadith/management/commands/populate_langchain_db.py
from django.core.management.base import BaseCommand
from Hadith.models import Hadith
try:
    # Try to import from langchain_chroma (new package)
    from langchain_chroma import Chroma
except ImportError:
    # Fall back to community version
    from langchain_community.vectorstores import Chroma

from langchain_core.documents import Document
import numpy as np
import os
import time
import json

class Command(BaseCommand):
    help = 'Populate LangChain ChromaDB with hadith documents'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
        parser.add_argument('--force', action='store_true', help='Force recreate the collection')

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        force = options['force']

        # Path to LangChain ChromaDB data - this is separate from the main ChromaDB
        # Using a separate database allows for different embedding approaches
        chroma_path = "./langchain_db"

        # Check if collection exists and handle force option
        if os.path.exists(chroma_path) and force:
            self.stdout.write(self.style.WARNING(f'Removing existing LangChain ChromaDB at {chroma_path}'))
            import shutil
            shutil.rmtree(chroma_path)

        # Create directory if it doesn't exist
        os.makedirs(chroma_path, exist_ok=True)

        # Get hadiths that have text_without_tashkeel
        hadiths = Hadith.objects.exclude(text_without_tashkeel__isnull=True).exclude(text_without_tashkeel='')
        total = hadiths.count()

        if total == 0:
            self.stdout.write(self.style.ERROR('No hadiths found with text_without_tashkeel. Run update_tashkeel first.'))
            return

        self.stdout.write(self.style.SUCCESS(f'Found {total} hadiths to process'))

        # Process in batches
        processed = 0
        start_time = time.time()

        # Create documents for LangChain
        for i in range(0, total, batch_size):
            batch = hadiths[i:i+batch_size]
            documents = self.process_batch(batch)

            # Add documents to ChromaDB
            if documents:
                try:
                    # Create a custom embedding function that returns random embeddings
                    # This is just for testing - in production, you'd use a real embedding model
                    class RandomEmbeddings:
                        def __init__(self, dimension=768):
                            self.dimension = dimension

                        def embed_documents(self, texts):
                            return [list(np.random.rand(self.dimension)) for _ in texts]

                        def embed_query(self, text):
                            return list(np.random.rand(self.dimension))

                    # Create or get the vector store
                    if i == 0:  # First batch
                        vector_store = Chroma.from_documents(
                            documents=documents,
                            collection_name="hadith_collection",
                            persist_directory=chroma_path,
                            embedding_function=RandomEmbeddings()
                        )
                        vector_store.persist()
                    else:  # Subsequent batches
                        vector_store = Chroma(
                            collection_name="hadith_collection",
                            persist_directory=chroma_path,
                            embedding_function=RandomEmbeddings()
                        )
                        vector_store.add_documents(documents)
                        vector_store.persist()

                    processed += len(documents)

                    # Calculate progress and ETA
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    eta = (total - processed) / rate if rate > 0 else 0

                    self.stdout.write(self.style.SUCCESS(
                        f'Processed {processed}/{total} hadiths ({processed/total*100:.1f}%) - '
                        f'Rate: {rate:.1f} hadiths/sec - ETA: {eta/60:.1f} minutes'
                    ))
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'Error adding documents to ChromaDB: {e}'))

        self.stdout.write(self.style.SUCCESS(f'Successfully processed {processed} hadiths'))

    def process_batch(self, batch):
        """Process a batch of hadiths and convert to LangChain documents"""
        documents = []

        for hadith in batch:
            try:
                # Create metadata
                metadata = {
                    'hadith_id': hadith.hadith_id,
                    'source': hadith.source,
                    'narrator': hadith.narrator,
                    'book_name': hadith.book_name,
                    'chapter': hadith.chapter or '',
                    'url': hadith.url
                }

                # Create document
                doc = Document(
                    page_content=hadith.text_without_tashkeel,
                    metadata=metadata
                )

                documents.append(doc)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error processing hadith {hadith.hadith_id}: {e}'))

        return documents
