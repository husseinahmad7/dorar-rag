# Hadith/management/commands/reset_and_update_with_jina.py
from django.core.management.base import BaseCommand
from Hadith.models import Hadith
from Hadith.services.vector_store import VectorStoreService
import os
import time
import shutil

class Command(BaseCommand):
    help = 'Reset ChromaDB and update embeddings with Jina AI'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int, default=20, help='Batch size for processing')
        parser.add_argument('--limit', type=int, default=None, help='Limit the number of hadiths to process')
        parser.add_argument('--skip-reset', action='store_true', help='Skip resetting ChromaDB')

    def handle(self, **options):
        batch_size = options['batch_size']
        limit = options['limit']
        skip_reset = options['skip_reset']
        
        # Set environment variables for Jina
        os.environ['EMBEDDING_MODEL'] = 'jina-embeddings-v3'
        self.stdout.write(self.style.SUCCESS(f'Using embedding model: jina-embeddings-v3'))
        
        # Initialize vector store service
        vector_store = VectorStoreService()
        
        if not vector_store.is_available:
            self.stdout.write(self.style.ERROR('Vector store service not available'))
            return
        
        if not vector_store.use_jina or not vector_store.jina_ef:
            self.stdout.write(self.style.ERROR('Jina embedding function not initialized. Check your JINAAI_API_KEY.'))
            return
        
        # Reset ChromaDB if not skipped
        if not skip_reset:
            self.reset_chroma(vector_store)
        
        # Get hadiths to process
        hadiths = Hadith.objects.all()
        
        if limit:
            hadiths = hadiths[:limit]
            self.stdout.write(self.style.WARNING(f'Limiting to {limit} hadiths.'))
        
        total = hadiths.count()
        self.stdout.write(self.style.SUCCESS(f'Found {total} hadiths to process.'))
        
        if total == 0:
            self.stdout.write(self.style.SUCCESS('No hadiths to process.'))
            return
        
        # Process in batches
        processed = 0
        start_time = time.time()
        
        for i in range(0, total, batch_size):
            batch = hadiths[i:i+batch_size]
            self.process_batch(batch, vector_store)
            
            processed += len(batch)
            
            # Calculate progress and ETA
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            
            self.stdout.write(self.style.SUCCESS(
                f'Processed {processed}/{total} hadiths ({processed/total*100:.1f}%) - '
                f'Rate: {rate:.1f} hadiths/sec - ETA: {eta/60:.1f} minutes'
            ))
    
    def reset_chroma(self, vector_store):
        """Reset ChromaDB collection"""
        self.stdout.write(self.style.WARNING('Resetting ChromaDB collection...'))
        
        try:
            # Delete collection if it exists
            try:
                collection = vector_store.chroma_client.get_collection("hadith_collection")
                vector_store.chroma_client.delete_collection("hadith_collection")
                self.stdout.write(self.style.SUCCESS('Deleted existing collection'))
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'Collection not found or could not be deleted: {e}'))
            
            # Create a new collection with Jina embedding function
            collection = vector_store.chroma_client.create_collection(
                name="hadith_collection",
                metadata={"description": "Collection of hadith embeddings with Jina"},
                embedding_function=vector_store.jina_ef
            )
            self.stdout.write(self.style.SUCCESS('Created new collection with Jina embedding function'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error resetting ChromaDB: {e}'))
            self.stdout.write(self.style.WARNING('Continuing with existing collection...'))
    
    def process_batch(self, batch, vector_store):
        """Process a batch of hadiths"""
        for hadith in batch:
            try:
                # Get text without tashkeel
                text_without_tashkeel = hadith.text_without_tashkeel or Hadith.remove_tashkeel(hadith.text)
                
                # Create metadata
                metadata = {
                    'id': hadith.hadith_id,
                    'source': hadith.source,
                    'narrator': hadith.narrator,
                    'book_name': hadith.book_name,
                    'chapter': hadith.chapter,
                    'url': hadith.url
                }
                
                # Store in ChromaDB
                vector_store.store_embedding(hadith.hadith_id, text_without_tashkeel, metadata)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error processing hadith {hadith.hadith_id}: {e}'))
