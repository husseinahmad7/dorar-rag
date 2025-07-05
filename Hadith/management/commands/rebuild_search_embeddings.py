# Hadith/management/commands/rebuild_search_embeddings.py
from django.core.management.base import BaseCommand
from Hadith.models import Hadith
from Hadith.services.vector_store import VectorStoreService
import time
import os

class Command(BaseCommand):
    help = 'Rebuild embeddings optimized for search quality'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int, default=20, help='Batch size for processing')
        parser.add_argument('--limit', type=int, default=None, help='Limit the number of hadiths to process')
        parser.add_argument('--force', action='store_true', help='Force update all embeddings')

    def handle(self, **options):
        batch_size = options['batch_size']
        limit = options['limit']
        force = options['force']
        
        # Initialize vector store service
        vector_store = VectorStoreService()
        
        if not vector_store.is_available:
            self.stdout.write(self.style.ERROR('Vector store service not available'))
            return
        
        # Get hadiths to process
        if force:
            self.stdout.write(self.style.WARNING('Forcing update of all embeddings.'))
            hadiths = Hadith.objects.all()
        else:
            self.stdout.write(self.style.WARNING('Updating missing embeddings only.'))
            hadiths = Hadith.objects.filter(embedding_json__isnull=True)
        
        if limit:
            hadiths = hadiths[:limit]
            self.stdout.write(self.style.WARNING(f'Limiting to {limit} hadiths.'))
        
        total = hadiths.count()
        self.stdout.write(self.style.SUCCESS(f'Found {total} hadiths that need updating.'))
        
        if total == 0:
            self.stdout.write(self.style.SUCCESS('No hadiths need updating.'))
            return
        
        # Process in batches
        processed = 0
        start_time = time.time()
        
        # First, reset the ChromaDB collection
        try:
            collection = vector_store._get_or_create_collection()
            self.stdout.write(self.style.WARNING('Resetting ChromaDB collection...'))
            vector_store.chroma_client.delete_collection("hadith_collection")
            collection = vector_store.chroma_client.create_collection("hadith_collection")
            self.stdout.write(self.style.SUCCESS('ChromaDB collection reset successfully.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error resetting ChromaDB collection: {e}'))
            self.stdout.write(self.style.WARNING('Continuing with existing collection...'))
        
        # Process hadiths
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
    
    def process_batch(self, batch, vector_store):
        for hadith in batch:
            try:
                # Get text without tashkeel
                text_without_tashkeel = hadith.text_without_tashkeel or Hadith.remove_tashkeel(hadith.text)
                
                # Generate embedding optimized for search (using is_query=True)
                # This is the key change - we're generating embeddings as if they were queries
                embedding = vector_store.generate_embedding(text_without_tashkeel, use_without_tashkeel=False, is_query=True)
                
                # Store embedding in model
                hadith.embedding = embedding
                hadith.save(update_fields=['embedding_json'])
                
                # Store in ChromaDB
                try:
                    metadata = {
                        'id': hadith.hadith_id,
                        'source': hadith.source,
                        'narrator': hadith.narrator,
                        'book_name': hadith.book_name,
                        'chapter': hadith.chapter,
                        'url': hadith.url
                    }
                    vector_store.store_embedding(hadith.hadith_id, text_without_tashkeel, metadata)
                    self.stdout.write(self.style.SUCCESS(f'Updated embedding for hadith {hadith.hadith_id}'))
                except Exception as e:
                    # If ChromaDB fails, we still have the embedding in the database
                    self.stdout.write(self.style.WARNING(
                        f'Embedding saved to database but ChromaDB storage failed for hadith {hadith.hadith_id}: {e}'
                    ))
                    # Continue with the next hadith
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error updating embedding for hadith {hadith.hadith_id}: {e}'))
