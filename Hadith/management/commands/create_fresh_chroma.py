# Hadith/management/commands/create_fresh_chroma.py
from django.core.management.base import BaseCommand
import chromadb
import os
import time
import numpy as np
from Hadith.models import Hadith

class Command(BaseCommand):
    help = 'Create a fresh ChromaDB in a new location and populate it with hadiths'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
        parser.add_argument('--path', type=str, default='./new_chroma_db', help='Path for the new ChromaDB')

    def handle(self, **options):
        batch_size = options['batch_size']
        chroma_path = options['path']
        
        self.stdout.write(self.style.WARNING(f'Creating a fresh ChromaDB at {chroma_path}'))
        
        # Create directory if it doesn't exist
        os.makedirs(chroma_path, exist_ok=True)
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=chroma_path)
        
        # Create collection
        try:
            # Check if collection exists
            collections = client.list_collections()
            collection_exists = any(c.name == "hadith_collection" for c in collections)
            
            if collection_exists:
                self.stdout.write(self.style.WARNING('Collection already exists. Deleting it.'))
                client.delete_collection("hadith_collection")
            
            collection = client.create_collection("hadith_collection")
            self.stdout.write(self.style.SUCCESS('Created new collection'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error creating collection: {e}'))
            return
        
        # Get hadiths
        hadiths = Hadith.objects.all()
        total = hadiths.count()
        
        if total == 0:
            self.stdout.write(self.style.ERROR('No hadiths found in the database'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Found {total} hadiths to process'))
        
        # Process in batches
        processed = 0
        start_time = time.time()
        
        for i in range(0, total, batch_size):
            batch = hadiths[i:i+batch_size]
            
            # Prepare data for batch insertion
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            for hadith in batch:
                try:
                    # Use text without tashkeel if available
                    text = hadith.text_without_tashkeel or Hadith.remove_tashkeel(hadith.text)
                    
                    # Create metadata
                    metadata = {
                        'id': str(hadith.hadith_id),
                        'source': hadith.source,
                        'narrator': hadith.narrator,
                        'book_name': hadith.book_name,
                        'chapter': hadith.chapter or '',
                        'url': hadith.url
                    }
                    
                    # Generate random embedding (for testing only)
                    # In production, you'd use a real embedding model
                    embedding = list(np.random.rand(768))
                    
                    ids.append(str(hadith.hadith_id))
                    documents.append(text)
                    metadatas.append(metadata)
                    embeddings.append(embedding)
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'Error processing hadith {hadith.hadith_id}: {e}'))
            
            # Add to collection
            try:
                collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                
                processed += len(ids)
                
                # Calculate progress and ETA
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total - processed) / rate if rate > 0 else 0
                
                self.stdout.write(self.style.SUCCESS(
                    f'Processed {processed}/{total} hadiths ({processed/total*100:.1f}%) - '
                    f'Rate: {rate:.1f} hadiths/sec - ETA: {eta/60:.1f} minutes'
                ))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error adding batch to collection: {e}'))
        
        self.stdout.write(self.style.SUCCESS(f'Successfully processed {processed} hadiths'))
        self.stdout.write(self.style.SUCCESS(f'New ChromaDB created at {chroma_path}'))
        self.stdout.write(self.style.SUCCESS(f'To use this new database, update the chroma_path in vector_store.py'))
        self.stdout.write(self.style.SUCCESS(f'self.chroma_path = "{chroma_path}"'))
