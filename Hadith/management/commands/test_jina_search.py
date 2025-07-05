# Hadith/management/commands/test_jina_search.py
from django.core.management.base import BaseCommand
from Hadith.services.vector_store import VectorStoreService
import os
import time

class Command(BaseCommand):
    help = 'Test semantic search with Jina embeddings'

    def add_arguments(self, parser):
        parser.add_argument('query', type=str, help='Query to search for')
        parser.add_argument('--n', type=int, default=5, help='Number of results to return')
        parser.add_argument('--use-google', action='store_true', help='Use Google embeddings instead of Jina')

    def handle(self, *args, **options):
        query = options['query']
        n_results = options['n']
        use_google = options['use_google']
        
        # Set environment variables based on options
        if use_google:
            os.environ['EMBEDDING_MODEL'] = 'text-multilingual-embedding-002'
            self.stdout.write(self.style.WARNING('Using Google embeddings: text-multilingual-embedding-002'))
        else:
            os.environ['EMBEDDING_MODEL'] = 'jina-embeddings-v3'
            self.stdout.write(self.style.WARNING('Using Jina embeddings: jina-embeddings-v3'))
        
        # Initialize vector store service
        vector_store = VectorStoreService()
        
        if not vector_store.is_available:
            self.stdout.write(self.style.ERROR('Vector store service not available'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Testing semantic search for query: "{query}"'))
        
        # Perform semantic search
        try:
            start_time = time.time()
            results = vector_store.semantic_search(query, n_results)
            elapsed = time.time() - start_time
            
            self.stdout.write(self.style.SUCCESS(f'Found {len(results)} results in {elapsed:.2f} seconds'))
            
            # Print results
            for i, result in enumerate(results):
                self.stdout.write(self.style.SUCCESS(f'\nResult {i+1}:'))
                self.stdout.write(f'Text: {result["text"]}')
                self.stdout.write(f'Source: {result["source"]}')
                self.stdout.write(f'Similarity: {result["similarity_score"]:.2f}%')
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error performing semantic search: {e}'))
