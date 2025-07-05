# Hadith/management/commands/direct_search_test.py
from django.core.management.base import BaseCommand
from Hadith.services.vector_store import VectorStoreService
import time

class Command(BaseCommand):
    help = 'Test semantic search directly using the vector store service'

    def add_arguments(self, parser):
        parser.add_argument('query', type=str, help='Query to search for')
        parser.add_argument('--n', type=int, default=5, help='Number of results to return')

    def handle(self, *args, **options):
        query = options['query']
        n_results = options['n']
        
        # Initialize vector store service
        vector_store = VectorStoreService()
        
        if not vector_store.is_available:
            self.stdout.write(self.style.ERROR('Vector store service not available'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Testing direct semantic search for query: "{query}"'))
        
        # Test with is_question=True
        try:
            start_time = time.time()
            results = vector_store.semantic_search(query, n_results, is_question=True)
            elapsed = time.time() - start_time
            self.stdout.write(self.style.SUCCESS(f'Found {len(results)} results in {elapsed:.2f} seconds'))
            
            # Print results
            for i, result in enumerate(results):
                self.stdout.write(self.style.SUCCESS(f'\nResult {i+1}:'))
                self.stdout.write(f'Text: {result["text"]}')
                self.stdout.write(f'Source: {result["source"]}')
                self.stdout.write(f'Similarity: {result["similarity_score"]:.2f}%')
                
                # Print metadata
                self.stdout.write(self.style.SUCCESS('Metadata:'))
                for key, value in result.items():
                    if key not in ['text', 'similarity_score']:
                        self.stdout.write(f'  {key}: {value}')
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: {e}'))
