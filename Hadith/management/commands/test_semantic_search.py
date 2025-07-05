# Hadith/management/commands/test_semantic_search.py
from django.core.management.base import BaseCommand
from Hadith.services.vector_store import VectorStoreService
import time

class Command(BaseCommand):
    help = 'Test semantic search with different queries'

    def add_arguments(self, parser):
        parser.add_argument('--query', type=str, help='Query to search for')
        parser.add_argument('--n', type=int, default=5, help='Number of results to return')
        parser.add_argument('--as-question', action='store_true', help='Treat query as a question')
        parser.add_argument('--compare', action='store_true', help='Compare results with and without is_question')

    def handle(self, **options):
        query = options['query']
        n_results = options['n']
        as_question = options['as_question']
        compare = options['compare']
        
        if not query:
            self.stdout.write(self.style.ERROR('Please provide a query with --query'))
            return
        
        # Initialize vector store service
        vector_store = VectorStoreService()
        
        if not vector_store.is_available:
            self.stdout.write(self.style.ERROR('Vector store service not available'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Testing semantic search for query: "{query}"'))
        
        if compare:
            # Test with is_question=False
            self.stdout.write(self.style.WARNING('\n--- Testing with is_question=False ---'))
            try:
                start_time = time.time()
                results_false = vector_store.semantic_search(query, n_results, is_question=False)
                elapsed = time.time() - start_time
                self.stdout.write(self.style.SUCCESS(f'Found {len(results_false)} results in {elapsed:.2f} seconds'))
                self._print_results(results_false)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error: {e}'))
            
            # Test with is_question=True
            self.stdout.write(self.style.WARNING('\n--- Testing with is_question=True ---'))
            try:
                start_time = time.time()
                results_true = vector_store.semantic_search(query, n_results, is_question=True)
                elapsed = time.time() - start_time
                self.stdout.write(self.style.SUCCESS(f'Found {len(results_true)} results in {elapsed:.2f} seconds'))
                self._print_results(results_true)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error: {e}'))
        else:
            # Test with specified is_question value
            try:
                start_time = time.time()
                results = vector_store.semantic_search(query, n_results, is_question=as_question)
                elapsed = time.time() - start_time
                self.stdout.write(self.style.SUCCESS(f'Found {len(results)} results in {elapsed:.2f} seconds'))
                self._print_results(results)
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error: {e}'))
    
    def _print_results(self, results):
        if not results:
            self.stdout.write(self.style.WARNING('No results found'))
            return
        
        for i, result in enumerate(results):
            self.stdout.write(self.style.SUCCESS(f'\nResult {i+1}:'))
            self.stdout.write(f'Text: {result["text"]}')
            self.stdout.write(f'Source: {result["source"]}')
            self.stdout.write(f'Similarity: {result["similarity_score"]:.2f}%')
