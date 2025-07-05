# Hadith/management/commands/test_rag.py
from django.core.management.base import BaseCommand
from Hadith.services.vector_store import VectorStoreService
import time

class Command(BaseCommand):
    help = 'Test RAG functionality directly'

    def add_arguments(self, parser):
        parser.add_argument('query', type=str, help='Query to search for')
        parser.add_argument('--use-langchain', action='store_true', help='Use LangChain for RAG')

    def handle(self, *args, **options):
        query = options['query']
        use_langchain = options['use_langchain']
        
        # Initialize vector store service
        vector_store = VectorStoreService()
        
        if not vector_store.is_available:
            self.stdout.write(self.style.ERROR('Vector store service not available'))
            return
        
        self.stdout.write(self.style.SUCCESS(f'Testing RAG for query: "{query}"'))
        self.stdout.write(self.style.SUCCESS(f'Using LangChain: {use_langchain}'))
        
        try:
            # First get search results
            start_time = time.time()
            search_results = vector_store.semantic_search(query, n_results=5, is_question=True)
            search_time = time.time() - start_time
            
            self.stdout.write(self.style.SUCCESS(f'Found {len(search_results)} search results in {search_time:.2f} seconds'))
            
            # Print search results
            for i, result in enumerate(search_results[:3]):  # Show top 3 results
                self.stdout.write(self.style.SUCCESS(f'\nSearch Result {i+1}:'))
                self.stdout.write(f'Text: {result["text"]}')
                self.stdout.write(f'Similarity: {result["similarity_score"]:.2f}%')
            
            # Generate RAG response
            start_time = time.time()
            context_texts = [result['text'] for result in search_results]
            
            if use_langchain:
                response = vector_store._generate_langchain_rag_response(query, context_texts)
            else:
                response = vector_store._generate_standard_rag_response(query, context_texts)
                
            rag_time = time.time() - start_time
            
            self.stdout.write(self.style.SUCCESS(f'\nGenerated RAG response in {rag_time:.2f} seconds:'))
            self.stdout.write(self.style.SUCCESS('-' * 80))
            self.stdout.write(response)
            self.stdout.write(self.style.SUCCESS('-' * 80))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: {e}'))
