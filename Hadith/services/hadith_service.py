"""
Service for hadith operations
"""
from Hadith.repositories.hadith_repository import HadithRepository
from Hadith.services.vector_store import VectorStoreService

class HadithService:
    """Service for hadith operations"""

    def __init__(self):
        """Initialize the hadith service"""
        self.repository = HadithRepository()
        self.vector_store = VectorStoreService()

    def get_hadith_by_id(self, hadith_id):
        """
        Get a hadith by ID

        Args:
            hadith_id (int): The hadith ID

        Returns:
            dict: The hadith data
        """
        hadith = self.repository.get_by_id(hadith_id)
        if not hadith:
            return None

        return self.repository.format_hadith_dict(hadith)

    def search_hadiths(self, query, page=1, per_page=10, filters=None, search_mode='contains'):
        """
        Search hadiths by text and other criteria

        Args:
            query (str): The search query
            page (int): Page number for pagination
            per_page (int): Number of results per page
            filters (dict): Additional filters to apply
            search_mode (str): Search mode - 'contains' (default), 'all_words', 'exact', or 'any_word'

        Returns:
            tuple: (page_obj, total_count)
        """
        _, page_obj, total_count = self.repository.search(
            query, page, per_page, filters, search_mode
        )

        return page_obj, total_count

    def get_all_hadiths(self, page=1, per_page=10, filters=None):
        """
        Get all hadiths with optional filtering

        Args:
            page (int): Page number for pagination
            per_page (int): Number of results per page
            filters (dict): Filters to apply

        Returns:
            tuple: (page_obj, total_count)
        """
        _, page_obj, total_count = self.repository.get_all(
            page, per_page, filters
        )

        return page_obj, total_count

    def semantic_search(self, query, n_results=10):
        """
        Perform semantic search

        Args:
            query (str): The search query
            n_results (int): Number of results to return

        Returns:
            list: List of search results
        """
        print('semantic search')
        if not self.vector_store.is_available:
            raise ValueError("Vector store service not available")

        try:
            # Use the same retrieval approach as in RAG search for consistency
            if self.vector_store.langchain_retriever is not None:
                try:
                    # Update the LangChain retriever to use the specified number of results
                    self.vector_store.langchain_retriever.search_kwargs["k"] = n_results

                    # Use LangChain retriever to get relevant documents
                    docs = self.vector_store.langchain_retriever.invoke(query)
                    # For semantic search, we want to include similarity scores
                    search_results = self._convert_langchain_docs_to_results(docs, include_similarity=True)
                    print(f"Retrieved {len(search_results)} documents using LangChain retriever")
                    return search_results
                except Exception as e:
                    print(f"Error retrieving documents with LangChain: {e}")
                    # Fall back to direct semantic search

            # Fall back to direct semantic search if LangChain retriever is not available or failed
            search_results = self.vector_store.semantic_search(query, n_results, is_question=True)
            print(f"Found {len(search_results)} results for query: {query}")
            return search_results
        except Exception as e:
            raise ValueError(f"Error performing semantic search: {str(e)}")

    def rag_search(self, query, n_results=5, generate_answer=False):
        """
        Perform RAG search and optionally generate a response

        Args:
            query (str): The search query
            n_results (int): Number of results to use as context
            generate_answer (bool): Whether to generate an answer using LLM

        Returns:
            dict: RAG response with results and optionally a generated answer
        """
        print('rag search')
        if not self.vector_store.is_available:
            raise ValueError("Vector store service not available")

        try:
            # Try to use LangChain retriever if available
            if self.vector_store.langchain_retriever is not None:
                try:
                    # Update the LangChain retriever to use the specified number of results
                    self.vector_store.langchain_retriever.search_kwargs["k"] = n_results

                    # Use LangChain retriever to get relevant documents
                    docs = self.vector_store.langchain_retriever.invoke(query)
                    search_results = self._convert_langchain_docs_to_results(docs)
                    print(f"Retrieved {len(search_results)} documents using LangChain retriever for RAG")
                except Exception as e:
                    print(f"Error retrieving documents with LangChain: {e}")
                    # Fall back to semantic search if LangChain retriever fails
                    is_question = '?' in query or any(q in query for q in ['ما', 'ماذا', 'لماذا', 'كيف', 'متى', 'أين', 'من', 'هل'])
                    search_results = self.vector_store.semantic_search(query, n_results, is_question=is_question)
            else:
                # Fall back to semantic search if LangChain retriever is not available
                is_question = '?' in query or any(q in query for q in ['ما', 'ماذا', 'لماذا', 'كيف', 'متى', 'أين', 'من', 'هل'])
                search_results = self.vector_store.semantic_search(query, n_results, is_question=is_question)

            # Generate answer if requested
            answer = None
            if generate_answer and search_results:
                # Extract texts for context
                context_texts = [result['text'] for result in search_results]

                # Generate answer using the vector store service
                answer = self.vector_store.generate_rag_response(query, context_texts=context_texts)

            return {
                'query': query,
                'answer': answer,
                'sources': search_results
            }
        except Exception as e:
            raise ValueError(f"Error performing RAG search: {str(e)}")

    def _convert_langchain_docs_to_results(self, docs, include_similarity=False):
        """
        Convert LangChain documents to our standard result format

        Args:
            docs: LangChain documents
            include_similarity: Whether to include similarity score (default: False)

        Returns:
            list: Formatted search results
        """
        search_results = []

        for doc in docs:
            metadata = doc.metadata
            # Process narrators - it might be stored as a string representation of a list
            narrators_data = metadata.get('narrators', [])
            if isinstance(narrators_data, str):
                try:
                    import json
                    narrators = json.loads(narrators_data.replace("'", '"'))
                except Exception as e:
                    print(f"Error parsing narrators: {e}")
                    narrators = []
            else:
                narrators = narrators_data

            # Create formatted source
            book_name = metadata.get('book_name', '')
            volume = metadata.get('volume', '')
            page_number = metadata.get('page_number', '')
            chapter = metadata.get('chapter', '')
            formatted_source = book_name
            if volume:
                formatted_source += f", الجزء {volume}"
            if page_number:
                formatted_source += f", الصفحة {page_number}"
            if chapter:
                formatted_source += f", الباب: {chapter}"

            # Create narration chain
            narration_chain = ""
            if narrators:
                # Join narrators with 'عن'
                narrators_chain = " عن ".join(narrators[:-1]) if len(narrators) > 1 else ""

                # Add last narrator if available
                if len(narrators) > 0:
                    if narrators_chain:
                        narrators_chain += " عن " + narrators[-1]
                    else:
                        narrators_chain = narrators[-1]

                # Add exporter if available
                exporter = metadata.get('exporter')
                if exporter:
                    narration_chain = f"{narrators_chain} قال {exporter}"
                else:
                    narration_chain = narrators_chain
            elif metadata.get('exporter'):
                narration_chain = f"قال {metadata.get('exporter')}"

            result = {
                'id': metadata.get('id'),
                'text': doc.page_content,
                'book_name': metadata.get('book_name', ''),
                'volume': metadata.get('volume', ''),
                'page_number': metadata.get('page_number', ''),
                'chapter': metadata.get('chapter', ''),
                'exporter': metadata.get('exporter', ''),
                'narrators': narrators,
                'url': metadata.get('url', ''),
                'get_formatted_source': formatted_source,
                'get_narration_chain': narration_chain,
            }

            # Only include similarity score if requested
            # This helps avoid showing 100% similarity when we don't have real scores
            if include_similarity:
                result['similarity_score'] = 100.0

            search_results.append(result)

        return search_results
