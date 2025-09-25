"""
Service for hadith operations
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from Hadith.repositories.hadith_repository import HadithRepository
from Hadith.services.vector_store import VectorStoreService

# Optional import for agentic RAG
try:
    from Hadith.services.agentic_rag import AgenticRagService
    agentic_rag_available = True
except ImportError as e:
    agentic_rag_available = False
    logging.warning(f"AgenticRagService not available: {e}")

# Configure logging
logger = logging.getLogger(__name__)

class HadithService:
    """
    Service for hadith operations.
    
    This service provides various search capabilities including traditional text search,
    semantic search, RAG (Retrieval-Augmented Generation), and agentic RAG with
    internet integration.
    """

    def __init__(self):
        """Initialize the hadith service with all available search capabilities."""
        self.repository = HadithRepository()
        self.vector_store = VectorStoreService()
        
        # Initialize agentic RAG service if available
        self.agentic_rag = None
        if agentic_rag_available:
            try:
                logger.info("Attempting to initialize agentic RAG service...")
                self.agentic_rag = AgenticRagService(self.vector_store)
                if self.agentic_rag.is_available:
                    logger.info("Agentic RAG service initialized successfully")
                else:
                    logger.warning("Agentic RAG service created but marked as unavailable")
            except Exception as e:
                logger.error(f"Failed to initialize agentic RAG service: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.agentic_rag = None
        else:
            logger.warning("Agentic RAG service not available - import failed")

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

    def agentic_rag_search(self, query: str, use_internet: Optional[bool] = None,
                          max_subagents: Optional[int] = None, 
                          memory_enabled: bool = True,
                          fallback_to_rag: bool = True) -> Dict[str, Any]:
        """
        Perform agentic RAG search with intelligent agent capabilities.
        
        This method uses an intelligent agent that can decide when to use hadith search,
        internet search, or both, and combines the results for comprehensive answers.
        
        Args:
            query (str): The search query
            use_internet (Optional[bool]): Whether to enable internet search.
                                         If None, uses service default
            max_subagents (Optional[int]): Maximum number of subagents to use.
                                         If None, uses service default
            memory_enabled (bool): Whether to use conversation memory for context
            fallback_to_rag (bool): Whether to fall back to regular RAG if agentic RAG fails
            
        Returns:
            Dict[str, Any]: Structured results including:
                - query: The original query
                - answer: The agent's final answer
                - reasoning_steps: List of agent's reasoning steps
                - tool_usage: Dictionary of tools used and their inputs
                - sources: Dictionary with 'hadith_sources' and 'web_sources'
                - metadata: Search metadata including timing and configuration
                
        Raises:
            ValueError: If agentic RAG is not available and fallback is disabled
        """
        logger.info(f'Agentic RAG search for query: "{query}"')
        
        # Check if agentic RAG is available
        logger.info(f"Checking agentic RAG availability - service: {self.agentic_rag is not None}, is_available: {self.agentic_rag.is_available if self.agentic_rag else 'N/A'}")

        if not self.agentic_rag or not self.agentic_rag.is_available:
            if fallback_to_rag:
                logger.warning(f"Agentic RAG not available (service: {self.agentic_rag is not None}, available: {self.agentic_rag.is_available if self.agentic_rag else 'N/A'}), falling back to regular RAG")
                return self._fallback_to_regular_rag(query)
            else:
                raise ValueError("Agentic RAG service not available")
        
        try:
            # Perform agentic RAG search
            logger.info(f"Calling agentic RAG search with params: use_internet={use_internet}, max_subagents={max_subagents}, memory_enabled={memory_enabled}")
            result = self.agentic_rag.search(
                query=query,
                use_internet=use_internet,
                max_subagents=max_subagents,
                memory_enabled=memory_enabled
            )

            logger.info(f"Agentic RAG search completed successfully for query: {query}")
            return result

        except Exception as e:
            logger.error(f"Error in agentic RAG search: {e}")

            if fallback_to_rag:
                logger.warning("Agentic RAG failed, falling back to regular RAG")
                return self._fallback_to_regular_rag(query, error_message=str(e))
            else:
                raise

    async def agentic_rag_search_stream(self, query: str, use_internet: Optional[bool] = None,
                                       max_subagents: Optional[int] = None,
                                       memory_enabled: bool = True):
        """
        Perform streaming agentic RAG search with real-time progress updates.

        This method provides real-time updates about the agent's progress, including
        which tools are being used and intermediate results.

        Args:
            query (str): The search query
            use_internet (Optional[bool]): Whether to enable internet search
            max_subagents (Optional[int]): Maximum number of subagents to use
            memory_enabled (bool): Whether to use conversation memory for context

        Yields:
            Dict[str, Any]: Progress updates and final results
        """
        logger.info(f'Streaming agentic RAG search for query: "{query}"')

        # Check if agentic RAG is available
        if not self.agentic_rag or not self.agentic_rag.is_available:
            yield {
                'type': 'error',
                'message': 'Agentic RAG service not available',
                'timestamp': datetime.now().isoformat()
            }
            return

        try:
            # Stream the search results
            async for update in self.agentic_rag.asearch_stream(
                query=query,
                use_internet=use_internet,
                max_subagents=max_subagents,
                memory_enabled=memory_enabled
            ):
                yield update

        except Exception as e:
            logger.error(f"Error in streaming agentic RAG search: {e}")
            yield {
                'type': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _fallback_to_regular_rag(self, query: str, error_message: Optional[str] = None) -> Dict[str, Any]:
        """
        Fallback to regular RAG search when agentic RAG is not available or fails.
        
        Args:
            query (str): The search query
            error_message (Optional[str]): Error message from failed agentic RAG attempt
            
        Returns:
            Dict[str, Any]: Structured results in agentic RAG format
        """
        try:
            # Perform regular RAG search
            rag_result = self.rag_search(query, n_results=5, generate_answer=True)
            
            # Format the result to match agentic RAG structure
            fallback_result = {
                'query': query,
                'answer': rag_result.get('answer', 'No answer generated'),
                'reasoning_steps': [
                    "Using fallback RAG search due to agentic RAG unavailability",
                    f"Searched hadith collection for: {query}",
                    f"Found {len(rag_result.get('sources', []))} relevant hadiths",
                    "Generated answer based on retrieved hadiths"
                ],
                'tool_usage': {
                    'hadith_search': [f"query: {query}"]
                },
                'sources': {
                    'hadith_sources': [
                        result.get('get_formatted_source', 'Unknown source')
                        for result in rag_result.get('sources', [])
                    ],
                    'web_sources': []
                },
                'metadata': {
                    'search_time_seconds': 0,
                    'memory_enabled': False,
                    'agent_available': False,
                    'tools_used': ['hadith_search'],
                    'fallback_used': True,
                    'fallback_reason': error_message or 'Agentic RAG not available'
                }
            }
            
            logger.info("Successfully generated fallback RAG response")
            return fallback_result
            
        except Exception as e:
            logger.error(f"Error in fallback RAG search: {e}")
            # Return a minimal error response
            return {
                'query': query,
                'answer': f"I apologize, but I encountered an error while searching: {str(e)}",
                'reasoning_steps': ["Error occurred during search"],
                'tool_usage': {},
                'sources': {'hadith_sources': [], 'web_sources': []},
                'metadata': {
                    'search_time_seconds': 0,
                    'memory_enabled': False,
                    'agent_available': False,
                    'tools_used': [],
                    'fallback_used': True,
                    'error': str(e)
                }
            }
    
    def clear_agentic_memory(self) -> bool:
        """
        Clear the agentic RAG conversation memory.
        
        Returns:
            bool: True if memory was cleared successfully, False otherwise
        """
        if self.agentic_rag and self.agentic_rag.is_available:
            try:
                self.agentic_rag.clear_memory()
                logger.info("Agentic RAG memory cleared successfully")
                return True
            except Exception as e:
                logger.error(f"Error clearing agentic RAG memory: {e}")
                return False
        else:
            logger.warning("Agentic RAG not available, cannot clear memory")
            return False
    
    def get_agentic_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current agentic RAG memory state.
        
        Returns:
            Dict[str, Any]: Memory summary including availability and message count
        """
        if self.agentic_rag and self.agentic_rag.is_available:
            try:
                return self.agentic_rag.get_memory_summary()
            except Exception as e:
                logger.error(f"Error getting agentic RAG memory summary: {e}")
                return {'available': False, 'error': str(e)}
        else:
            return {'available': False, 'reason': 'Agentic RAG not available'}
    
    def is_agentic_rag_available(self) -> bool:
        """
        Check if agentic RAG functionality is available.
        
        Returns:
            bool: True if agentic RAG is available and ready to use
        """
        return (self.agentic_rag is not None and 
                self.agentic_rag.is_available)
