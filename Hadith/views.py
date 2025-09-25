from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from .services.hadith_service import HadithService
from .repositories.hadith_repository import HadithRepository
import json
import asyncio

# Initialize services
hadith_service = HadithService()

def search_hadiths(request):
    """View for searching hadiths"""
    query = request.GET.get('q', '')
    search_type = request.GET.get('type', 'db')  # 'db', 'semantic', or 'rag'
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 10))

    # Get search mode and book filter for database search
    search_mode = request.GET.get('search_mode', 'contains')  # 'contains', 'all_words', 'exact', 'any_word'

    # Handle multiple book selections
    book_filters = request.GET.getlist('book_filter', [])

    # For backward compatibility
    if not book_filters and request.GET.get('book_filter'):
        book_filters = [request.GET.get('book_filter')]

    if not query:
        return render(request, 'Hadith/index.html')

    try:
        # Handle different search types
        if search_type == 'semantic':
            # Semantic search using vector embeddings
            try:
                # Get number of results from request
                semantic_results = int(request.GET.get('semantic_results', 10))

                hadiths = hadith_service.semantic_search(query, semantic_results)
                return render(request, 'Hadith/search_results.html', {
                    'hadiths': hadiths,
                    'query': query,
                    'search_type': search_type,
                    'semantic_results': semantic_results
                })
            except ValueError as e:
                # Fallback to database search if semantic search is not available
                search_type = 'db'

        elif search_type == 'rag':
            # RAG search with optional answer generation
            try:
                # RAG search always generates an answer
                generate_answer = True

                # Get number of results to use as context
                rag_results = int(request.GET.get('rag_results', 5))

                # Perform RAG search
                rag_result = hadith_service.rag_search(query, n_results=rag_results, generate_answer=generate_answer)

                return render(request, 'Hadith/rag_results.html', {
                    'query': query,
                    'answer': rag_result['answer'],
                    'sources': rag_result['sources'],
                    'search_type': search_type,
                    'generate_answer': generate_answer,
                    'rag_results': rag_results
                })
            except ValueError as e:
                # Fallback to database search if RAG is not available
                print(e)
                search_type = 'db'

        elif search_type == 'agentic':
            # Agentic RAG search with intelligent agent capabilities
            try:
                # Parse agentic RAG configuration parameters
                use_internet = request.GET.get('use_internet', 'true').lower() == 'true'
                max_subagents = request.GET.get('max_subagents')
                if max_subagents:
                    max_subagents = int(max_subagents)
                memory_enabled = request.GET.get('memory_enabled', 'true').lower() == 'true'

                # Perform agentic RAG search
                agentic_result = hadith_service.agentic_rag_search(
                    query=query,
                    use_internet=use_internet,
                    max_subagents=max_subagents,
                    memory_enabled=memory_enabled,
                    fallback_to_rag=True
                )

                return render(request, 'Hadith/agentic_results.html', {
                    'query': query,
                    'answer': agentic_result['answer'],
                    'reasoning_steps': agentic_result.get('reasoning_steps', []),
                    'tool_usage': agentic_result.get('tool_usage', {}),
                    'sources': agentic_result['sources'],
                    'metadata': agentic_result['metadata'],
                    'search_type': search_type,
                    'use_internet': use_internet,
                    'max_subagents': max_subagents,
                    'memory_enabled': memory_enabled
                })
            except ValueError as e:
                # Fallback to database search if agentic RAG is not available
                print(f"Agentic RAG error: {e}")
                search_type = 'db'
        print(search_type)

        # Default to database search
        if search_type == 'db':
            # Prepare filters
            filters = {}
            if book_filters:
                filters['books'] = book_filters

            # Database search with search mode and filters
            page_obj, total_results = hadith_service.search_hadiths(
                query, page, per_page, filters, search_mode
            )

            return render(request, 'Hadith/search_results.html', {
                'page_obj': page_obj,
                'query': query,
                'search_type': search_type,
                'search_mode': search_mode,
                'book_filters': book_filters,
                'total_results': total_results
            })

    except Exception as e:
        return render(request, 'Hadith/error.html', {
            'error': str(e)
        })

def index(request):
    # Import here to avoid circular imports
    from Hadith.utils.cache_utils import get_book_names

    # Get book names from cache
    book_names = get_book_names()

    return render(request, 'Hadith/index.html', {
        'book_names': book_names
    })


def agentic_stream_demo(request):
    """Demo page for streaming agentic search"""
    return render(request, 'Hadith/agentic_stream_demo.html')


def debug_agentic_service(request):
    """Debug endpoint to check agentic service status"""
    from django.http import JsonResponse
    from django.conf import settings
    from datetime import datetime

    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'gemini_api_key_set': bool(getattr(settings, 'GEMINI_API_KEY', '')),
        'tavily_api_key_set': bool(getattr(settings, 'TAVILY_API_KEY', '')),
    }

    # Test deepagents
    try:
        from deepagents import create_deep_agent
        debug_info['deepagents_available'] = True
    except ImportError as e:
        debug_info['deepagents_available'] = False
        debug_info['deepagents_error'] = str(e)

    # Test hadith service
    try:
        debug_info['hadith_service_created'] = True
        debug_info['agentic_rag_service'] = hadith_service.agentic_rag is not None

        if hadith_service.agentic_rag:
            debug_info['agentic_rag_available'] = hadith_service.agentic_rag.is_available
            debug_info['agentic_rag_agent'] = hadith_service.agentic_rag.agent is not None
            debug_info['agentic_rag_tools'] = len(getattr(hadith_service.agentic_rag, 'tools', []))
        else:
            debug_info['agentic_rag_available'] = False

    except Exception as e:
        debug_info['hadith_service_created'] = False
        debug_info['hadith_service_error'] = str(e)

    # Test simple search if available
    if debug_info.get('agentic_rag_available', False):
        try:
            result = hadith_service.agentic_rag_search(
                query="test",
                use_internet=False,
                memory_enabled=False,
                fallback_to_rag=False
            )
            debug_info['test_search_success'] = True
            debug_info['test_search_result_type'] = type(result).__name__
            if isinstance(result, dict):
                debug_info['test_search_keys'] = list(result.keys())
                if 'error' in result.get('metadata', {}):
                    debug_info['test_search_error'] = result['metadata']['error']
        except Exception as e:
            debug_info['test_search_success'] = False
            debug_info['test_search_error'] = str(e)

    return JsonResponse(debug_info, json_dumps_params={'indent': 2})


def api_hadiths(request):
    """API endpoint to get hadiths with filtering and pagination"""
    # Get query parameters
    query = request.GET.get('q', '')
    exporter = request.GET.get('exporter', '')
    volume = request.GET.get('volume', '')
    page = int(request.GET.get('page', 1))
    per_page = int(request.GET.get('per_page', 10))
    search_mode = request.GET.get('search_mode', 'contains')  # 'contains', 'all_words', 'exact', 'any_word'

    # Handle multiple book selections
    books = request.GET.getlist('book', [])

    # For backward compatibility
    if not books and request.GET.get('book'):
        books = [request.GET.get('book')]

    try:
        # Prepare filters
        filters = {}
        if query:
            filters['query'] = query
        if books:
            filters['books'] = books
        if exporter:
            filters['exporter'] = exporter
        if volume:
            filters['volume'] = volume

        # Get hadiths using service
        if query:
            page_obj, total_count = hadith_service.search_hadiths(
                query, page, per_page, filters, search_mode
            )
        else:
            page_obj, total_count = hadith_service.get_all_hadiths(
                page, per_page, filters
            )

        # Format response
        results = []
        for hadith in page_obj:
            results.append(HadithRepository.format_hadith_dict(hadith))

        return JsonResponse({
            'success': True,
            'total': total_count,
            'page': page,
            'per_page': per_page,
            'total_pages': page_obj.paginator.num_pages,
            'search_mode': search_mode,
            'results': results
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def api_hadith_detail(request, hadith_id):
    """API endpoint to get a single hadith by ID"""
    try:
        # Get hadith using service
        hadith_dict = hadith_service.get_hadith_by_id(hadith_id)

        if not hadith_dict:
            return JsonResponse({
                'success': False,
                'error': f'Hadith with ID {hadith_id} not found'
            }, status=404)

        return JsonResponse({
            'success': True,
            'hadith': hadith_dict
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def api_semantic_search(request):
    """API endpoint for semantic search using vector embeddings"""
    query = request.GET.get('q', '')
    n_results = int(request.GET.get('n', 10))

    if not query:
        return JsonResponse({
            'success': False,
            'error': 'Query parameter is required'
        }, status=400)

    try:
        # Perform semantic search using service
        hadiths = hadith_service.semantic_search(query, n_results)

        return JsonResponse({
            'success': True,
            'query': query,
            'results': hadiths
        })
    except ValueError as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=503)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def api_rag_search(request):
    """API endpoint for RAG search with generated answer"""
    query = request.GET.get('q', '')
    n_results = int(request.GET.get('n', 5))
    use_langchain = request.GET.get('langchain', 'false').lower() == 'true'

    if not query:
        return JsonResponse({
            'success': False,
            'error': 'Query parameter is required'
        }, status=400)

    try:
        # Perform RAG search using service
        rag_result = hadith_service.rag_search(query, n_results, use_langchain=use_langchain)

        return JsonResponse({
            'success': True,
            'query': query,
            'answer': rag_result['answer'],
            'sources': rag_result['sources'],
            'using_langchain': use_langchain
        })
    except ValueError as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=503)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


def api_agentic_search(request):
    """API endpoint for agentic RAG search with intelligent agent capabilities"""
    query = request.GET.get('q', '')
    
    if not query:
        return JsonResponse({
            'success': False,
            'error': 'Query parameter is required'
        }, status=400)

    try:
        # Parse agentic RAG configuration parameters
        use_internet = request.GET.get('use_internet', 'true').lower() == 'true'
        max_subagents = request.GET.get('max_subagents')
        if max_subagents:
            try:
                max_subagents = int(max_subagents)
                if max_subagents < 1 or max_subagents > 10:  # Reasonable limits
                    return JsonResponse({
                        'success': False,
                        'error': 'max_subagents must be between 1 and 10'
                    }, status=400)
            except ValueError:
                return JsonResponse({
                    'success': False,
                    'error': 'max_subagents must be a valid integer'
                }, status=400)
        
        memory_enabled = request.GET.get('memory_enabled', 'true').lower() == 'true'
        fallback_to_rag = request.GET.get('fallback_to_rag', 'true').lower() == 'true'

        # Perform agentic RAG search
        agentic_result = hadith_service.agentic_rag_search(
            query=query,
            use_internet=use_internet,
            max_subagents=max_subagents,
            memory_enabled=memory_enabled,
            fallback_to_rag=fallback_to_rag
        )

        return JsonResponse({
            'success': True,
            'query': query,
            'answer': agentic_result['answer'],
            'reasoning_steps': agentic_result.get('reasoning_steps', []),
            'tool_usage': agentic_result.get('tool_usage', {}),
            'sources': agentic_result['sources'],
            'metadata': agentic_result['metadata']
        })
        
    except ValueError as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=503)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }, status=500)


def api_recommended_hadiths(request, hadith_id):
    """API endpoint for getting recommended hadiths similar to a given hadith"""
    try:
        # Convert hadith_id to int
        hadith_id = int(hadith_id)
        n_results = int(request.GET.get('n', 5))

        # Get recommended hadiths using the text-matching task type
        recommended_hadiths = hadith_service.vector_store.get_recommended_hadiths(hadith_id, n_results)

        return JsonResponse({
            'success': True,
            'hadith_id': hadith_id,
            'recommendations': recommended_hadiths
        })
    except ValueError as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


async def api_agentic_search_stream(request):
    """API endpoint for streaming agentic RAG search with real-time progress updates"""
    query = request.GET.get('q', '')

    if not query:
        def error_generator():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Query parameter is required'})}\n\n"

        return StreamingHttpResponse(
            error_generator(),
            content_type='text/event-stream',
            status=400
        )

    try:
        # Parse agentic RAG configuration parameters
        use_internet = request.GET.get('use_internet', 'true').lower() == 'true'
        max_subagents = request.GET.get('max_subagents')
        if max_subagents:
            try:
                max_subagents = int(max_subagents)
            except ValueError:
                def error_generator():
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid max_subagents parameter'})}\n\n"

                return StreamingHttpResponse(
                    error_generator(),
                    content_type='text/event-stream',
                    status=400
                )

        memory_enabled = request.GET.get('memory_enabled', 'true').lower() == 'true'

        async def stream_generator():
            """Generator function for streaming search results"""
            try:
                async for update in hadith_service.agentic_rag_search_stream(
                    query=query,
                    use_internet=use_internet,
                    max_subagents=max_subagents,
                    memory_enabled=memory_enabled
                ):
                    # Format as Server-Sent Events
                    yield f"data: {json.dumps(update)}\n\n"

                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        def sync_wrapper():
            """Synchronous wrapper for the async generator"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async_gen = stream_generator()
                while True:
                    try:
                        result = loop.run_until_complete(async_gen.__anext__())
                        yield result
                    except StopAsyncIteration:
                        break
            finally:
                loop.close()

        response = StreamingHttpResponse(
            sync_wrapper(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['Connection'] = 'keep-alive'
        response['Access-Control-Allow-Origin'] = '*'
        return response

    except Exception as e:
        def error_generator():
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingHttpResponse(
            error_generator(),
            content_type='text/event-stream',
            status=500
        )
