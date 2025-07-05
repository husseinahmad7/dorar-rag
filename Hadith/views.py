from django.shortcuts import render
from django.http import JsonResponse
from .services.hadith_service import HadithService
from .repositories.hadith_repository import HadithRepository

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