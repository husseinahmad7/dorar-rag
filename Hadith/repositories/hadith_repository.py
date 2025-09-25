"""
Repository for Hadith data access
"""
from django.db.models import Q
from django.core.paginator import Paginator
from Hadith.models import Hadith

class HadithRepository:
    """Repository for accessing Hadith data"""

    @staticmethod
    def get_by_id(hadith_id):
        """Get a hadith by ID"""
        return Hadith.objects.filter(hadith_id=hadith_id).first()

    @staticmethod
    def search(query, page=1, per_page=10, filters=None, search_mode='contains'):
        """
        Search hadiths by text and other criteria

        Args:
            query (str): The search query
            page (int): Page number for pagination
            per_page (int): Number of results per page
            filters (dict): Additional filters to apply
            search_mode (str): Search mode - 'contains' (default), 'all_words', 'exact', or 'any_word'

        Returns:
            tuple: (paginator, page_obj, total_count)
        """
        # Remove diacritics from query for better matching
        query_without_tashkeel = Hadith.remove_tashkeel(query)

        # Build base query based on search mode
        if search_mode == 'exact':
            # Exact phrase match
            base_query = Q(text__icontains=query) | \
                        Q(text_without_tashkeel__icontains=query_without_tashkeel)
        elif search_mode == 'all_words':
            # All words must be present
            base_query = Q()
            words = query.split()
            words_without_tashkeel = query_without_tashkeel.split()

            for word, word_without_tashkeel in zip(words, words_without_tashkeel):
                base_query &= (Q(text__icontains=word) | Q(text_without_tashkeel__icontains=word_without_tashkeel))
        elif search_mode == 'any_word':
            # Any word can match
            base_query = Q()
            words = query.split()
            words_without_tashkeel = query_without_tashkeel.split()

            for word, word_without_tashkeel in zip(words, words_without_tashkeel):
                base_query |= (Q(text__icontains=word) | Q(text_without_tashkeel__icontains=word_without_tashkeel))
        else:  # Default 'contains' mode
            # Default behavior - contains the phrase
            base_query = Q(text__icontains=query) | \
                        Q(text_without_tashkeel__icontains=query_without_tashkeel) | \
                        Q(book_name__icontains=query) | \
                        Q(exporter__icontains=query)

        # Apply additional filters if provided
        if filters:
            filter_query = Q()

            # Handle book filter (single book or multiple books)
            if filters.get('book'):
                filter_query &= Q(book_name__icontains=filters['book'])
            elif filters.get('books'):
                book_filter = Q()
                for book in filters['books']:
                    book_filter |= Q(book_name__icontains=book)
                if book_filter:
                    filter_query &= book_filter

            if filters.get('exporter'):
                filter_query &= Q(exporter__icontains=filters['exporter'])
            if filters.get('volume'):
                filter_query &= Q(volume__icontains=filters['volume'])

            # Combine with base query
            base_query &= filter_query

        # Execute query
        hadiths = Hadith.objects.filter(base_query)

        # Paginate results
        paginator = Paginator(hadiths, per_page)
        page_obj = paginator.get_page(page)

        return paginator, page_obj, hadiths.count()

    @staticmethod
    def get_all(page=1, per_page=10, filters=None):
        """
        Get all hadiths with optional filtering

        Args:
            page (int): Page number for pagination
            per_page (int): Number of results per page
            filters (dict): Filters to apply

        Returns:
            tuple: (paginator, page_obj, total_count)
        """
        # Start with all hadiths
        query = Hadith.objects.all()

        # Apply filters if provided
        if filters:
            # Handle book filter (single book or multiple books)
            if filters.get('book'):
                query = query.filter(book_name__icontains=filters['book'])
            elif filters.get('books'):
                book_filter = Q()
                for book in filters['books']:
                    book_filter |= Q(book_name=book)
                if book_filter:
                    query = query.filter(book_filter)

            if filters.get('exporter'):
                query = query.filter(exporter__icontains=filters['exporter'])
            if filters.get('volume'):
                query = query.filter(volume__icontains=filters['volume'])
            if filters.get('query'):
                query_without_tashkeel = Hadith.remove_tashkeel(filters['query'])
                query = query.filter(
                    Q(text__icontains=filters['query']) |
                    Q(text_without_tashkeel__icontains=query_without_tashkeel)
                )

        # Paginate results
        paginator = Paginator(query, per_page)
        page_obj = paginator.get_page(page)

        return paginator, page_obj, query.count()

    @staticmethod
    def format_hadith_dict(hadith):
        """
        Format a hadith object as a dictionary

        Args:
            hadith (Hadith): The hadith object

        Returns:
            dict: Formatted hadith dictionary
        """
        return {
            'id': hadith.hadith_id,
            'text': hadith.text,
            'text_without_tashkeel': hadith.text_without_tashkeel,
            'book_name': hadith.book_name,
            'page_number': hadith.page_number,
            'volume': hadith.volume,
            'exporter': hadith.exporter,
            'narrators': hadith.narrators,
            'formatted_source': hadith.get_formatted_source(),
            'narration_chain': hadith.get_narration_chain(),
            'url': hadith.url
        }
