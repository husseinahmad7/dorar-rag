"""
Utility functions for caching
"""
from django.core.cache import cache
from Hadith.models import Hadith

# Cache keys
BOOK_NAMES_CACHE_KEY = 'hadith_book_names'

def get_book_names():
    """
    Get all unique book names from the database or cache
    
    Returns:
        list: List of unique book names
    """
    # Try to get from cache first
    book_names = cache.get(BOOK_NAMES_CACHE_KEY)
    
    # If not in cache, fetch from database and cache it
    if book_names is None:
        book_names = list(Hadith.objects.values_list('book_name', flat=True).distinct().order_by('book_name'))
        cache.set(BOOK_NAMES_CACHE_KEY, book_names, timeout=None)  # No timeout - cache indefinitely
    
    return book_names

def refresh_book_names_cache():
    """
    Refresh the book names cache
    
    Returns:
        list: Updated list of unique book names
    """
    # Fetch from database and update cache
    book_names = list(Hadith.objects.values_list('book_name', flat=True).distinct().order_by('book_name'))
    cache.set(BOOK_NAMES_CACHE_KEY, book_names, timeout=None)
    return book_names
