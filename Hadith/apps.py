from django.apps import AppConfig


class HadithConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Hadith'

    def ready(self):
        """
        Initialize app when Django starts
        """
        # Import here to avoid circular imports
        from Hadith.utils.cache_utils import refresh_book_names_cache

        # Cache book names on startup
        try:
            refresh_book_names_cache()
            print("Book names cached successfully")
        except Exception as e:
            print(f"Error caching book names: {e}")
