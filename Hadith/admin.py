from django.contrib import admin
from .models import Hadith

@admin.register(Hadith)
class HadithAdmin(admin.ModelAdmin):
    list_display = ('hadith_id', 'short_text', 'formatted_source', 'narration_chain', 'is_embedded')
    list_filter = ('book_name', 'is_embedded')
    search_fields = ('text', 'text_without_tashkeel', 'book_name', 'exporter')
    readonly_fields = ('text_without_tashkeel',)
    fieldsets = (
        (None, {
            'fields': ('hadith_id', 'text', 'text_without_tashkeel', 'is_embedded')
        }),
        ('Source Information', {
            'fields': ('book_name', 'volume', 'chapter', 'page_number', 'exporter', 'narrators')
        }),
        ('Metadata', {
            'fields': ('url',)
        }),
    )

    def short_text(self, obj):
        return obj.text[:50] + '...' if len(obj.text) > 50 else obj.text
    short_text.short_description = 'Text'

    def formatted_source(self, obj):
        return obj.get_formatted_source()
    formatted_source.short_description = 'Source'

    def narration_chain(self, obj):
        return obj.get_narration_chain()
    narration_chain.short_description = 'Narrators'
