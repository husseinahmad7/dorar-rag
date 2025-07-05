from django.db import models
import re
from django.db.models import JSONField

class Hadith(models.Model):
    hadith_id = models.IntegerField(unique=True)
    text = models.TextField()
    text_without_tashkeel = models.TextField(null=True, blank=True)  # Text without diacritics
    book_name = models.CharField(max_length=255)
    volume = models.CharField(max_length=50, null=True, blank=True)
    chapter = models.CharField(max_length=255, null=True, blank=True)  # Chapter field
    page_number = models.CharField(max_length=50, null=True, blank=True)
    exporter = models.CharField(max_length=255, null=True, blank=True)  # The speaker/final narrator
    narrators = JSONField(null=True, blank=True)  # Store list of narrators as JSON
    url = models.URLField()
    is_embedded = models.BooleanField(default=False)  # Track if hadith has been embedded in ChromaDB
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        # Remove tashkeel (diacritics) before saving
        if self.text and not self.text_without_tashkeel:
            self.text_without_tashkeel = self.remove_tashkeel(self.text)
        super().save(*args, **kwargs)

    @staticmethod
    def remove_tashkeel(text):
        """Remove Arabic diacritics (tashkeel) from text"""
        # Arabic diacritics Unicode ranges
        tashkeel = re.compile(r'[ً-ْٰ]')
        return re.sub(tashkeel, '', text)

    def __str__(self):
        return f"Hadith {self.hadith_id}: {self.text[:100]}..."

    class Meta:
        indexes = [
            models.Index(fields=['hadith_id']),
            models.Index(fields=['book_name']),
            models.Index(fields=['is_embedded']),
        ]

    def get_formatted_source(self):
        """Return formatted source string: book_name, volume, page_number, chapter"""
        source_parts = []
        if self.book_name:
            source_parts.append(self.book_name)
        if self.volume:
            source_parts.append(f"الجزء {self.volume}")
        if self.page_number:
            source_parts.append(f"الصفحة {self.page_number}")
        if self.chapter:
            source_parts.append(f"الباب: {self.chapter}")
        return "، ".join(source_parts)

    def get_narration_chain(self):
        """Return formatted narration chain with narrators separated by 'عن'"""
        if not self.narrators:
            # If no narrators but exporter is available, return 'قال {exporter}'
            if self.exporter:
                return f"قال {self.exporter}"
            return ""

        # Join narrators with 'عن'
        narrators_chain = " عن ".join(self.narrators)

        # Add exporter if available
        if self.exporter:
            return f"{narrators_chain} قال {self.exporter}"
        return narrators_chain
