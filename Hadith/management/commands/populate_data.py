# hadith_app/management/commands/load_hadiths.py
from django.core.management.base import BaseCommand
import json
import os
import time
import logging
from glob import glob
from Hadith.models import Hadith

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Load hadiths from JSON file(s) and create embeddings (optional)'

    def add_arguments(self, parser):
        parser.add_argument(
            'path',
            type=str,
            help='Path to a JSON file or directory containing hadith JSON files'
        )
        parser.add_argument('--skip-embeddings', action='store_true', help='Skip generating embeddings')
        parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    def handle(self, **options):
        path = options['path']
        skip_embeddings = options['skip_embeddings']
        batch_size = options['batch_size']
        verbose = options['verbose']

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # Collect JSON files
        json_files = []
        if os.path.isdir(path):
            json_files = sorted(glob(os.path.join(path, "hadiths-*-true.json")))
        elif os.path.isfile(path):
            json_files = [path]
        else:
            self.stdout.write(self.style.ERROR(f"Invalid path: {path}"))
            return

        if not json_files:
            self.stdout.write(self.style.WARNING(f"No JSON files found in {path}"))
            return

        self.stdout.write(self.style.SUCCESS(f"Found {len(json_files)} JSON files to process."))

        # Loop over files
        for idx, json_file in enumerate(json_files, start=1):
            self.stdout.write(self.style.SUCCESS(f"[{idx}/{len(json_files)}] Processing {json_file}..."))
            self.process_file(json_file, batch_size, skip_embeddings)

    def process_file(self, json_file, batch_size, skip_embeddings):
        with open(json_file, 'r', encoding='utf-8') as f:
            hadiths = json.load(f)

        total = len(hadiths)
        processed = 0
        start_time = time.time()

        for i in range(0, total, batch_size):
            batch = hadiths[i:i+batch_size]
            self.process_batch(batch, skip_embeddings)
            processed += len(batch)

            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0

            self.stdout.write(self.style.SUCCESS(
                f'Processed {processed}/{total} hadiths ({processed/total*100:.1f}%) - '
                f'Rate: {rate:.1f}/sec - ETA: {eta/60:.1f} minutes'
            ))

    def process_batch(self, batch, skip_embeddings):
        """Process a batch of hadiths (embedding logic omitted for brevity)."""
        for hadith_data in batch:
            try:
                hadith_id = hadith_data['id']
                text = hadith_data['text']
                text_without_tashkeel = Hadith.remove_tashkeel(text)

                hadith, created = Hadith.objects.update_or_create(
                    hadith_id=hadith_id,
                    defaults={
                        'text': text,
                        'text_without_tashkeel': text_without_tashkeel,
                        'book_name': hadith_data.get('book_name', ''),
                        'volume': hadith_data.get('volume', ''),
                        'chapter': hadith_data.get('chapter', ''),
                        'page_number': hadith_data.get('page', ''),
                        'exporter': hadith_data.get('exporter', ''),
                        'narrators': hadith_data.get('narrators', []),
                        'url': hadith_data.get('url', f"https://hadith.inoor.ir/ar/hadith/{hadith_id}"),
                        'is_embedded': False
                    }
                )

                # TODO: Add embeddings logic here if needed (like your original)

                action = 'Created' if created else 'Updated'
                self.stdout.write(self.style.SUCCESS(f'{action} hadith {hadith_id}'))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error processing hadith {hadith_data.get("id")}: {e}'))
