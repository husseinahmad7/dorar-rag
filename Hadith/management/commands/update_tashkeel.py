# Hadith/management/commands/update_tashkeel.py
from django.core.management.base import BaseCommand
from Hadith.models import Hadith
import time

class Command(BaseCommand):
    help = 'Update existing hadiths with text_without_tashkeel field'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int, default=100, help='Batch size for processing')

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        
        # Get all hadiths that need updating
        hadiths_to_update = Hadith.objects.filter(text_without_tashkeel__isnull=True)
        total = hadiths_to_update.count()
        
        if total == 0:
            self.stdout.write(self.style.SUCCESS('No hadiths need updating.'))
            return
            
        self.stdout.write(self.style.SUCCESS(f'Found {total} hadiths that need updating.'))
        
        # Process in batches
        processed = 0
        start_time = time.time()
        
        for i in range(0, total, batch_size):
            batch = hadiths_to_update[i:i+batch_size]
            self.process_batch(batch)
            processed += len(batch)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            
            self.stdout.write(self.style.SUCCESS(
                f'Processed {processed}/{total} hadiths ({processed/total*100:.1f}%) - '
                f'Rate: {rate:.1f} hadiths/sec - ETA: {eta/60:.1f} minutes'
            ))
            
        self.stdout.write(self.style.SUCCESS(f'Successfully updated {processed} hadiths'))
    
    def process_batch(self, batch):
        for hadith in batch:
            try:
                # Generate text without tashkeel
                hadith.text_without_tashkeel = Hadith.remove_tashkeel(hadith.text)
                hadith.save(update_fields=['text_without_tashkeel'])
                
                self.stdout.write(self.style.SUCCESS(f'Updated hadith {hadith.hadith_id}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error updating hadith {hadith.hadith_id}: {e}'))
