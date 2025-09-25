"""
Django management command for updating embeddings to use gemini-embedding-001 model.

This command re-generates embeddings for all hadiths using the new gemini-embedding-001
model and updates the ChromaDB collection with proper batch processing, progress reporting,
and error handling.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.conf import settings
from django.utils import timezone

from Hadith.models import Hadith
from Hadith.services.vector_store import VectorStoreService


class Command(BaseCommand):
    """
    Management command for updating embeddings to gemini-embedding-001.
    
    This command provides comprehensive functionality for migrating embeddings
    to the new model with batch processing, progress tracking, and error recovery.
    """
    
    help = 'Update embeddings to use gemini-embedding-001 model with batch processing and progress tracking'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_service = None
        self.start_time = None
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0
        self.batch_errors = []
        
        # Rate limiting settings
        self.requests_per_minute = 60  # Conservative rate limit for Gemini API
        self.request_interval = 60.0 / self.requests_per_minute  # Seconds between requests
        self.last_request_time = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def add_arguments(self, parser):
        """Add command line arguments."""
        parser.add_argument(
            '--batch-size',
            type=int,
            default=50,
            help='Number of hadiths to process in each batch (default: 50)'
        )
        
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Perform a dry run to estimate time and cost without making changes'
        )
        
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force re-embedding of already embedded hadiths'
        )
        
        parser.add_argument(
            '--start-from',
            type=int,
            help='Start processing from a specific hadith ID (for resuming)'
        )
        
        parser.add_argument(
            '--limit',
            type=int,
            help='Limit the number of hadiths to process'
        )
        
        parser.add_argument(
            '--book-name',
            type=str,
            help='Process only hadiths from a specific book'
        )
        
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Enable verbose output with detailed progress information'
        )
        
        parser.add_argument(
            '--rate-limit',
            type=int,
            default=60,
            help='Requests per minute for API rate limiting (default: 60)'
        )
        
        parser.add_argument(
            '--retry-failed',
            action='store_true',
            help='Retry processing hadiths that previously failed'
        )
        
        parser.add_argument(
            '--backup-collection',
            action='store_true',
            help='Create a backup of the existing collection before updating'
        )

    def handle(self, *args, **options):
        """Main command handler."""
        self.start_time = timezone.now()
        
        # Set up rate limiting
        self.requests_per_minute = options['rate_limit']
        self.request_interval = 60.0 / self.requests_per_minute
        
        try:
            # Initialize vector store service
            self._initialize_vector_service()
            
            # Validate options
            self._validate_options(options)
            
            # Get hadiths to process
            hadiths_queryset = self._get_hadiths_queryset(options)
            total_count = hadiths_queryset.count()
            
            if total_count == 0:
                self.stdout.write(
                    self.style.WARNING('No hadiths found matching the criteria.')
                )
                return
            
            # Display initial information
            self._display_initial_info(options, total_count)
            
            # Perform dry run if requested
            if options['dry_run']:
                self._perform_dry_run(total_count, options)
                return
            
            # Create backup if requested
            if options['backup_collection']:
                self._create_collection_backup()
            
            # Process hadiths
            self._process_hadiths(hadiths_queryset, options)
            
            # Display final summary
            self._display_final_summary()
            
        except KeyboardInterrupt:
            self.stdout.write(
                self.style.WARNING('\nOperation interrupted by user.')
            )
            self._display_final_summary()
        except Exception as e:
            self.logger.error(f"Command failed with error: {e}")
            raise CommandError(f"Command failed: {e}")

    def _initialize_vector_service(self):
        """Initialize the vector store service."""
        self.vector_service = VectorStoreService()
        
        if not self.vector_service.is_available:
            raise CommandError(
                "Vector store service is not available. "
                "Please check your GEMINI_API_KEY and other settings."
            )
        
        self.stdout.write(
            self.style.SUCCESS(
                f"Vector store service initialized with model: {self.vector_service.embedding_model}"
            )
        )

    def _validate_options(self, options):
        """Validate command options."""
        if options['batch_size'] <= 0:
            raise CommandError("Batch size must be greater than 0")
        
        if options['rate_limit'] <= 0:
            raise CommandError("Rate limit must be greater than 0")
        
        if options['start_from'] and options['start_from'] < 0:
            raise CommandError("Start-from value must be non-negative")
        
        if options['limit'] and options['limit'] <= 0:
            raise CommandError("Limit must be greater than 0")

    def _get_hadiths_queryset(self, options):
        """Get the queryset of hadiths to process based on options."""
        queryset = Hadith.objects.all()
        
        # Filter by book name if specified
        if options['book_name']:
            queryset = queryset.filter(book_name__icontains=options['book_name'])
        
        # Filter by embedding status
        if not options['force'] and not options['retry_failed']:
            # Only process hadiths that haven't been embedded yet
            queryset = queryset.filter(is_embedded=False)
        elif options['retry_failed']:
            # Process hadiths that failed previously (you might want to add a failed_embedding field)
            # For now, we'll process non-embedded ones
            queryset = queryset.filter(is_embedded=False)
        
        # Apply start-from filter
        if options['start_from']:
            queryset = queryset.filter(hadith_id__gte=options['start_from'])
        
        # Apply limit
        if options['limit']:
            queryset = queryset[:options['limit']]
        
        # Order by hadith_id for consistent processing
        return queryset.order_by('hadith_id')

    def _display_initial_info(self, options, total_count):
        """Display initial information about the operation."""
        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(self.style.SUCCESS("Gemini Embeddings Update Command"))
        self.stdout.write(self.style.SUCCESS("=" * 60))
        
        self.stdout.write(f"Total hadiths to process: {total_count:,}")
        self.stdout.write(f"Batch size: {options['batch_size']}")
        self.stdout.write(f"Rate limit: {options['rate_limit']} requests/minute")
        self.stdout.write(f"Embedding model: {self.vector_service.embedding_model}")
        
        if options['force']:
            self.stdout.write(self.style.WARNING("Force mode: Re-embedding already embedded hadiths"))
        
        if options['book_name']:
            self.stdout.write(f"Book filter: {options['book_name']}")
        
        if options['start_from']:
            self.stdout.write(f"Starting from hadith ID: {options['start_from']}")
        
        if options['limit']:
            self.stdout.write(f"Processing limit: {options['limit']} hadiths")
        
        self.stdout.write(self.style.SUCCESS("-" * 60))

    def _perform_dry_run(self, total_count, options):
        """Perform a dry run to estimate time and cost."""
        self.stdout.write(self.style.WARNING("DRY RUN MODE - No changes will be made"))
        self.stdout.write(self.style.SUCCESS("=" * 50))
        
        # Estimate processing time
        estimated_time_minutes = (total_count * self.request_interval) / 60
        estimated_time_hours = estimated_time_minutes / 60
        
        # Estimate API costs (approximate - you may need to adjust based on actual pricing)
        # Gemini embedding costs are typically very low, but this gives an idea
        estimated_requests = total_count
        estimated_cost_usd = estimated_requests * 0.0001  # Very rough estimate
        
        self.stdout.write(f"Estimated processing time: {estimated_time_minutes:.1f} minutes ({estimated_time_hours:.1f} hours)")
        self.stdout.write(f"Estimated API requests: {estimated_requests:,}")
        self.stdout.write(f"Estimated cost (rough): ${estimated_cost_usd:.4f} USD")
        self.stdout.write(f"Rate limit: {self.requests_per_minute} requests/minute")
        
        # Sample a few hadiths to test embedding generation
        sample_hadiths = Hadith.objects.all()[:3]
        self.stdout.write(f"\nTesting embedding generation with {len(sample_hadiths)} sample hadiths...")
        
        for hadith in sample_hadiths:
            try:
                start_time = time.time()
                embedding = self.vector_service.generate_embedding(
                    hadith.text, 
                    use_without_tashkeel=True, 
                    is_query=False
                )
                end_time = time.time()
                
                self.stdout.write(
                    f"✓ Hadith {hadith.hadith_id}: {len(embedding)} dimensions, "
                    f"{end_time - start_time:.2f}s"
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"✗ Hadith {hadith.hadith_id}: Error - {e}")
                )
        
        self.stdout.write(self.style.SUCCESS("\nDry run completed successfully!"))

    def _create_collection_backup(self):
        """Create a backup of the existing ChromaDB collection."""
        self.stdout.write("Creating backup of existing collection...")
        
        try:
            # Get the current collection
            collection = self.vector_service._get_or_create_collection()
            
            # Create backup collection name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{self.vector_service.COLLECTION_NAME}_backup_{timestamp}"
            
            # Get all documents from the current collection
            all_results = collection.get(include=['documents', 'metadatas', 'embeddings'])
            
            if all_results['ids']:
                # Create backup collection
                backup_collection = self.vector_service.chroma_client.create_collection(
                    name=backup_name,
                    metadata={
                        "description": f"Backup of {self.vector_service.COLLECTION_NAME} created on {timestamp}",
                        "original_collection": self.vector_service.COLLECTION_NAME,
                        "backup_date": timestamp
                    }
                )
                
                # Add all documents to backup collection
                backup_collection.add(
                    ids=all_results['ids'],
                    documents=all_results['documents'],
                    metadatas=all_results['metadatas'],
                    embeddings=all_results['embeddings']
                )
                
                self.stdout.write(
                    self.style.SUCCESS(f"Backup created: {backup_name} with {len(all_results['ids'])} documents")
                )
            else:
                self.stdout.write(
                    self.style.WARNING("No documents found in collection to backup")
                )
                
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Failed to create backup: {e}")
            )
            raise CommandError("Backup creation failed")

    def _process_hadiths(self, hadiths_queryset, options):
        """Process hadiths in batches with progress tracking."""
        total_count = hadiths_queryset.count()
        batch_size = options['batch_size']
        verbose = options['verbose']
        
        self.stdout.write(f"\nStarting processing of {total_count:,} hadiths...")
        self.stdout.write(self.style.SUCCESS("-" * 60))
        
        # Process in batches
        for batch_start in range(0, total_count, batch_size):
            batch_end = min(batch_start + batch_size, total_count)
            batch_hadiths = hadiths_queryset[batch_start:batch_end]
            
            self._process_batch(batch_hadiths, batch_start + 1, batch_end, total_count, verbose)
            
            # Display progress
            if not verbose:
                self._display_progress(batch_end, total_count)

    def _process_batch(self, batch_hadiths, start_num, end_num, total_count, verbose):
        """Process a single batch of hadiths."""
        if verbose:
            self.stdout.write(f"\nProcessing batch {start_num}-{end_num} of {total_count}")
        
        for hadith in batch_hadiths:
            try:
                self._process_single_hadith(hadith, verbose)
                self.processed_count += 1
                
            except Exception as e:
                self.error_count += 1
                error_msg = f"Error processing hadith {hadith.hadith_id}: {e}"
                self.batch_errors.append(error_msg)
                
                if verbose:
                    self.stdout.write(self.style.ERROR(f"✗ {error_msg}"))
                
                self.logger.error(error_msg)
                
                # Continue processing other hadiths in the batch
                continue

    def _process_single_hadith(self, hadith, verbose):
        """Process a single hadith with rate limiting."""
        # Apply rate limiting
        self._apply_rate_limiting()
        
        # Skip if already embedded and not forcing
        if hadith.is_embedded and not hasattr(self, '_force_mode'):
            self.skipped_count += 1
            if verbose:
                self.stdout.write(f"⏭ Skipping hadith {hadith.hadith_id} (already embedded)")
            return
        
        # Prepare metadata
        metadata = self._prepare_hadith_metadata(hadith)
        
        # Generate and store embedding
        success = self.vector_service.store_embedding(
            hadith_id=hadith.hadith_id,
            text=hadith.text,
            metadata=metadata
        )
        
        if success:
            if verbose:
                self.stdout.write(f"✓ Processed hadith {hadith.hadith_id}")
        else:
            raise Exception("Failed to store embedding")

    def _apply_rate_limiting(self):
        """Apply rate limiting to API requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_interval:
            sleep_time = self.request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _prepare_hadith_metadata(self, hadith):
        """Prepare metadata dictionary for a hadith."""
        return {
            'id': str(hadith.hadith_id),
            'book_name': hadith.book_name or '',
            'volume': hadith.volume or '',
            'page_number': hadith.page_number or '',
            'chapter': hadith.chapter or '',
            'exporter': hadith.exporter or '',
            'narrators': hadith.narrators or [],
            'url': hadith.url or '',
        }

    def _display_progress(self, current, total):
        """Display progress bar and statistics."""
        percentage = (current / total) * 100
        bar_length = 50
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Calculate ETA
        elapsed_time = timezone.now() - self.start_time
        if current > 0:
            avg_time_per_item = elapsed_time.total_seconds() / current
            remaining_items = total - current
            eta_seconds = avg_time_per_item * remaining_items
            eta = timedelta(seconds=int(eta_seconds))
        else:
            eta = timedelta(0)
        
        # Display progress
        self.stdout.write(
            f'\rProgress: |{bar}| {current}/{total} ({percentage:.1f}%) '
            f'Processed: {self.processed_count}, Errors: {self.error_count}, '
            f'Skipped: {self.skipped_count}, ETA: {eta}',
            ending=''
        )
        self.stdout.flush()

    def _display_final_summary(self):
        """Display final summary of the operation."""
        end_time = timezone.now()
        total_time = end_time - self.start_time if self.start_time else timedelta(0)
        
        self.stdout.write("\n")
        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(self.style.SUCCESS("OPERATION COMPLETED"))
        self.stdout.write(self.style.SUCCESS("=" * 60))
        
        self.stdout.write(f"Total processing time: {total_time}")
        self.stdout.write(f"Successfully processed: {self.processed_count:,}")
        self.stdout.write(f"Errors encountered: {self.error_count:,}")
        self.stdout.write(f"Skipped (already embedded): {self.skipped_count:,}")
        
        if self.processed_count > 0:
            avg_time = total_time.total_seconds() / self.processed_count
            self.stdout.write(f"Average time per hadith: {avg_time:.2f} seconds")
        
        # Display errors if any
        if self.batch_errors:
            self.stdout.write(self.style.WARNING(f"\nErrors encountered ({len(self.batch_errors)}):"))
            for error in self.batch_errors[-10]:  # Show last 10 errors
                self.stdout.write(self.style.ERROR(f"  • {error}"))
            
            if len(self.batch_errors) > 10:
                self.stdout.write(f"  ... and {len(self.batch_errors) - 10} more errors")
        
        # Success rate
        total_attempted = self.processed_count + self.error_count
        if total_attempted > 0:
            success_rate = (self.processed_count / total_attempted) * 100
            self.stdout.write(f"Success rate: {success_rate:.1f}%")
        
        self.stdout.write(self.style.SUCCESS("-" * 60))
        
        # Recommendations
        if self.error_count > 0:
            self.stdout.write(
                self.style.WARNING(
                    "Some hadiths failed to process. You can retry with --retry-failed option."
                )
            )
        
        if self.processed_count > 0:
            self.stdout.write(
                self.style.SUCCESS(
                    "Embeddings have been successfully updated! "
                    "You can now use the new gemini-embedding-001 model for searches."
                )
            )