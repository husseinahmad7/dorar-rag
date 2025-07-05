# Hadith/management/commands/reset_langchain_db.py
from django.core.management.base import BaseCommand
import os
import time
import shutil

class Command(BaseCommand):
    help = 'Reset the LangChain ChromaDB collection'

    def add_arguments(self, parser):
        parser.add_argument('--confirm', action='store_true', help='Confirm deletion without prompting')

    def handle(self, **options):
        confirm = options['confirm']
        
        if not confirm:
            self.stdout.write(self.style.WARNING('This will delete all LangChain embeddings. Are you sure? (y/n)'))
            response = input()
            if response.lower() != 'y':
                self.stdout.write(self.style.SUCCESS('Operation cancelled.'))
                return
        
        # Path to LangChain ChromaDB data
        chroma_path = "./langchain_db"
        
        if os.path.exists(chroma_path):
            try:
                # Close any open connections
                self.stdout.write(self.style.WARNING(f'Attempting to close any open connections'))
                time.sleep(2)  # Give time for connections to close
                
                # Delete the directory
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        shutil.rmtree(chroma_path)
                        self.stdout.write(self.style.SUCCESS(f'Successfully deleted LangChain ChromaDB directory'))
                        break
                    except Exception as e:
                        if attempt < max_attempts - 1:
                            self.stdout.write(self.style.WARNING(f'Error deleting directory, waiting and retrying... ({attempt+1}/{max_attempts})'))
                            time.sleep(2)  # Wait before retrying
                        else:
                            raise e
                
                # Create a new empty directory
                os.makedirs(chroma_path, exist_ok=True)
                self.stdout.write(self.style.SUCCESS(f'Created new empty directory at {chroma_path}'))
                
                self.stdout.write(self.style.SUCCESS('LangChain ChromaDB reset complete. You can now run populate_langchain_db to rebuild the embeddings.'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error resetting LangChain ChromaDB: {e}'))
        else:
            self.stdout.write(self.style.WARNING(f'LangChain ChromaDB directory {chroma_path} does not exist. Creating it now.'))
            try:
                # Create a new empty directory
                os.makedirs(chroma_path, exist_ok=True)
                self.stdout.write(self.style.SUCCESS('LangChain ChromaDB directory created successfully.'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error creating LangChain ChromaDB directory: {e}'))
