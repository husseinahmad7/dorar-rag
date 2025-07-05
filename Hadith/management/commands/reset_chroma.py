# Hadith/management/commands/reset_chroma.py
from django.core.management.base import BaseCommand
import chromadb
import os
import time
import psutil
import sys
import sqlite3
import shutil

class Command(BaseCommand):
    help = 'Reset the ChromaDB collection'

    def add_arguments(self, parser):
        parser.add_argument('--confirm', action='store_true', help='Confirm deletion without prompting')

    def handle(self, *args, **options):
        confirm = options['confirm']

        if not confirm:
            self.stdout.write(self.style.WARNING('This will delete all embeddings in ChromaDB. Are you sure? (y/n)'))
            response = input()
            if response.lower() != 'y':
                self.stdout.write(self.style.SUCCESS('Operation cancelled.'))
                return

        # Path to ChromaDB data
        chroma_path = "./chroma_db"
        db_file = os.path.join(chroma_path, "chroma.sqlite3")

        # First approach: Try to use the API to delete the collection
        if os.path.exists(chroma_path):
            try:
                # Initialize ChromaDB client
                client = chromadb.PersistentClient(path=chroma_path)

                # Try to delete the collection
                try:
                    # Check if collection exists
                    collections = client.list_collections()
                    collection_exists = any(c.name == "hadith_collection" for c in collections)

                    if collection_exists:
                        client.delete_collection("hadith_collection")
                        self.stdout.write(self.style.SUCCESS(f'Successfully deleted ChromaDB collection'))
                    else:
                        self.stdout.write(self.style.WARNING(f'Collection "hadith_collection" does not exist'))
                except Exception as e:
                    self.stdout.write(self.style.WARNING(f'Error deleting collection via API: {e}'))
                    self.stdout.write(self.style.WARNING(f'Will try to delete the database file directly'))
                    # If API fails, we'll fall back to file deletion
                    raise e

                # Create a new collection
                collection = client.create_collection("hadith_collection")
                self.stdout.write(self.style.SUCCESS(f'Created new empty collection "hadith_collection"'))

                self.stdout.write(self.style.SUCCESS('ChromaDB reset complete. You can now run update_embeddings to rebuild the embeddings.'))
                return
            except Exception as e:
                self.stdout.write(self.style.WARNING(f'API-based reset failed: {e}'))
                self.stdout.write(self.style.WARNING(f'Falling back to file-based reset'))

        # Second approach: Delete and recreate the directory
        if os.path.exists(chroma_path):
            try:
                # Close any open connections
                self.stdout.write(self.style.WARNING(f'Attempting to close any open connections'))
                time.sleep(2)  # Give time for connections to close

                # Delete the directory
                import shutil
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        shutil.rmtree(chroma_path)
                        self.stdout.write(self.style.SUCCESS(f'Successfully deleted ChromaDB directory'))
                        break
                    except Exception as e:
                        if attempt < max_attempts - 1:
                            self.stdout.write(self.style.WARNING(f'Error deleting directory, waiting and retrying... ({attempt+1}/{max_attempts})'))
                            time.sleep(2)  # Wait before retrying
                        else:
                            # If we can't delete the directory, try to delete just the database file
                            if os.path.exists(db_file):
                                try:
                                    os.remove(db_file)
                                    self.stdout.write(self.style.SUCCESS(f'Successfully deleted ChromaDB database file'))
                                except Exception as file_e:
                                    self.stdout.write(self.style.ERROR(f'Error deleting database file: {file_e}'))
                                    raise e

                # Create a new directory
                os.makedirs(chroma_path, exist_ok=True)

                # Initialize a new ChromaDB client
                client = chromadb.PersistentClient(path=chroma_path)
                collection = client.create_collection("hadith_collection")
                self.stdout.write(self.style.SUCCESS(f'Created new empty collection "hadith_collection"'))

                self.stdout.write(self.style.SUCCESS('ChromaDB reset complete. You can now run update_embeddings to rebuild the embeddings.'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error resetting ChromaDB: {e}'))
        else:
            self.stdout.write(self.style.WARNING(f'ChromaDB directory {chroma_path} does not exist. Creating it now.'))
            try:
                # Create a new empty directory
                os.makedirs(chroma_path, exist_ok=True)

                # Initialize a new ChromaDB client
                client = chromadb.PersistentClient(path=chroma_path)
                collection = client.create_collection("hadith_collection")

                self.stdout.write(self.style.SUCCESS('ChromaDB directory and collection created successfully.'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error creating ChromaDB: {e}'))
