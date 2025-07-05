# Hadith/management/commands/force_reset_chroma.py
from django.core.management.base import BaseCommand
import os
import time
import psutil
import sys
import sqlite3
import shutil
import subprocess

class Command(BaseCommand):
    help = 'Force reset the ChromaDB by killing processes and recreating the database'

    def add_arguments(self, parser):
        parser.add_argument('--confirm', action='store_true', help='Confirm deletion without prompting')
        parser.add_argument('--kill', action='store_true', help='Kill processes that have the database open')

    def handle(self, **options):
        confirm = options['confirm']
        kill_processes = options['kill']
        
        if not confirm:
            self.stdout.write(self.style.WARNING('This will forcefully delete ChromaDB. Are you sure? (y/n)'))
            response = input()
            if response.lower() != 'y':
                self.stdout.write(self.style.SUCCESS('Operation cancelled.'))
                return
        
        # Path to ChromaDB data
        chroma_path = "./chroma_db"
        db_file = os.path.join(chroma_path, "chroma.sqlite3")
        
        # Find and optionally kill processes that have the database open
        if os.path.exists(db_file):
            self.stdout.write(self.style.WARNING(f'Checking processes that have {db_file} open'))
            
            # Find processes that have the file open
            for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                try:
                    open_files = proc.info.get('open_files', [])
                    if open_files:
                        for file in open_files:
                            if file and hasattr(file, 'path') and db_file in file.path:
                                self.stdout.write(self.style.WARNING(
                                    f'Process {proc.info["name"]} (PID: {proc.info["pid"]}) has the file open'
                                ))
                                
                                if kill_processes:
                                    self.stdout.write(self.style.WARNING(f'Killing process {proc.info["pid"]}'))
                                    try:
                                        proc.kill()
                                        self.stdout.write(self.style.SUCCESS(f'Process {proc.info["pid"]} killed'))
                                    except Exception as e:
                                        self.stdout.write(self.style.ERROR(f'Failed to kill process: {e}'))
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
            
            # Wait a moment for processes to terminate
            if kill_processes:
                self.stdout.write(self.style.WARNING('Waiting for processes to terminate...'))
                time.sleep(3)
        
        # Try to force close any SQLite connections
        try:
            # This is a Windows-specific approach to close file handles
            if sys.platform == 'win32' and os.path.exists(db_file):
                self.stdout.write(self.style.WARNING('Attempting to force close file handles (Windows only)'))
                
                # Use handle.exe from Sysinternals if available
                handle_exe = shutil.which('handle.exe') or shutil.which('handle')
                if handle_exe:
                    try:
                        # Find handles to the database file
                        output = subprocess.check_output([handle_exe, db_file], text=True)
                        self.stdout.write(self.style.WARNING(f'Handle output: {output}'))
                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f'Error running handle.exe: {e}'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error trying to force close connections: {e}'))
        
        # Try to delete the directory
        if os.path.exists(chroma_path):
            self.stdout.write(self.style.WARNING(f'Attempting to delete {chroma_path}'))
            
            try:
                # Try to delete the directory
                shutil.rmtree(chroma_path)
                self.stdout.write(self.style.SUCCESS(f'Successfully deleted {chroma_path}'))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Failed to delete directory: {e}'))
                
                # If directory deletion fails, try to delete just the database file
                if os.path.exists(db_file):
                    try:
                        self.stdout.write(self.style.WARNING(f'Attempting to delete just the database file {db_file}'))
                        os.remove(db_file)
                        self.stdout.write(self.style.SUCCESS(f'Successfully deleted {db_file}'))
                    except Exception as e2:
                        self.stdout.write(self.style.ERROR(f'Failed to delete database file: {e2}'))
                        
                        # Last resort: try to corrupt/zero out the database file
                        try:
                            self.stdout.write(self.style.WARNING('Attempting to zero out the database file'))
                            with open(db_file, 'wb') as f:
                                f.write(b'')  # Write empty content
                            self.stdout.write(self.style.SUCCESS('Database file zeroed out'))
                        except Exception as e3:
                            self.stdout.write(self.style.ERROR(f'Failed to zero out database file: {e3}'))
        
        # Create a new directory and initialize ChromaDB
        try:
            os.makedirs(chroma_path, exist_ok=True)
            self.stdout.write(self.style.SUCCESS(f'Created directory {chroma_path}'))
            
            # Initialize a new ChromaDB client
            client = chromadb.PersistentClient(path=chroma_path)
            collection = client.create_collection("hadith_collection")
            self.stdout.write(self.style.SUCCESS('Created new ChromaDB collection'))
            
            self.stdout.write(self.style.SUCCESS('ChromaDB reset complete. You can now run update_embeddings to rebuild the embeddings.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error creating new ChromaDB: {e}'))
