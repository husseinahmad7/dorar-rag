import os
import sys
import subprocess
import time

def run_command(command):
    """Run a command and return its output"""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"Error: {stderr}")
        return False

    print(stdout)
    return True

def setup():
    """Setup the project"""
    # Make migrations
    if not run_command("python manage.py makemigrations"):
        return False

    # Apply migrations
    if not run_command("python manage.py migrate"):
        return False

    # Load hadiths from JSON file
    if os.path.exists("data/hadiths.json"):
        if not run_command("python manage.py load_hadiths data/hadiths.json --skip-embeddings"):
            return False
    else:
        print("Warning: data/hadiths.json not found. Skipping data loading.")

    # Update text_without_tashkeel for existing records
    if not run_command("python manage.py update_tashkeel"):
        return False

    print("\nSetup completed successfully!")
    return True

if __name__ == "__main__":
    start_time = time.time()
    success = setup()
    elapsed = time.time() - start_time

    if success:
        print(f"Setup completed in {elapsed:.2f} seconds")
        sys.exit(0)
    else:
        print(f"Setup failed after {elapsed:.2f} seconds")
        sys.exit(1)
