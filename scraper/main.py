import asyncio
from hadith_scraper import HadithScraper
import time
from requests.exceptions import ConnectionError
import sys

async def main():
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            # Create scraper with 5 concurrent workers
            scraper = HadithScraper(max_workers=5)
            
            try:
                # Scrape 100 hadiths starting from ID 103902
                await scraper.scrape_hadiths(start_id=49362, num_hadiths=100_000)
            finally:
                del scraper
            break
        except ConnectionError as e:
            if attempt < max_retries - 1:
                print(f"Connection error: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to connect after {max_retries} attempts. Please check your internet connection.")
                sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
