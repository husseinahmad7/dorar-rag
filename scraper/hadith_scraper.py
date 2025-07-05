import asyncio
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from environ import Env

import sys
import io

# Configure stdout for UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
env = Env()
env.read_env()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('hadith_scraper')

class HadithScraper:
    def __init__(self, max_workers=5, timeout=20, headless=True, verbose=False,
                 output_file="hadiths.json", batch_size=10, max_retries=3):
        """Initialize the scraper with configurable parameters

        Args:
            max_workers (int): Number of concurrent workers
            timeout (int): Timeout in seconds for page loading
            headless (bool): Whether to run browser in headless mode
            verbose (bool): Whether to enable verbose output
            output_file (str): Name of the output file (will be saved in data directory)
            batch_size (int): Number of hadiths to process before saving
            max_retries (int): Number of retries for failed requests
        """
        self.base_url = "https://hadith.inoor.ir/ar/hadith/"
        self.timeout = timeout
        self.headless = headless
        self.verbose = verbose
        self.output_file = output_file
        self.batch_size = batch_size
        self.max_retries = max_retries

        # Initialize collections
        self.scraped_ids = set()  # Keep track of scraped hadith IDs
        self.hadith_num_error_list = []
        self.driver_pool = []

        # Configure logging verbosity
        if not verbose:
            logging.getLogger('hadith_scraper').setLevel(logging.WARNING)

        # Setup concurrency
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Load previously scraped IDs
        self.scraped_ids = self.load_scraped_ids()

        # Initialize driver
        logger.info(f"Initializing scraper with base URL: {self.base_url}")
        self.driver = self.setup_driver()

    def load_scraped_ids(self) -> Set[int]:
        """Load previously scraped data to avoid duplicates

        Returns:
            Set[int]: Set of already scraped hadith IDs
        """
        try:
            data_dir = Path("data")
            if data_dir.exists():
                data_file = data_dir / self.output_file
                if data_file.exists():
                    logger.info(f"Loading existing data from {data_file}")
                    with open(data_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        scraped_ids = {h['id'] for h in existing_data}
                        logger.info(f"Loaded {len(scraped_ids)} existing hadith IDs")
                        return scraped_ids
                else:
                    logger.info(f"No existing data file found at {data_file}")
            else:
                logger.info("Data directory does not exist, will be created")
        except Exception as e:
            logger.error(f"Error loading existing data: {str(e)}")
        return set()

    def setup_driver(self):
        """Create and return a new WebDriver instance with optimized settings

        Returns:
            webdriver.Edge: Configured Edge WebDriver instance
        """
        try:
            logger.info("Setting up WebDriver")
            service = Service(EdgeChromiumDriverManager().install())
            options = Options()

            # Apply headless mode if configured
            if self.headless:
                logger.info("Running in headless mode")
                options.add_argument('--headless')

            # Performance optimizations
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--window-size=1920x1080')
            options.add_argument('--disable-extensions')
            options.add_argument('--disable-infobars')

            # Disable images to save bandwidth and improve performance
            prefs = {
                "profile.managed_default_content_settings.images": 2,
                "profile.default_content_setting_values.notifications": 2,
                "profile.managed_default_content_settings.stylesheets": 2,
                "profile.managed_default_content_settings.cookies": 1,
                "profile.managed_default_content_settings.javascript": 1,
                "profile.managed_default_content_settings.plugins": 1,
                "profile.managed_default_content_settings.popups": 2,
                "profile.managed_default_content_settings.geolocation": 2,
                "profile.managed_default_content_settings.media_stream": 2,
            }
            options.add_experimental_option("prefs", prefs)

            # Create and return the driver
            driver = webdriver.Edge(service=service, options=options)
            driver.set_page_load_timeout(self.timeout)
            logger.info("WebDriver setup complete")
            return driver
        except Exception as e:
            logger.error(f"Error setting up WebDriver: {str(e)}")
            raise

    def wait_for_element(self, driver, by, selector, timeout=None):
        """Wait for an element to be present and visible

        Args:
            driver: WebDriver instance
            by: Selenium By locator strategy
            selector: Element selector string
            timeout: Custom timeout in seconds (uses self.timeout if None)

        Returns:
            WebElement or None: The found element or None if not found
        """
        if timeout is None:
            timeout = self.timeout

        try:
            # Wait for Angular to finish loading
            try:
                WebDriverWait(driver, timeout).until(
                    lambda driver: driver.execute_script('return window.getAllAngularTestabilities().findIndex(x=>!x.isStable()) === -1')
                )
            except Exception as e:
                logger.warning(f"Could not verify Angular stability: {str(e)}")

            # Wait for element
            logger.debug(f"Waiting for element: {selector}")
            element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((by, selector))
            )

            # Reduced wait time for better performance
            time.sleep(1)
            return element
        except TimeoutException:
            logger.warning(f"Timeout waiting for element: {selector}")
            return None
        except Exception as e:
            logger.error(f"Error waiting for element {selector}: {str(e)}")
            return None

    async def scrape_single_hadith(self, hadith_id) -> Optional[Dict[str, Any]]:
        """Scrape a single hadith by ID with retry mechanism

        Args:
            hadith_id: The ID of the hadith to scrape

        Returns:
            Optional[Dict[str, Any]]: The scraped hadith data or None if failed
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Starting to scrape hadith ID: {hadith_id}")
        driver = None

        # Implement retry mechanism
        for attempt in range(1, self.max_retries + 1):
            try:
                url = f"{self.base_url}{hadith_id}"
                logger.info(f"Attempt {attempt}/{self.max_retries} - Accessing URL: {url}")

                def _scrape():
                    nonlocal driver
                    # Use driver pool for better resource management
                    if not self.driver_pool:
                        driver = self.setup_driver()
                    else:
                        driver = self.driver_pool.pop()

                    try:
                        logger.info(f"Navigating to page...")
                        driver.get(url)
                        logger.info(f"Page loaded, waiting for content...")
                        # Reduced wait time for better performance
                        time.sleep(2)

                        # Extract content
                        logger.info("Extracting content...")
                        content = self.extract_hadith_content(driver)
                        if not content:
                            logger.warning(f"Failed to extract content for hadith {hadith_id}")
                            return None

                        # Extract metadata
                        logger.info("Extracting metadata...")
                        metadata = self.extract_metadata(driver)
                        if not metadata:
                            logger.warning(f"Failed to extract metadata for hadith {hadith_id}")
                            return None

                        # Combine results
                        result = {
                            'id': hadith_id,
                            **content,
                            **metadata
                        }

                        if self.verbose:
                            logger.info(f"Successfully scraped hadith {hadith_id}")
                            logger.debug(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
                        return result
                    finally:
                        # Return driver to pool instead of quitting for reuse
                        if driver and len(self.driver_pool) < self.max_workers:
                            try:
                                # Clear cookies to prevent memory buildup
                                driver.delete_all_cookies()
                                self.driver_pool.append(driver)
                            except WebDriverException:
                                # If there's an issue with the driver, don't reuse it
                                try:
                                    driver.quit()
                                except:
                                    pass

                # Run in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self.executor, _scrape)

                if result:
                    return result

                # If we get here, scraping failed but no exception was raised
                # Wait before retrying
                if attempt < self.max_retries:
                    retry_delay = 2 * attempt  # Exponential backoff
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)

            except Exception as e:
                logger.error(f"Error scraping hadith {hadith_id} (attempt {attempt}/{self.max_retries}): {str(e)}")
                if attempt < self.max_retries:
                    retry_delay = 2 * attempt  # Exponential backoff
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    # All retries failed
                    self.hadith_num_error_list.append(hadith_id)
                    return None

        # If we get here, all retries failed
        self.hadith_num_error_list.append(hadith_id)
        return None

    def extract_hadith_content(self, driver) -> Optional[Dict[str, Any]]:
        """Extract the hadith content with proper structure

        Args:
            driver: WebDriver instance

        Returns:
            Optional[Dict[str, Any]]: Dictionary with narrators, exporter, and text or None if failed
        """
        try:
            logger.debug("Extracting hadith content...")
            # Wait for the main content container and find the hadith div
            content_container = self.wait_for_element(driver, By.CSS_SELECTOR, "div.hadith-content")
            if not content_container:
                logger.warning("Could not find main content container")
                if self.verbose:
                    logger.debug(f"Page source preview: {driver.page_source[:200]}...")
                return None

            # Find the document tag which contains narrators
            try:
                document = content_container.find_element(By.CSS_SELECTOR, "document")
                logger.debug("Found document container")
            except NoSuchElementException:
                logger.warning("Could not find document tag")
                return None

            # Extract narrators from the chain
            narrators = []
            try:
                narrator_elements = document.find_elements(By.CSS_SELECTOR, "narrator")
                for narrator in narrator_elements:
                    narrators.append(narrator.text.strip())
                logger.debug(f"Found {len(narrators)} narrators")
            except Exception as e:
                logger.warning(f"Error extracting narrators: {str(e)}")

            # Extract exporter
            try:
                exporter_element = document.find_element(By.CSS_SELECTOR, "exporter")
                exporter = exporter_element.text.strip() if exporter_element else None
                logger.debug(f"Found exporter: {exporter}")
            except NoSuchElementException:
                logger.warning("Could not find exporter")
                exporter = None

            # Extract the main hadith text
            try:
                hadith_element = content_container.find_element(By.CSS_SELECTOR, "hadith")
                # Get text excluding the document part
                text = hadith_element.text.replace(document.text, '').strip()
                if text:
                    if self.verbose:
                        logger.debug(f"Successfully extracted hadith text: {text[:50]}...")
                else:
                    logger.warning("Could not find hadith text")
                    return None
            except Exception as e:
                logger.error(f"Error extracting text: {str(e)}")
                return None

            return {
                'narrators': narrators,
                'exporter': exporter,
                'text': text
            }
        except Exception as e:
            logger.error(f"Error extracting hadith content: {str(e)}")
            return None

    def extract_metadata(self, driver) -> Optional[Dict[str, Any]]:
        """Extract metadata about the hadith

        Args:
            driver: WebDriver instance

        Returns:
            Optional[Dict[str, Any]]: Dictionary with hadith metadata or None if failed
        """
        try:
            logger.debug("Extracting metadata...")
            metadata = {}

            # Extract hadith number
            try:
                number_container = self.wait_for_element(
                    driver,
                    By.CSS_SELECTOR,
                    "span.hadith-title span"
                )
                if number_container:
                    metadata['hadith_number'] = number_container.text.strip()
                    logger.debug(f"Hadith number: {metadata['hadith_number']}")
                else:
                    logger.warning("Could not find hadith number")
                    metadata['hadith_number'] = None
            except Exception as e:
                logger.warning(f"Error extracting hadith number: {str(e)}")
                metadata['hadith_number'] = None

            # Extract book info, volume and page
            try:
                book_container = self.wait_for_element(
                    driver,
                    By.CSS_SELECTOR,
                    "span.hadith-title:nth-child(2) span"
                )
                if book_container:
                    book_text = book_container.text.strip()
                    parts = book_text.split(',')
                    if len(parts) >= 3:
                        metadata['book_name'] = parts[0].strip()
                        metadata['volume'] = parts[1].replace('الجزء', '').strip()
                        metadata['page'] = parts[2].replace('الصفحة', '').strip()
                        logger.debug(f"Book info: {book_text}")
                    else:
                        metadata['book_name'] = book_text
                        metadata['volume'] = None
                        metadata['page'] = None
                else:
                    logger.warning("Could not find book info")
                    metadata['book_name'] = None
                    metadata['volume'] = None
                    metadata['page'] = None
            except Exception as e:
                logger.warning(f"Error extracting book info: {str(e)}")
                metadata['book_name'] = None
                metadata['volume'] = None
                metadata['page'] = None

            # Extract chapter
            try:
                chapter_container = self.wait_for_element(
                    driver,
                    By.CSS_SELECTOR,
                    "span.hadith-title span.toc-list"
                )
                if chapter_container:
                    metadata['chapter'] = chapter_container.text.strip()
                    logger.debug(f"Chapter: {metadata['chapter']}")
                else:
                    logger.warning("Could not find chapter")
                    metadata['chapter'] = None
            except Exception as e:
                logger.warning(f"Error extracting chapter: {str(e)}")
                metadata['chapter'] = None

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return None

    async def scrape_hadiths(self, start_id=103963, num_hadiths=30):
        """Scrape multiple hadiths concurrently with batch processing

        Args:
            start_id: Starting hadith ID
            num_hadiths: Number of hadiths to scrape
        """
        logger.info(f"Starting to scrape {num_hadiths} hadiths from ID {start_id}")

        # Track statistics
        stats = {
            'total': num_hadiths,
            'skipped': 0,
            'success': 0,
            'failed': 0,
            'start_time': time.time()
        }

        tasks = []
        for hadith_id in range(start_id, start_id + num_hadiths):
            # Skip already scraped hadiths
            if hadith_id in self.scraped_ids:
                logger.debug(f"Skipping already scraped hadith ID: {hadith_id}")
                stats['skipped'] += 1
                continue

            # Add task to the batch
            task = self.scrape_single_hadith(hadith_id)
            tasks.append(task)

            # Process in batches to avoid overwhelming the server
            if len(tasks) >= self.batch_size:
                results = await asyncio.gather(*tasks)
                valid_results = [r for r in results if r]

                # Update statistics
                stats['success'] += len(valid_results)
                stats['failed'] += len(results) - len(valid_results)

                # Save batch results
                if valid_results:
                    self.save_results(valid_results)

                # Log progress
                elapsed = time.time() - stats['start_time']
                progress = (stats['success'] + stats['failed'] + stats['skipped']) / stats['total'] * 100
                logger.info(f"Progress: {progress:.1f}% | Success: {stats['success']} | Failed: {stats['failed']} | Skipped: {stats['skipped']} | Elapsed: {elapsed:.1f}s")

                # Clear tasks for next batch
                tasks = []

        # Process remaining tasks
        if tasks:
            results = await asyncio.gather(*tasks)
            valid_results = [r for r in results if r]

            # Update statistics
            stats['success'] += len(valid_results)
            stats['failed'] += len(results) - len(valid_results)

            # Save remaining results
            if valid_results:
                self.save_results(valid_results)

        # Log final statistics
        elapsed = time.time() - stats['start_time']
        logger.info(f"Scraping completed in {elapsed:.1f} seconds")
        logger.info(f"Total: {stats['total']} | Success: {stats['success']} | Failed: {stats['failed']} | Skipped: {stats['skipped']}")
        if stats['failed'] > 0:
            logger.info(f"Failed hadith IDs: {self.hadith_num_error_list}")

    def save_results(self, results: List[Dict[str, Any]]):
        """Save scraped results to file with optimized I/O

        Args:
            results: List of hadith dictionaries to save
        """
        if not results:
            return

        try:
            data_dir = Path("data")
            data_dir.mkdir(exist_ok=True)

            # Load existing data
            data_file = data_dir / self.output_file
            existing_data = []

            if data_file.exists():
                logger.info(f"Loading existing data from {data_file}")
                try:
                    with open(data_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Error decoding JSON from {data_file}, creating new file")
            else:
                logger.info(f"No existing data file found at {data_file}, creating new file")

            # Update scraped IDs set
            for result in results:
                self.scraped_ids.add(result['id'])

            # Append new results
            existing_data.extend(results)

            # Save updated data with a temporary file to prevent data loss
            temp_file = data_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

            # Rename temp file to target file (atomic operation)
            temp_file.replace(data_file)

            logger.info(f"Saved {len(results)} new hadiths to {data_file}")

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    async def cleanup(self):
        """Clean up resources and save error list

        This method should be called when scraping is complete
        """
        logger.info("Cleaning up resources...")

        # Save error list to JSON file
        if self.hadith_num_error_list:
            try:
                error_file = Path("data").joinpath(f"errors_{self.output_file}")
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(self.hadith_num_error_list, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(self.hadith_num_error_list)} hadith errors to {error_file}")
            except Exception as e:
                logger.error(f"Error saving hadith errors: {str(e)}")

        # Clean up driver pool
        for driver in self.driver_pool:
            try:
                driver.quit()
            except:
                pass

        # Clean up main driver
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
            except:
                pass

        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        logger.info("Cleanup complete")

    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        # Clean up driver pool
        for driver in self.driver_pool:
            try:
                driver.quit()
            except:
                pass

        # Clean up main driver
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.quit()
            except:
                pass

        # Shutdown thread pool
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

async def main():
    """Main function for direct script execution

    This is only used when the script is run directly, not when imported.
    For command-line usage, use the scraper_command.py script instead.
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Hadith Scraper")
    parser.add_argument("--output", "-o", type=str, default="hadiths.json", help="Output file name")
    parser.add_argument("--start-id", "-s", type=int, default=50000, help="Starting hadith ID")
    parser.add_argument("--count", "-c", type=int, default=10, help="Number of hadiths to scrape")
    parser.add_argument("--workers", "-w", type=int, default=3, help="Number of concurrent workers")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Configure logging level based on verbosity
    if args.verbose:
        logging.getLogger('hadith_scraper').setLevel(logging.DEBUG)

    logger.info(f"Starting hadith scraper with output file: {args.output}")

    # Create and run scraper
    scraper = HadithScraper(
        max_workers=args.workers,
        output_file=args.output,
        verbose=args.verbose
    )

    try:
        await scraper.scrape_hadiths(start_id=args.start_id, num_hadiths=args.count)
    except KeyboardInterrupt:
        logger.info("Scraping interrupted by user")
    finally:
        await scraper.cleanup()

if __name__ == "__main__":
    asyncio.run(main())