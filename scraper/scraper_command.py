#!/usr/bin/env python
"""
Hadith Scraper Command Line Interface

This script provides a command-line interface for the hadith scraper.
It allows configuring various parameters for the scraping process.

Usage:
    python scraper_command.py --output hadiths.json --start-id 50000 --count 100
"""

import argparse
import asyncio
import sys
import os
import logging
from pathlib import Path

# Import the scraper
from hadith_scraper import HadithScraper, logger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Hadith Scraper - Efficiently scrape hadiths from hadith.inoor.ir",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="hadiths.json",
        help="Output file name (will be saved in the data directory)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--start-id", "-s", 
        type=int, 
        default=50000,
        help="Starting hadith ID to scrape"
    )
    
    parser.add_argument(
        "--count", "-c", 
        type=int, 
        default=10,
        help="Number of hadiths to scrape"
    )
    
    parser.add_argument(
        "--workers", "-w", 
        type=int, 
        default=3,
        help="Number of concurrent workers (be careful with high values)"
    )
    
    parser.add_argument(
        "--batch-size", "-b", 
        type=int, 
        default=5,
        help="Batch size for processing and saving results"
    )
    
    parser.add_argument(
        "--timeout", "-t", 
        type=int, 
        default=30,
        help="Timeout in seconds for page loading"
    )
    
    parser.add_argument(
        "--retries", "-r", 
        type=int, 
        default=3,
        help="Number of retries for failed requests"
    )
    
    parser.add_argument(
        "--headless", 
        action="store_true",
        default=True,
        help="Run browser in headless mode"
    )
    
    parser.add_argument(
        "--no-headless", 
        dest="headless",
        action="store_false",
        help="Run browser in visible mode (not headless)"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="scraper.log",
        help="Log file path"
    )
    
    return parser.parse_args()

async def main():
    """Main entry point for the scraper CLI"""
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log_file, encoding='utf-8')
        ]
    )
    
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Construct output file path
    output_file = args.output
    
    # Print configuration
    logger.info(f"Hadith Scraper Configuration:")
    logger.info(f"  - Starting ID: {args.start_id}")
    logger.info(f"  - Count: {args.count}")
    logger.info(f"  - Workers: {args.workers}")
    logger.info(f"  - Batch Size: {args.batch_size}")
    logger.info(f"  - Output File: {output_file}")
    logger.info(f"  - Headless Mode: {'Enabled' if args.headless else 'Disabled'}")
    logger.info(f"  - Timeout: {args.timeout} seconds")
    logger.info(f"  - Retries: {args.retries}")
    logger.info(f"  - Verbose: {'Enabled' if args.verbose else 'Disabled'}")
    logger.info(f"  - Log File: {args.log_file}")
    logger.info("\nStarting scraper...\n")
    
    # Initialize and run the scraper
    scraper = HadithScraper(
        max_workers=args.workers,
        timeout=args.timeout,
        headless=args.headless,
        verbose=args.verbose,
        output_file=output_file,
        batch_size=args.batch_size,
        max_retries=args.retries
    )
    
    try:
        await scraper.scrape_hadiths(
            start_id=args.start_id,
            num_hadiths=args.count
        )
    except KeyboardInterrupt:
        logger.info("\nScraping interrupted by user. Cleaning up...")
    finally:
        # Ensure proper cleanup
        await scraper.cleanup()
        logger.info("\nScraping completed.")

if __name__ == "__main__":
    # Make the script executable
    if sys.platform.startswith('win'):
        # Windows doesn't support shebang
        asyncio.run(main())
    else:
        # Make sure the script is executable on Unix-like systems
        script_path = Path(__file__)
        if not os.access(script_path, os.X_OK):
            os.chmod(script_path, 0o755)
        asyncio.run(main())
