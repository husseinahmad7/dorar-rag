#!/usr/bin/env python
import argparse
import asyncio
import sys
import os
from pathlib import Path

# Import the scraper
from hadith_scraper import HadithScraper

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
    
    return parser.parse_args()

async def main():
    """Main entry point for the scraper CLI"""
    args = parse_arguments()
    
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Construct output file path
    output_file = data_dir / args.output
    
    # Print configuration
    print(f"Hadith Scraper Configuration:")
    print(f"  - Starting ID: {args.start_id}")
    print(f"  - Count: {args.count}")
    print(f"  - Workers: {args.workers}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Output File: {output_file}")
    print(f"  - Headless Mode: {'Enabled' if args.headless else 'Disabled'}")
    print(f"  - Timeout: {args.timeout} seconds")
    print(f"  - Retries: {args.retries}")
    print(f"  - Verbose: {'Enabled' if args.verbose else 'Disabled'}")
    print("\nStarting scraper...\n")
    
    # Initialize and run the scraper
    scraper = HadithScraper(
        max_workers=args.workers,
        timeout=args.timeout,
        headless=args.headless,
        verbose=args.verbose,
        output_file=args.output,
        batch_size=args.batch_size,
        max_retries=args.retries
    )
    
    try:
        await scraper.scrape_hadiths(
            start_id=args.start_id,
            num_hadiths=args.count
        )
    except KeyboardInterrupt:
        print("\nScraping interrupted by user. Cleaning up...")
    finally:
        # Ensure proper cleanup
        await scraper.cleanup()
        print("\nScraping completed.")

if __name__ == "__main__":
    asyncio.run(main())
