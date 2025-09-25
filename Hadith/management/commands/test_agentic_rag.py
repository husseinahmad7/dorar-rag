"""
Django management command for testing agentic RAG functionality.

This command provides a comprehensive testing interface for the agentic RAG system,
allowing developers and administrators to test hadith search with internet integration,
configure various parameters, and observe the agent's decision-making process.

Usage:
    python manage.py test_agentic_rag "What does Islam say about charity?"
    python manage.py test_agentic_rag "Recent developments in AI" --use-internet
    python manage.py test_agentic_rag "Hadith about prayer" --no-internet --max-subagents 1
"""

import json
import logging
import time
from typing import Dict, Any, Optional

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings

from Hadith.services.hadith_service import HadithService


class Command(BaseCommand):
    """
    Django management command for testing agentic RAG functionality.
    
    This command demonstrates the agentic RAG system capabilities including:
    - Hadith semantic search
    - Internet search integration
    - Agent reasoning and tool selection
    - Memory management
    - Subagent coordination
    """
    
    help = (
        'Test the agentic RAG functionality with various configuration options. '
        'This command demonstrates how the agent combines hadith search with '
        'internet search to provide comprehensive answers.'
    )
    
    def add_arguments(self, parser):
        """Add command line arguments."""
        # Required query argument
        parser.add_argument(
            'query',
            type=str,
            help='The test query to search for (required)'
        )
        
        # Internet search options
        parser.add_argument(
            '--use-internet',
            action='store_true',
            default=None,
            help='Enable internet search (default: use service default)'
        )
        
        parser.add_argument(
            '--no-internet',
            action='store_true',
            default=False,
            help='Disable internet search (overrides --use-internet)'
        )
        
        # Subagent configuration
        parser.add_argument(
            '--max-subagents',
            type=int,
            default=None,
            help='Maximum number of subagents to use (default: use service default)'
        )
        
        # Memory options
        parser.add_argument(
            '--no-memory',
            action='store_true',
            default=False,
            help='Disable conversation memory'
        )
        
        parser.add_argument(
            '--clear-memory',
            action='store_true',
            default=False,
            help='Clear conversation memory before testing'
        )
        
        # Output options
        parser.add_argument(
            '--verbose',
            '-v',
            action='store_true',
            default=False,
            help='Enable verbose output with detailed logging'
        )
        
        parser.add_argument(
            '--json-output',
            action='store_true',
            default=False,
            help='Output results in JSON format'
        )
        
        parser.add_argument(
            '--save-results',
            type=str,
            default=None,
            help='Save results to specified file path'
        )
        
        # Testing options
        parser.add_argument(
            '--fallback-test',
            action='store_true',
            default=False,
            help='Test fallback to regular RAG (simulates agentic RAG failure)'
        )
        
        parser.add_argument(
            '--benchmark',
            action='store_true',
            default=False,
            help='Run performance benchmarking'
        )
        
        # Multiple queries for batch testing
        parser.add_argument(
            '--batch-file',
            type=str,
            default=None,
            help='File containing multiple queries to test (one per line)'
        )
    
    def handle(self, *args, **options):
        """Main command handler."""
        try:
            # Configure logging
            self._configure_logging(options['verbose'])
            
            # Initialize the service
            self.stdout.write("Initializing Hadith Service...")
            hadith_service = HadithService()
            
            # Check agentic RAG availability
            if not hadith_service.is_agentic_rag_available() and not options['fallback_test']:
                raise CommandError(
                    "Agentic RAG service is not available. "
                    "Check your configuration and dependencies. "
                    "Use --fallback-test to test regular RAG fallback."
                )
            
            # Clear memory if requested
            if options['clear_memory']:
                self._clear_memory(hadith_service)
            
            # Process batch file or single query
            if options['batch_file']:
                self._process_batch_file(hadith_service, options)
            else:
                self._process_single_query(hadith_service, options)
                
        except Exception as e:
            self.stderr.write(
                self.style.ERROR(f'Command failed: {str(e)}')
            )
            if options['verbose']:
                import traceback
                self.stderr.write(traceback.format_exc())
            raise CommandError(f'Test failed: {str(e)}')
    
    def _configure_logging(self, verbose: bool) -> None:
        """Configure logging for the command."""
        if verbose:
            # Enable detailed logging
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Enable specific loggers
            loggers = [
                'Hadith.services.agentic_rag',
                'Hadith.services.hadith_service',
                'Hadith.services.vector_store'
            ]
            
            for logger_name in loggers:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.DEBUG)
        else:
            # Standard logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(levelname)s: %(message)s'
            )
    
    def _clear_memory(self, hadith_service: HadithService) -> None:
        """Clear conversation memory."""
        self.stdout.write("Clearing conversation memory...")
        
        if hadith_service.clear_agentic_memory():
            self.stdout.write(
                self.style.SUCCESS("✓ Memory cleared successfully")
            )
        else:
            self.stdout.write(
                self.style.WARNING("⚠ Could not clear memory (may not be available)")
            )
    
    def _process_single_query(self, hadith_service: HadithService, options: Dict[str, Any]) -> None:
        """Process a single query."""
        query = options['query']
        
        self.stdout.write(f"\n{self.style.HTTP_INFO('='*60)}")
        self.stdout.write(f"{self.style.HTTP_INFO('TESTING AGENTIC RAG FUNCTIONALITY')}")
        self.stdout.write(f"{self.style.HTTP_INFO('='*60)}")
        
        # Display configuration
        self._display_configuration(options)
        
        # Display query
        self.stdout.write(f"\n{self.style.WARNING('Query:')} {query}")
        
        # Show memory status
        self._display_memory_status(hadith_service)
        
        # Perform the search
        self.stdout.write(f"\n{self.style.HTTP_INFO('Executing agentic RAG search...')}")
        
        start_time = time.time()
        
        try:
            if options['fallback_test']:
                # Test fallback by forcing agentic RAG to be unavailable
                result = hadith_service._fallback_to_regular_rag(
                    query, 
                    error_message="Simulated agentic RAG failure for testing"
                )
            else:
                # Normal agentic RAG search
                result = hadith_service.agentic_rag_search(
                    query=query,
                    use_internet=self._get_internet_setting(options),
                    max_subagents=options.get('max_subagents'),
                    memory_enabled=not options['no_memory'],
                    fallback_to_rag=True
                )
            
            end_time = time.time()
            
            # Display results
            self._display_results(result, options, end_time - start_time)
            
            # Save results if requested
            if options['save_results']:
                self._save_results(result, options['save_results'], options)
                
            # Run benchmark if requested
            if options['benchmark']:
                self._run_benchmark(hadith_service, query, options)
                
        except Exception as e:
            self.stderr.write(
                self.style.ERROR(f'Search failed: {str(e)}')
            )
            raise
    
    def _process_batch_file(self, hadith_service: HadithService, options: Dict[str, Any]) -> None:
        """Process multiple queries from a batch file."""
        batch_file = options['batch_file']
        
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            raise CommandError(f"Batch file not found: {batch_file}")
        except Exception as e:
            raise CommandError(f"Error reading batch file: {str(e)}")
        
        if not queries:
            raise CommandError("No queries found in batch file")
        
        self.stdout.write(f"\n{self.style.HTTP_INFO('='*60)}")
        self.stdout.write(f"{self.style.HTTP_INFO('BATCH TESTING AGENTIC RAG FUNCTIONALITY')}")
        self.stdout.write(f"{self.style.HTTP_INFO('='*60)}")
        self.stdout.write(f"Processing {len(queries)} queries from: {batch_file}")
        
        results = []
        total_time = 0
        
        for i, query in enumerate(queries, 1):
            self.stdout.write(f"\n{self.style.WARNING(f'Query {i}/{len(queries)}:')} {query}")
            
            start_time = time.time()
            
            try:
                result = hadith_service.agentic_rag_search(
                    query=query,
                    use_internet=self._get_internet_setting(options),
                    max_subagents=options.get('max_subagents'),
                    memory_enabled=not options['no_memory'],
                    fallback_to_rag=True
                )
                
                end_time = time.time()
                query_time = end_time - start_time
                total_time += query_time
                
                # Add timing to result
                result['batch_metadata'] = {
                    'query_number': i,
                    'query_time_seconds': query_time
                }
                
                results.append(result)
                
                # Brief result summary
                self.stdout.write(f"  ✓ Completed in {query_time:.2f}s")
                if options['verbose']:
                    self.stdout.write(f"  Answer: {result['answer'][:100]}...")
                
            except Exception as e:
                self.stderr.write(
                    self.style.ERROR(f"  ✗ Failed: {str(e)}")
                )
                results.append({
                    'query': query,
                    'error': str(e),
                    'batch_metadata': {
                        'query_number': i,
                        'query_time_seconds': 0
                    }
                })
        
        # Display batch summary
        self._display_batch_summary(results, total_time)
        
        # Save batch results if requested
        if options['save_results']:
            self._save_batch_results(results, options['save_results'], options)
    
    def _get_internet_setting(self, options: Dict[str, Any]) -> Optional[bool]:
        """Determine internet search setting from options."""
        if options['no_internet']:
            return False
        elif options['use_internet']:
            return True
        else:
            return None  # Use service default
    
    def _display_configuration(self, options: Dict[str, Any]) -> None:
        """Display current configuration."""
        self.stdout.write(f"\n{self.style.HTTP_INFO('Configuration:')}")
        
        # Internet search setting
        internet_setting = self._get_internet_setting(options)
        if internet_setting is None:
            internet_text = "Service default"
        elif internet_setting:
            internet_text = "Enabled"
        else:
            internet_text = "Disabled"
        
        self.stdout.write(f"  Internet Search: {internet_text}")
        
        # Subagents setting
        max_subagents = options.get('max_subagents')
        subagents_text = str(max_subagents) if max_subagents else "Service default"
        self.stdout.write(f"  Max Subagents: {subagents_text}")
        
        # Memory setting
        memory_text = "Disabled" if options['no_memory'] else "Enabled"
        self.stdout.write(f"  Memory: {memory_text}")
        
        # Special modes
        if options['fallback_test']:
            self.stdout.write(f"  {self.style.WARNING('Mode: Fallback Testing')}")
        
        if options['benchmark']:
            self.stdout.write(f"  {self.style.WARNING('Mode: Benchmarking Enabled')}")
    
    def _display_memory_status(self, hadith_service: HadithService) -> None:
        """Display current memory status."""
        memory_summary = hadith_service.get_agentic_memory_summary()
        
        if memory_summary.get('available'):
            message_count = memory_summary.get('message_count', 0)
            self.stdout.write(f"Memory Status: {message_count} messages in history")
        else:
            reason = memory_summary.get('reason', 'Unknown')
            self.stdout.write(f"Memory Status: Not available ({reason})")
    
    def _display_results(self, result: Dict[str, Any], options: Dict[str, Any], 
                        execution_time: float) -> None:
        """Display search results."""
        if options['json_output']:
            self._display_json_results(result)
            return
        
        self.stdout.write(f"\n{self.style.SUCCESS('='*60)}")
        self.stdout.write(f"{self.style.SUCCESS('SEARCH RESULTS')}")
        self.stdout.write(f"{self.style.SUCCESS('='*60)}")
        
        # Execution time
        self.stdout.write(f"Execution Time: {execution_time:.2f} seconds")
        
        # Agent reasoning steps
        reasoning_steps = result.get('reasoning_steps', [])
        if reasoning_steps:
            self.stdout.write(f"\n{self.style.HTTP_INFO('Agent Reasoning Process:')}")
            for i, step in enumerate(reasoning_steps, 1):
                self.stdout.write(f"  {i}. {step}")
        
        # Tool usage
        tool_usage = result.get('tool_usage', {})
        if tool_usage:
            self.stdout.write(f"\n{self.style.HTTP_INFO('Tools Used:')}")
            for tool_name, usages in tool_usage.items():
                self.stdout.write(f"  {tool_name}:")
                for usage in usages:
                    self.stdout.write(f"    - {usage}")
        
        # Sources
        sources = result.get('sources', {})
        hadith_sources = sources.get('hadith_sources', [])
        web_sources = sources.get('web_sources', [])
        
        if hadith_sources:
            self.stdout.write(f"\n{self.style.HTTP_INFO('Hadith Sources:')}")
            for i, source in enumerate(hadith_sources, 1):
                self.stdout.write(f"  {i}. {source}")
        
        if web_sources:
            self.stdout.write(f"\n{self.style.HTTP_INFO('Web Sources:')}")
            for i, source in enumerate(web_sources, 1):
                self.stdout.write(f"  {i}. {source}")
        
        # Final answer
        answer = result.get('answer', 'No answer generated')
        self.stdout.write(f"\n{self.style.WARNING('Final Answer:')}")
        self.stdout.write(f"{answer}")
        
        # Metadata
        metadata = result.get('metadata', {})
        if options['verbose'] and metadata:
            self.stdout.write(f"\n{self.style.HTTP_INFO('Metadata:')}")
            for key, value in metadata.items():
                self.stdout.write(f"  {key}: {value}")
        
        # Fallback information
        if metadata.get('fallback_used'):
            fallback_reason = metadata.get('fallback_reason', 'Unknown')
            self.stdout.write(f"\n{self.style.WARNING('Note: Fallback to regular RAG was used')}")
            self.stdout.write(f"Reason: {fallback_reason}")
    
    def _display_json_results(self, result: Dict[str, Any]) -> None:
        """Display results in JSON format."""
        try:
            json_output = json.dumps(result, indent=2, ensure_ascii=False)
            self.stdout.write(json_output)
        except Exception as e:
            self.stderr.write(f"Error formatting JSON: {str(e)}")
            self.stdout.write(str(result))
    
    def _display_batch_summary(self, results: list, total_time: float) -> None:
        """Display summary of batch processing."""
        successful = len([r for r in results if 'error' not in r])
        failed = len(results) - successful
        avg_time = total_time / len(results) if results else 0
        
        self.stdout.write(f"\n{self.style.SUCCESS('='*60)}")
        self.stdout.write(f"{self.style.SUCCESS('BATCH PROCESSING SUMMARY')}")
        self.stdout.write(f"{self.style.SUCCESS('='*60)}")
        
        self.stdout.write(f"Total Queries: {len(results)}")
        self.stdout.write(f"Successful: {successful}")
        self.stdout.write(f"Failed: {failed}")
        self.stdout.write(f"Total Time: {total_time:.2f} seconds")
        self.stdout.write(f"Average Time: {avg_time:.2f} seconds per query")
        
        if failed > 0:
            self.stdout.write(f"\n{self.style.ERROR('Failed Queries:')}")
            for result in results:
                if 'error' in result:
                    query_num = result.get('batch_metadata', {}).get('query_number', '?')
                    self.stdout.write(f"  {query_num}. {result['query']}: {result['error']}")
    
    def _save_results(self, result: Dict[str, Any], file_path: str, 
                     options: Dict[str, Any]) -> None:
        """Save single query results to file."""
        try:
            # Add command options to result for context
            result['test_metadata'] = {
                'command_options': {
                    'use_internet': self._get_internet_setting(options),
                    'max_subagents': options.get('max_subagents'),
                    'memory_enabled': not options['no_memory'],
                    'fallback_test': options['fallback_test']
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.stdout.write(f"\n✓ Results saved to: {file_path}")
            
        except Exception as e:
            self.stderr.write(f"Error saving results: {str(e)}")
    
    def _save_batch_results(self, results: list, file_path: str, 
                           options: Dict[str, Any]) -> None:
        """Save batch results to file."""
        try:
            batch_data = {
                'batch_metadata': {
                    'total_queries': len(results),
                    'successful': len([r for r in results if 'error' not in r]),
                    'failed': len([r for r in results if 'error' in r]),
                    'command_options': {
                        'use_internet': self._get_internet_setting(options),
                        'max_subagents': options.get('max_subagents'),
                        'memory_enabled': not options['no_memory'],
                        'batch_file': options['batch_file']
                    },
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                },
                'results': results
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(batch_data, f, indent=2, ensure_ascii=False)
            
            self.stdout.write(f"\n✓ Batch results saved to: {file_path}")
            
        except Exception as e:
            self.stderr.write(f"Error saving batch results: {str(e)}")
    
    def _run_benchmark(self, hadith_service: HadithService, query: str, 
                      options: Dict[str, Any]) -> None:
        """Run performance benchmarking."""
        self.stdout.write(f"\n{self.style.HTTP_INFO('='*60)}")
        self.stdout.write(f"{self.style.HTTP_INFO('PERFORMANCE BENCHMARK')}")
        self.stdout.write(f"{self.style.HTTP_INFO('='*60)}")
        
        # Test configurations
        test_configs = [
            {'name': 'Hadith Only', 'use_internet': False, 'max_subagents': 1},
            {'name': 'With Internet', 'use_internet': True, 'max_subagents': 1},
            {'name': 'Multiple Subagents', 'use_internet': True, 'max_subagents': 2},
        ]
        
        benchmark_results = []
        
        for config in test_configs:
            self.stdout.write(f"\nTesting: {config['name']}")
            
            times = []
            for i in range(3):  # Run 3 times for average
                start_time = time.time()
                
                try:
                    result = hadith_service.agentic_rag_search(
                        query=query,
                        use_internet=config['use_internet'],
                        max_subagents=config['max_subagents'],
                        memory_enabled=False,  # Disable memory for consistent benchmarking
                        fallback_to_rag=True
                    )
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    times.append(execution_time)
                    
                    self.stdout.write(f"  Run {i+1}: {execution_time:.2f}s")
                    
                except Exception as e:
                    self.stderr.write(f"  Run {i+1}: Failed - {str(e)}")
                    times.append(float('inf'))
            
            # Calculate statistics
            valid_times = [t for t in times if t != float('inf')]
            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)
                
                benchmark_results.append({
                    'config': config['name'],
                    'avg_time': avg_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'success_rate': len(valid_times) / len(times)
                })
                
                self.stdout.write(f"  Average: {avg_time:.2f}s")
            else:
                self.stdout.write(f"  All runs failed")
        
        # Display benchmark summary
        self.stdout.write(f"\n{self.style.SUCCESS('Benchmark Summary:')}")
        for result in benchmark_results:
            self.stdout.write(
                f"  {result['config']}: "
                f"Avg {result['avg_time']:.2f}s, "
                f"Min {result['min_time']:.2f}s, "
                f"Max {result['max_time']:.2f}s, "
                f"Success {result['success_rate']*100:.0f}%"
            )