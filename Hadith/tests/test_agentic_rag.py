"""
Comprehensive tests for the Agentic RAG system.

This module contains unit tests, integration tests, and performance tests
for the agentic RAG functionality including tools, services, and views.
"""
import asyncio
import json
import time
from unittest.mock import Mock, patch, MagicMock
from django.test import TestCase, TransactionTestCase, Client
from django.urls import reverse
from django.conf import settings
from django.test.utils import override_settings

from Hadith.models import Hadith
from Hadith.services.agentic_rag import (
    AgenticRagService, hadith_search, database_hadith_search,
    intelligent_hadith_search, internet_search,
    shia_book_ranking_tool, imam_hadith_rules_tool, SHIA_BOOKS
)
from Hadith.services.hadith_service import HadithService
from Hadith.repositories.hadith_repository import HadithRepository


class AgenticRagToolsTestCase(TestCase):
    """Test cases for agentic RAG tools."""
    
    def setUp(self):
        """Set up test data."""
        # Create test hadiths
        self.test_hadith = Hadith.objects.create(
            hadith_id=1,
            text="هذا حديث تجريبي عن الصلاة",
            text_without_tashkeel="هذا حديث تجريبي عن الصلاة",
            book_name="الكافي للكليني",
            volume="1",
            page_number="123",
            chapter="باب الصلاة",
            exporter="الكليني",
            narrators=["علي بن أبي طالب", "محمد بن علي"],
            url="http://example.com/hadith/1"
        )
        
        self.non_shia_hadith = Hadith.objects.create(
            hadith_id=2,
            text="حديث من مصدر غير شيعي",
            text_without_tashkeel="حديث من مصدر غير شيعي",
            book_name="صحيح البخاري",  # Not in SHIA_BOOKS
            volume="1",
            page_number="456",
            chapter="باب الطهارة",
            exporter="البخاري",
            narrators=["أبو هريرة"],
            url="http://example.com/hadith/2"
        )

    def test_database_hadith_search_filters_shia_books(self):
        """Test that database_hadith_search only returns results from Shia books."""
        result = database_hadith_search("الصلاة", n_results=10)
        
        # Should find the Shia hadith but not the non-Shia one
        self.assertIn("الكافي للكليني", result)
        self.assertNotIn("صحيح البخاري", result)
        self.assertIn("هذا حديث تجريبي عن الصلاة", result)

    def test_database_hadith_search_different_modes(self):
        """Test different search modes in database_hadith_search."""
        # Test exact search
        result_exact = database_hadith_search("هذا حديث تجريبي", search_mode='exact')
        self.assertIn("هذا حديث تجريبي عن الصلاة", result_exact)
        
        # Test all_words search
        result_all = database_hadith_search("حديث الصلاة", search_mode='all_words')
        self.assertIn("هذا حديث تجريبي عن الصلاة", result_all)
        
        # Test any_word search
        result_any = database_hadith_search("الصلاة الزكاة", search_mode='any_word')
        self.assertIn("هذا حديث تجريبي عن الصلاة", result_any)

    def test_database_hadith_search_no_results(self):
        """Test database_hadith_search when no results are found."""
        result = database_hadith_search("كلمة غير موجودة")
        self.assertIn("No Shia hadiths found", result)

    @patch('Hadith.services.agentic_rag._vector_store_service')
    def test_hadith_search_with_vector_store(self, mock_vector_store):
        """Test hadith_search tool with vector store available."""
        # Mock vector store service
        mock_vector_store.is_available = True
        mock_vector_store.semantic_search.return_value = [
            {
                'text': 'هذا حديث تجريبي عن الصلاة',
                'get_formatted_source': 'الكافي للكليني، ج1، ص123',
                'get_narration_chain': 'علي بن أبي طالب عن محمد بن علي',
                'similarity_score': 0.95
            }
        ]
        
        result = hadith_search("الصلاة", n_results=5)
        
        # Verify the mock was called with correct parameters
        mock_vector_store.semantic_search.assert_called_once_with(
            query="الصلاة",
            n_results=5,
            is_question=True,
            book_filter=SHIA_BOOKS
        )
        
        # Verify result format
        self.assertIn("Hadith 1:", result)
        self.assertIn("هذا حديث تجريبي عن الصلاة", result)
        self.assertIn("Similarity Score: 95.0%", result)

    @patch('Hadith.services.agentic_rag._vector_store_service')
    def test_hadith_search_no_vector_store(self, mock_vector_store):
        """Test hadith_search tool when vector store is unavailable."""
        mock_vector_store = None
        
        result = hadith_search("الصلاة")
        self.assertIn("Shia hadith search service is not available", result)


class AgenticRagServiceTestCase(TestCase):
    """Test cases for AgenticRagService."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_vector_store = Mock()
        self.mock_vector_store.is_available = True

    @patch('Hadith.services.agentic_rag.deepagents_available', True)
    @patch('Hadith.services.agentic_rag.settings.GEMINI_API_KEY', 'test_key')
    def test_service_initialization_success(self):
        """Test successful service initialization."""
        with patch('Hadith.services.agentic_rag.ChatGoogleGenerativeAI'), \
             patch('Hadith.services.agentic_rag.create_deep_agent'):
            
            service = AgenticRagService(self.mock_vector_store)
            self.assertTrue(service.is_available)

    @patch('Hadith.services.agentic_rag.deepagents_available', False)
    def test_service_initialization_no_deepagents(self):
        """Test service initialization when deepagents is not available."""
        service = AgenticRagService(self.mock_vector_store)
        self.assertFalse(service.is_available)

    @patch('Hadith.services.agentic_rag.settings.GEMINI_API_KEY', '')
    def test_service_initialization_no_api_key(self):
        """Test service initialization when API key is missing."""
        service = AgenticRagService(self.mock_vector_store)
        self.assertFalse(service.is_available)

    @patch('Hadith.services.agentic_rag.deepagents_available', True)
    @patch('Hadith.services.agentic_rag.settings.GEMINI_API_KEY', 'test_key')
    def test_tool_initialization_with_vector_store(self):
        """Test tool initialization when vector store is available."""
        with patch('Hadith.services.agentic_rag.ChatGoogleGenerativeAI'), \
             patch('Hadith.services.agentic_rag.create_deep_agent'):
            
            service = AgenticRagService(self.mock_vector_store)
            
            # Should have semantic search tool
            tool_names = [tool.__name__ for tool in service.tools]
            self.assertIn('hadith_search', tool_names)
            self.assertNotIn('database_hadith_search', tool_names)

    @patch('Hadith.services.agentic_rag.deepagents_available', True)
    @patch('Hadith.services.agentic_rag.settings.GEMINI_API_KEY', 'test_key')
    def test_tool_initialization_without_vector_store(self):
        """Test tool initialization when vector store is unavailable."""
        mock_vector_store = Mock()
        mock_vector_store.is_available = False
        
        with patch('Hadith.services.agentic_rag.ChatGoogleGenerativeAI'), \
             patch('Hadith.services.agentic_rag.create_deep_agent'):
            
            service = AgenticRagService(mock_vector_store)
            
            # Should have database search tool
            tool_names = [tool.__name__ for tool in service.tools]
            self.assertIn('database_hadith_search', tool_names)
            self.assertNotIn('hadith_search', tool_names)


class AgenticRagViewsTestCase(TestCase):
    """Test cases for agentic RAG views."""
    
    def setUp(self):
        """Set up test client and data."""
        self.client = Client()
        
        # Create test hadith
        self.test_hadith = Hadith.objects.create(
            hadith_id=1,
            text="هذا حديث تجريبي عن الصلاة",
            text_without_tashkeel="هذا حديث تجريبي عن الصلاة",
            book_name="الكافي للكليني",
            volume="1",
            page_number="123",
            chapter="باب الصلاة",
            exporter="الكليني",
            narrators=["علي بن أبي طالب"],
            url="http://example.com/hadith/1"
        )

    def test_agentic_stream_demo_view(self):
        """Test the agentic stream demo view."""
        url = reverse('hadith:agentic_stream_demo')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'البحث الذكي المتدفق')
        self.assertContains(response, 'EventSource')

    @patch('Hadith.services.hadith_service.HadithService.agentic_rag_search')
    def test_api_agentic_search_success(self, mock_search):
        """Test successful agentic search API call."""
        # Mock successful response
        mock_search.return_value = {
            'query': 'الصلاة',
            'answer': 'الصلاة هي عماد الدين',
            'reasoning_steps': ['البحث في المصادر الشيعية'],
            'tool_usage': {'hadith_search': 1},
            'sources': {
                'hadith_sources': [{'text': 'حديث عن الصلاة'}],
                'web_sources': []
            },
            'metadata': {'search_time': 2.5}
        }
        
        url = reverse('hadith:api_agentic_search')
        response = self.client.get(url, {'q': 'الصلاة'})
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data['success'])
        self.assertEqual(data['answer'], 'الصلاة هي عماد الدين')

    def test_api_agentic_search_no_query(self):
        """Test agentic search API with missing query."""
        url = reverse('hadith:api_agentic_search')
        response = self.client.get(url)
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertFalse(data['success'])
        self.assertIn('Query parameter is required', data['error'])


class AgenticRagPerformanceTestCase(TransactionTestCase):
    """Performance tests for agentic RAG system."""
    
    def setUp(self):
        """Set up performance test environment."""
        # Create multiple test hadiths for performance testing
        for i in range(100):
            Hadith.objects.create(
                hadith_id=i + 1,
                text=f"هذا حديث تجريبي رقم {i + 1} عن الصلاة والزكاة",
                text_without_tashkeel=f"هذا حديث تجريبي رقم {i + 1} عن الصلاة والزكاة",
                book_name="الكافي للكليني",
                volume="1",
                page_number=str(i + 1),
                chapter="باب الصلاة",
                exporter="الكليني",
                narrators=["علي بن أبي طالب"],
                url=f"http://example.com/hadith/{i + 1}"
            )

    def test_database_search_performance(self):
        """Test database search performance with multiple results."""
        start_time = time.time()
        
        result = database_hadith_search("الصلاة", n_results=50)
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # Should complete within 2 seconds
        self.assertLess(search_time, 2.0)
        self.assertIn("Found 50 hadiths", result)

    def test_repository_search_performance(self):
        """Test repository search performance."""
        start_time = time.time()
        
        filters = {'books': SHIA_BOOKS}
        paginator, page_obj, total_count = HadithRepository.search(
            query="الصلاة",
            page=1,
            per_page=50,
            filters=filters,
            search_mode='contains'
        )
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # Should complete within 1 second
        self.assertLess(search_time, 1.0)
        self.assertGreater(total_count, 0)
        self.assertLessEqual(len(page_obj.object_list), 50)


@override_settings(VECTOR_DB_ENABLED=False)
class AgenticRagDisabledVectorDBTestCase(TestCase):
    """Test agentic RAG behavior when vector database is disabled."""
    
    def setUp(self):
        """Set up test with disabled vector DB."""
        self.test_hadith = Hadith.objects.create(
            hadith_id=1,
            text="هذا حديث تجريبي عن الصلاة",
            text_without_tashkeel="هذا حديث تجريبي عن الصلاة",
            book_name="الكافي للكليني",
            volume="1",
            page_number="123",
            chapter="باب الصلاة",
            exporter="الكليني",
            narrators=["علي بن أبي طالب"],
            url="http://example.com/hadith/1"
        )

    def test_falls_back_to_database_search(self):
        """Test that system falls back to database search when vector DB is disabled."""
        result = database_hadith_search("الصلاة")
        
        # Should still work with database search
        self.assertIn("هذا حديث تجريبي عن الصلاة", result)
        self.assertIn("الكافي للكليني", result)

    @patch('Hadith.services.agentic_rag.deepagents_available', True)
    @patch('Hadith.services.agentic_rag.settings.GEMINI_API_KEY', 'test_key')
    def test_service_uses_database_tools_when_vector_disabled(self):
        """Test that service uses database tools when vector DB is disabled."""
        from Hadith.services.vector_store import VectorStoreService
        
        # Create a vector store service that should be disabled
        vector_store = VectorStoreService()
        
        with patch('Hadith.services.agentic_rag.ChatGoogleGenerativeAI'), \
             patch('Hadith.services.agentic_rag.create_deep_agent'):
            
            service = AgenticRagService(vector_store)
            
            # Should use database search tool instead of semantic search
            tool_names = [tool.__name__ for tool in service.tools]
            self.assertIn('database_hadith_search', tool_names)
            self.assertNotIn('hadith_search', tool_names)


class EnhancedAgenticRagToolsTestCase(TestCase):
    """Test cases for enhanced agentic RAG tools."""

    def setUp(self):
        """Set up test data."""
        # Create test hadiths
        self.test_hadith = Hadith.objects.create(
            hadith_id=1,
            text="هذا حديث تجريبي عن حوض النبي يوم القيامة",
            book_name="الكافي للكليني",
            volume=1,
            page_number=123,
            chapter="كتاب الطهارة",
            narration_chain="علي بن أبي طالب عن محمد بن علي"
        )

    def test_intelligent_hadith_search_multiple_strategies(self):
        """Test intelligent search with multiple strategies."""
        result = intelligent_hadith_search("حوض النبي يوم القيامة", n_results=3)

        # Should return a result with strategy information
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

        # Should indicate which strategy was used
        self.assertTrue(
            any(strategy in result for strategy in [
                "[Strategy:", "Exact match", "Contains match", "All words", "Any word", "Shorter query", "Key word"
            ])
        )

    def test_intelligent_hadith_search_no_results(self):
        """Test intelligent search when no results found."""
        result = intelligent_hadith_search("كلمات غير موجودة في قاعدة البيانات", n_results=3)

        # Should return a helpful message about trying different strategies
        self.assertIsInstance(result, str)
        self.assertIn("No hadiths found", result)
        self.assertIn("multiple search strategies", result)

    def test_shia_book_ranking_tool_specific_book(self):
        """Test Shia book ranking tool with specific book."""
        result = shia_book_ranking_tool("الكافي للكليني")

        self.assertIsInstance(result, str)
        self.assertIn("الكافي للكليني", result)
        self.assertIn("أصح الكتب الأربعة", result)

    def test_shia_book_ranking_tool_general(self):
        """Test Shia book ranking tool without specific book."""
        result = shia_book_ranking_tool()

        self.assertIsInstance(result, str)
        self.assertIn("ترتيب الكتب الشيعية", result)
        self.assertIn("الكافي للكليني", result)
        self.assertIn("من لا يحضره الفقيه", result)

    def test_imam_hadith_rules_tool_general(self):
        """Test Imam hadith rules tool with general guidance."""
        result = imam_hadith_rules_tool("general")

        self.assertIsInstance(result, str)
        self.assertIn("قواعد الأئمة", result)
        self.assertIn("ما وافق القرآن", result)

    def test_imam_hadith_rules_tool_interpretation(self):
        """Test Imam hadith rules tool for interpretation guidance."""
        result = imam_hadith_rules_tool("interpretation")

        self.assertIsInstance(result, str)
        self.assertIn("لا يفسر القرآن إلا النبي أو الإمام", result)

    def test_imam_hadith_rules_tool_fatwa(self):
        """Test Imam hadith rules tool for fatwa guidance."""
        result = imam_hadith_rules_tool("fatwa")

        self.assertIsInstance(result, str)
        self.assertIn("لا يفتي إلا النبي أو الإمام", result)

    @patch('Hadith.services.agentic_rag._tavily_client')
    def test_internet_search_enhanced_query(self, mock_tavily):
        """Test internet search with enhanced Shia-specific query."""
        # Mock Tavily response
        mock_tavily.search.return_value = {
            'results': [
                {
                    'title': 'Shia Islamic Library',
                    'url': 'https://shiaonlinelibrary.com/test',
                    'content': 'Test content about Shia Islam'
                }
            ]
        }

        result = internet_search("حوض النبي", max_results=3)

        # Should enhance query with Shia sites
        mock_tavily.search.assert_called_once()
        call_args = mock_tavily.search.call_args[0]
        enhanced_query = call_args[0]

        self.assertIn("site:shiaonlinelibrary.com", enhanced_query)
        self.assertIn("site:ar.lib.eshia.ir", enhanced_query)
        self.assertIn("حوض النبي", enhanced_query)

    def test_database_hadith_search_with_filtering(self):
        """Test database search with Shia filtering option."""
        # Test with Shia filtering enabled
        result = database_hadith_search("حوض النبي", n_results=5, filter_shia_only=True)

        self.assertIsInstance(result, str)
        self.assertIn("Shia sources", result)

        # Test with Shia filtering disabled
        result = database_hadith_search("حوض النبي", n_results=5, filter_shia_only=False)

        self.assertIsInstance(result, str)
        self.assertIn("all sources", result)
