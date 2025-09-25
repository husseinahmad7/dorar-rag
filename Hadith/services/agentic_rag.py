"""
Agentic RAG service for advanced hadith search with internet integration.

This module provides agentic RAG functionality using deepagents framework,
integrating hadith semantic search with internet search capabilities.
It follows Django service patterns with proper error handling and logging.
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from django.conf import settings
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Deepagents imports
try:
    from deepagents import create_deep_agent
    deepagents_available = True
except ImportError as e:
    deepagents_available = False
    logging.warning(f"deepagents not available: {e}. Agentic RAG will not be available.")

# Tavily imports
try:
    from tavily import TavilyClient
    tavily_available = True
except ImportError as e:
    tavily_available = False
    logging.warning(f"tavily not available: {e}. Internet search will not be available.")

# Import repository for database search
from Hadith.repositories.hadith_repository import HadithRepository

# Configure logging
logger = logging.getLogger(__name__)

# Global service instance for tools to access
_vector_store_service = None
_tavily_client = None

# String constants for prompts and messages
HADITH_EXPERT_PROMPT = """You are a strict expert in Shia Islamic hadiths and prophetic traditions.
Your role is to search for and analyze hadith content EXCLUSIVELY from authentic Shia sources.

CRITICAL AUTHENTICATION RULES:
1. NEVER fabricate or distort hadiths - only report what exists in authentic Shia sources
2. NEVER interpret the Quran - only the Prophet (PBUH) or Imams (AS) can interpret
3. NEVER issue fatwas (religious rulings) - only the Prophet or Imams can rule
4. When hadiths conflict, follow Imam guidance: prioritize what agrees with Quran, then by fame among companions, then by opposing the general public
5. Combine hadiths only as the Imams instructed for resolving contradictions
6. Always verify against the Four Main Books: الكافي (most authentic), من لا يحضره الفقيه, تهذيب الأحكام, الاستبصار

SEARCH STRATEGY:
- Try multiple search approaches: exact terms, partial terms, synonyms, related concepts
- Use different search modes when initial searches fail
- Search for key concepts even if exact wording isn't found
- Consider alternative Arabic expressions for the same concept

STRICT REQUIREMENTS:
- Only discuss matters with specific sacred texts from Shia sources
- Provide exact book names, volumes, and page numbers when available
- Distinguish between authentic (صحيح) and weak (ضعيف) narrations
- Never mix Shia and non-Shia sources without clear distinction"""
SHIA_BOOKS = [
    'كتاب سليم بن قيس الهلالي', 'الكافي للكليني', 'من لا يحضره الفقيه', 'عيون الرضا',
    'التوحيد للصدوق', 'تفسير الامام العسكري', 'تفسير العياشي', 'تفسير القمي',
    'تفسير النعماني', 'نحريف السياري', 'فصل الخطاب في تحريف كتاب رب الأرباب',
    'المحاسن للبرقي', 'بصائر الدرجات للصفار', 'قرب الاسناد للحميري', 'كامل الزيارات للقمي',
    'الخصال للصدوق', 'علل الشرائع للصدوق', 'أمالي الصدوق', 'كمال الدين وتمام النعمة',
    'معاني الأخبار', 'تحف العقول', 'الاختصاص للمفيد', 'الجعفريات', 'مستدرك الوسائل',
    'رجال الكشي', 'الاعتقادات للصدوق', 'توحيد المفضل', 'الكافئة للمفيد'
]
SHIA_RESEARCH_PROMPT = """You are a strict researcher in authentic Shia Islamic studies.
Your role is to search for and verify information from ONLY authentic Shia sources on the internet.

STRICT AUTHENTICATION RULES:
1. ONLY use authentic Shia websites: shiaonlinelibrary.com, ar.lib.eshia.ir, eshia.ir
2. NEVER cite non-Shia sources without clear warning about their non-Shia origin
3. NEVER provide interpretations - only report what Shia scholars have said
4. NEVER issue fatwas - only report existing Shia scholarly opinions
5. Always verify information against classical Shia sources when possible

PRIORITY SOURCES:
- Official Shia scholarly institutions and their websites
- Authenticated Shia digital libraries
- Verified contemporary Shia scholars' published works
- Cross-reference with the Four Main Books when possible

SEARCH STRATEGY:
- Focus on sites: shiaonlinelibrary.com, ar.lib.eshia.ir
- Look for multiple Shia scholarly confirmations
- Distinguish between authentic and questionable sources
- Report source authenticity level with each finding

You complement hadith database search by providing verified contemporary Shia scholarly context."""

SYSTEM_INSTRUCTIONS = """
You are an intelligent Shia Islamic knowledge assistant that specializes in authentic Shia sources
and can provide comprehensive answers about Shia Islamic topics.

Your capabilities:
1. Search authentic Shia hadith collections for prophetic traditions and narrations from the Imams (AS)
2. Search for contemporary Shia Islamic scholarship and resources online
3. Combine information from classical and contemporary Shia sources intelligently
4. Provide accurate citations from authentic Shia books and scholars
5. Distinguish between different levels of hadith authenticity in Shia tradition

Language Guidelines:
- Detect the language of the user's query (Arabic, English, etc.)
- Respond in the same language as the query
- If the query is in Arabic, provide your answer in Arabic
- If the query is in English, provide your answer in English
- For mixed language queries, use the dominant language
- Always maintain proper Arabic grammar and diacritics when responding in Arabic
- Use appropriate Shia Islamic terminology in the query language

Content Guidelines:
- Always prioritize authentic Shia hadith sources (الكافي, من لا يحضره الفقيه, تفسير العياشي، etc.)
- Search exclusively within the 28 authentic Shia books in your database
- Use internet search for contemporary Shia scholarly opinions and modern context
- Clearly distinguish between classical hadith content and contemporary scholarship
- Provide proper citations with book names, volumes, and page numbers when available
- Mention the chain of narration (isnad) when discussing hadith authenticity
- Be respectful and accurate when discussing Shia Islamic topics
- Always attribute narrations to the Prophet (PBUH) or specific Imams (AS) when known

When answering:
1. First determine the language of the query
2. Determine if the question requires classical Shia hadith sources (use hadith search)
3. Consider if contemporary Shia scholarly context would be helpful (use internet search)
4. Combine information from both classical and contemporary sources thoughtfully
5. Provide clear, well-cited responses in the appropriate language with proper Shia attribution
"""

# Error messages
ERROR_HADITH_SERVICE_UNAVAILABLE = "Shia hadith search service is not available."
ERROR_INTERNET_SERVICE_UNAVAILABLE = "Internet search service for Shia research is not available."
ERROR_AGENTIC_SERVICE_UNAVAILABLE = "Shia agentic RAG service is not available"

# Language detection helper
def detect_query_language(query: str) -> str:
    """
    Detect the primary language of a query.

    Args:
        query (str): The query text

    Returns:
        str: Language code ('ar' for Arabic, 'en' for English)
    """
    # Simple Arabic detection - check for Arabic characters
    arabic_chars = sum(1 for char in query if '\u0600' <= char <= '\u06FF')
    total_chars = len([char for char in query if char.isalpha()])

    if total_chars == 0:
        return 'en'  # Default to English if no alphabetic characters

    arabic_ratio = arabic_chars / total_chars
    return 'ar' if arabic_ratio > 0.3 else 'en'

def set_vector_store_service(service):
    """Set the global vector store service for tools to use."""
    global _vector_store_service
    _vector_store_service = service

def set_tavily_client(api_key: str):
    """Set the global Tavily client for tools to use."""
    global _tavily_client
    if tavily_available and api_key:
        try:
            _tavily_client = TavilyClient(api_key=api_key)
            logger.info("Tavily client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Tavily client: {e}")
            _tavily_client = None


# Tool functions for deepagents
def hadith_search(query: str, n_results: int = 5, keywords: List[str] = None) -> str:
    """
    Search for Shia Islamic hadiths using semantic similarity with dynamic keywords.

    This tool searches exclusively within authentic Shia Islamic books and sources.
    Use this tool when the user asks about Shia Islamic teachings, prophetic traditions,
    or specific hadith content from Shia sources.

    Args:
        query (str): The search query in Arabic
        n_results (int): Number of results to return (default: 5)
        keywords (List[str], optional): Additional keywords to enhance search

    Returns:
        str: Formatted search results from Shia sources with proper attribution
    """
    try:
        if not _vector_store_service or not _vector_store_service.is_available:
            return ERROR_HADITH_SERVICE_UNAVAILABLE

        # Enhance query with keywords if provided
        enhanced_query = query
        if keywords:
            keyword_str = " ".join(keywords)
            enhanced_query = f"{query} {keyword_str}"

        # Perform semantic search with Shia books filter
        results = _vector_store_service.semantic_search(
            query=enhanced_query,
            n_results=n_results,
            is_question=True,
            book_filter=SHIA_BOOKS
        )

        if not results:
            # Try with original query if enhanced query didn't work
            if keywords:
                results = _vector_store_service.semantic_search(
                    query=query,
                    n_results=n_results,
                    is_question=True,
                    book_filter=SHIA_BOOKS
                )

        if not results:
            return f"No Shia hadiths found for query: {query}. Search is limited to authentic Shia sources."

        # Format results for agent consumption
        formatted_results = []
        for i, hadith in enumerate(results, 1):
            formatted_result = f"""
Hadith {i}:
Text: {hadith['text']}
Source: {hadith.get('get_formatted_source', 'Unknown')}
Narration Chain: {hadith.get('get_narration_chain', 'Unknown')}
Similarity Score: {hadith.get('similarity_score', 0):.1f}%
---
"""
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Error in hadith search tool: {e}")
        return f"Hadith search failed: {str(e)}"


def database_hadith_search(query: str, n_results: int = 5, search_mode: str = 'contains',
                          filter_shia_only: bool = True) -> str:
    """
    Search for Islamic hadiths using database text search with intelligent strategies.

    This tool searches within Islamic hadith collections using database text matching.
    It can filter for Shia-specific sources or search all available sources based on need.

    Args:
        query (str): The search query in Arabic
        n_results (int): Number of results to return (default: 5)
        search_mode (str): Search mode - 'contains', 'all_words', 'exact', or 'any_word'
        filter_shia_only (bool): Whether to filter only Shia books (default: True)

    Returns:
        str: Formatted search results with proper source attribution
    """
    try:
        logger.info(f"Database search: query='{query}', mode={search_mode}, shia_only={filter_shia_only}")

        # Apply Shia book filtering only if requested
        filters = {'books': SHIA_BOOKS} if filter_shia_only else None

        # Perform database search
        paginator, page_obj, total_count = HadithRepository.search(
            query=query,
            page=1,
            per_page=n_results,
            filters=filters,
            search_mode=search_mode
        )

        if not page_obj.object_list:
            return f"No Shia hadiths found in database for query: {query}. Search is limited to authentic Shia sources."

        # Format results for agent consumption
        formatted_results = []
        for i, hadith in enumerate(page_obj.object_list, 1):
            hadith_dict = HadithRepository.format_hadith_dict(hadith)
            formatted_result = f"""
Hadith {i}:
Text: {hadith_dict['text']}
Source: {hadith_dict['formatted_source']}
Narration Chain: {hadith_dict['narration_chain']}
Book: {hadith_dict['book_name']}
---
"""
            formatted_results.append(formatted_result)

        source_type = "Shia" if filter_shia_only else "all"
        result_summary = f"Found {len(page_obj.object_list)} hadiths (out of {total_count} total matches) from {source_type} sources:\n\n"
        return result_summary + "\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Error in database hadith search tool: {e}")
        return f"Database hadith search failed: {str(e)}"


def intelligent_hadith_search(query: str, n_results: int = 5) -> str:
    """
    Perform intelligent hadith search with multiple strategies and query variations.

    This tool tries different search approaches when initial searches don't yield results:
    - Different search modes (exact, contains, all_words, any_word)
    - Query variations (removing words, using synonyms)
    - Both Shia-only and broader searches

    Args:
        query (str): The search query in Arabic
        n_results (int): Number of results to return (default: 5)

    Returns:
        str: Formatted search results with strategy information
    """
    try:
        logger.info(f"Intelligent search for: '{query}'")

        # Strategy 1: Exact search in Shia sources
        result = database_hadith_search(query, n_results, 'exact', True)
        if "Found 0 hadiths" not in result:
            return f"[Strategy: Exact match in Shia sources]\n{result}"

        # Strategy 2: Contains search in Shia sources
        result = database_hadith_search(query, n_results, 'contains', True)
        if "Found 0 hadiths" not in result:
            return f"[Strategy: Contains match in Shia sources]\n{result}"

        # Strategy 3: All words search in Shia sources
        result = database_hadith_search(query, n_results, 'all_words', True)
        if "Found 0 hadiths" not in result:
            return f"[Strategy: All words in Shia sources]\n{result}"

        # Strategy 4: Any word search in Shia sources
        result = database_hadith_search(query, n_results, 'any_word', True)
        if "Found 0 hadiths" not in result:
            return f"[Strategy: Any word in Shia sources]\n{result}"

        # Strategy 5: Try with shorter query (remove last word)
        words = query.split()
        if len(words) > 1:
            shorter_query = ' '.join(words[:-1])
            result = database_hadith_search(shorter_query, n_results, 'contains', True)
            if "Found 0 hadiths" not in result:
                return f"[Strategy: Shorter query '{shorter_query}' in Shia sources]\n{result}"

        # Strategy 6: Try individual key words
        key_words = [word for word in words if len(word) > 2]  # Skip short words
        for word in key_words:
            result = database_hadith_search(word, n_results, 'contains', True)
            if "Found 0 hadiths" not in result:
                return f"[Strategy: Key word '{word}' in Shia sources]\n{result}"

        return f"No hadiths found for query '{query}' using multiple search strategies. Consider rephrasing the query or checking for spelling variations."

    except Exception as e:
        logger.error(f"Error in intelligent hadith search: {e}")
        return f"Error in intelligent search: {str(e)}"


def internet_search(
    query: str,
    max_results: int = 5,
    include_raw_content: bool = False,
) -> str:
    """
    Search the internet for Shia Islamic information using Tavily.

    This tool searches for information prioritizing authentic Shia sources like
    shiaonlinelibrary.com and ar.lib.eshia.ir for reliable Islamic content.

    Args:
        query (str): The search query in Arabic
        max_results (int): Number of results to return (default: 5)
        include_raw_content (bool): Whether to include raw content (default: False)

    Returns:
        str: Formatted search results from internet sources
    """
    try:
        if not _tavily_client:
            return ERROR_INTERNET_SERVICE_UNAVAILABLE

        # Enhance query with Shia-specific sites for better results
        enhanced_query = f"{query} site:shiaonlinelibrary.com OR site:ar.lib.eshia.ir OR site:eshia.ir"

        # Perform search using Tavily client (remove topic parameter to avoid error)
        search_docs = _tavily_client.search(
            enhanced_query,
            max_results=max_results,
            include_raw_content=include_raw_content,
        )

        if not search_docs or not search_docs.get('results'):
            return f"No internet results found for query: {query}"

        # Format results for agent consumption
        formatted_results = []
        results = search_docs.get('results', [])

        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            content = result.get('content', 'No content')

            formatted_result = f"""
Result {i}:
Title: {title}
URL: {url}
Content: {content[:500]}{'...' if len(content) > 500 else ''}
---
"""
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Error in internet search tool: {e}")
        return f"Internet search failed: {str(e)}"


def shia_book_ranking_tool(book_name: str = "") -> str:
    """
    Get information about Shia book rankings and authenticity levels.

    Args:
        book_name (str): Specific book name to check (optional)

    Returns:
        str: Information about book rankings and authenticity
    """
    try:
        book_rankings = {
            'الكافي للكليني': 'أصح الكتب الأربعة - أهم مصدر للأحاديث الشيعية',
            'من لا يحضره الفقيه': 'من الكتب الأربعة المعتمدة - للشيخ الصدوق',
            'تهذيب الأحكام': 'من الكتب الأربعة المعتمدة - للشيخ الطوسي',
            'الاستبصار': 'من الكتب الأربعة المعتمدة - للشيخ الطوسي'
        }

        if book_name and book_name in book_rankings:
            return f"كتاب {book_name}: {book_rankings[book_name]}"

        return "ترتيب الكتب الشيعية: 1. الكافي للكليني (الأصح) 2. من لا يحضره الفقيه 3. تهذيب الأحكام 4. الاستبصار"

    except Exception as e:
        return f"خطأ في استرجاع معلومات الكتب: {str(e)}"


def imam_hadith_rules_tool(topic: str = "general") -> str:
    """
    Get Imam guidance on hadith methodology and dealing with conflicts.

    Args:
        topic (str): Type of guidance needed (general, interpretation, fatwa)

    Returns:
        str: Imam guidance on hadith methodology
    """
    try:
        rules = {
            'general': "قواعد الأئمة: ما وافق القرآن فخذوه، ما خالف القرآن فاتركوه، الترجيح بالشهرة عند الأئمة وشيعتهم الثقات ومخالفة العامة",
            'interpretation': "لا يفسر القرآن إلا النبي أو الإمام المعصوم - التفسير بالرأي محرم",
            'fatwa': "لا يفتي إلا النبي أو الإمام المعصوم - في الغيبة: الاحتياط وعدم الحكم"
        }

        return rules.get(topic, rules['general'])

    except Exception as e:
        return f"خطأ في استرجاع قواعد الأئمة: {str(e)}"




class AgenticRagService:
    """
    Service for agentic RAG functionality specialized in Shia Islamic knowledge using deepagents framework.

    This service integrates Shia hadith semantic search with contemporary Shia Islamic research capabilities
    using intelligent agents that can decide when to use each tool and how to combine the results
    for comprehensive answers about Shia Islamic topics.

    The service exclusively searches within authentic Shia sources and provides proper attribution
    to classical Shia books and contemporary Shia scholarship.

    Attributes:
        HADITH_EXPERT_AGENT (dict): Configuration for Shia hadith expert subagent
        SHIA_RESEARCH_AGENT (dict): Configuration for contemporary Shia research subagent
    """

    # Subagent configurations
    HADITH_EXPERT_AGENT = {
        "name": "shia_hadith_expert",
        "description": "Expert in Shia Islamic hadiths and prophetic traditions from authentic Shia sources",
        "prompt": HADITH_EXPERT_PROMPT
    }

    SHIA_RESEARCH_AGENT = {
        "name": "shia_contemporary_researcher",
        "description": "Expert in contemporary Shia Islamic scholarship and internet research",
        "prompt": SHIA_RESEARCH_PROMPT
    }

    def __init__(self, vector_store_service):
        """
        Initialize the agentic RAG service.

        Args:
            vector_store_service: Instance of VectorStoreService for hadith search
        """
        self.vector_store_service = vector_store_service
        self.is_available = False
        self.agent = None
        self.memory = None

        # Configuration from Django settings
        self.gemini_api_key = settings.GEMINI_API_KEY
        self.tavily_api_key = settings.TAVILY_API_KEY
        self.llm_model = settings.DEFAULT_LLM_MODEL
        self.max_subagents = settings.DEEPAGENTS_SETTINGS.get('MAX_SUBAGENTS', 2)
        self.memory_buffer_size = settings.DEEPAGENTS_SETTINGS.get('MEMORY_BUFFER_SIZE', 10)
        self.use_internet = settings.DEEPAGENTS_SETTINGS.get('DEFAULT_USE_INTERNET', True)
        self.max_runtime = settings.DEEPAGENTS_SETTINGS.get('MAX_RUNTIME_SECONDS', 60)

        # Set global services for tools to use
        set_vector_store_service(vector_store_service)
        if self.tavily_api_key:
            set_tavily_client(self.tavily_api_key)

        # Initialize if dependencies are available
        logger.info(f"Checking dependencies: deepagents_available={deepagents_available}, has_gemini_key={bool(self.gemini_api_key)}")

        if deepagents_available and self.gemini_api_key:
            try:
                logger.info("Starting agentic RAG service initialization...")
                self._initialize_tools()
                logger.info("Tools initialized successfully")
                self._initialize_memory()
                logger.info("Memory initialized successfully")
                self._initialize_agent()
                logger.info("Agent initialized successfully")
                self.is_available = True
                logger.info("Agentic RAG service initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing agentic RAG service: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                self.is_available = False
        else:
            missing = []
            if not deepagents_available:
                missing.append("deepagents")
            if not self.gemini_api_key:
                missing.append("GEMINI_API_KEY")
            logger.warning(f"Agentic RAG service not available. Missing: {', '.join(missing)}")

    def _initialize_tools(self) -> None:
        """Initialize the tools available to the agent."""
        self.tools = []

        # Add hadith search tools based on vector database availability
        if _vector_store_service and _vector_store_service.is_available:
            # Vector database is available - use semantic search
            self.tools.append(hadith_search)
            logger.info("Semantic hadith search tool added (vector database available)")
        else:
            # Vector database not available - use database search
            self.tools.append(database_hadith_search)
            logger.info("Database hadith search tool added (vector database not available)")

        # Add intelligent search tool for better query strategies
        self.tools.append(intelligent_hadith_search)
        logger.info("Intelligent hadith search tool added")

        # Add Shia knowledge tools
        self.tools.append(shia_book_ranking_tool)
        self.tools.append(imam_hadith_rules_tool)
        logger.info("Shia knowledge tools added")

        # Add internet search tool if available
        if _tavily_client:
            self.tools.append(internet_search)
            logger.info("Internet search tool added")
        else:
            logger.warning("Internet search tool not available (no Tavily client)")

    def _initialize_memory(self) -> None:
        """Initialize conversation memory for the agent."""
        try:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                max_token_limit=self.memory_buffer_size * 100  # Rough estimate
            )
            logger.info("Conversation memory initialized")
        except Exception as e:
            logger.error(f"Error initializing memory: {e}")
            self.memory = None

    def _initialize_agent(self) -> None:
        """Initialize the deep agent."""
        try:
            logger.info("Starting deep agent initialization...")

            # Initialize LLM with Gemini 2.0 Flash
            logger.info(f"Initializing LLM with model: {self.llm_model}")
            self.llm = ChatGoogleGenerativeAI(
                model=self.llm_model,
                google_api_key=self.gemini_api_key,
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192  # Increased for better responses
            )
            logger.info("LLM initialized successfully")

            # Get subagents configuration
            logger.info("Getting subagents configuration...")
            subagents = self._get_subagents_config()
            logger.info(f"Configured {len(subagents)} subagents")

            # Create the deep agent
            logger.info("Creating deep agent...")
            self.agent = create_deep_agent(
                tools=self.tools,
                instructions=self._get_system_instructions(),
                model=self.llm,
                subagents=subagents
            )

            logger.info("Deep agent initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing deep agent: {e}")
            raise

    def _get_system_instructions(self) -> str:
        """
        Get system instructions for the agent.

        Returns:
            str: System instructions
        """
        return SYSTEM_INSTRUCTIONS

    def _get_subagents_config(self) -> List[Dict[str, Any]]:
        """
        Get subagents configuration.

        Returns:
            List[Dict[str, Any]]: Subagents configuration
        """
        subagents = []

        # Add hadith expert subagent - specialized for Shia hadith search
        hadith_expert = self.HADITH_EXPERT_AGENT.copy()

        # Assign appropriate search tool based on vector database availability
        if _vector_store_service and _vector_store_service.is_available:
            hadith_expert["tools"] = ['hadith_search']  # Semantic search
        else:
            hadith_expert["tools"] = ['database_hadith_search']  # Database search

        subagents.append(hadith_expert)

        # Add research subagent if internet search is available - specialized for contemporary Shia research
        if _tavily_client:
            research_expert = self.SHIA_RESEARCH_AGENT.copy()
            research_expert["tools"] = ['internet_search']
            subagents.append(research_expert)

        return subagents[:self.max_subagents]

    def search(self, query: str, use_internet: Optional[bool] = None,
               max_subagents: Optional[int] = None,
               memory_enabled: bool = True) -> Dict[str, Any]:
        """
        Perform agentic RAG search.

        Args:
            query (str): The search query
            use_internet (Optional[bool]): Whether to use internet search
            max_subagents (Optional[int]): Maximum number of subagents to use
            memory_enabled (bool): Whether to use conversation memory

        Returns:
            Dict[str, Any]: Search results with agent reasoning
        """
        logger.info(f"Agentic search called - is_available: {self.is_available}, agent: {self.agent is not None}")

        if not self.is_available or not self.agent:
            logger.warning(f"Agentic service unavailable - is_available: {self.is_available}, agent: {self.agent is not None}")
            return self._create_error_response(query, ERROR_AGENTIC_SERVICE_UNAVAILABLE)

        try:
            start_time = datetime.now()

            # Prepare messages with memory context
            messages = self._prepare_messages(query, memory_enabled)

            # Prepare search configuration
            config = self._prepare_search_config(use_internet, max_subagents)

            # Invoke the agent
            logger.info(f"Invoking agent for query: {query}")
            result = self.agent.invoke({
                "messages": messages
            }, config=config)
            logger.info("Agent invocation completed successfully")

            end_time = datetime.now()

            # Format the result
            formatted_result = self._format_agent_result(
                result, query, start_time, end_time, memory_enabled
            )

            # Update memory if available
            if memory_enabled and self.memory:
                self._update_memory(query, formatted_result['answer'])

            return formatted_result

        except Exception as e:
            logger.error(f"Error in agentic RAG search: {e}")
            return self._create_error_response(query, str(e))

    async def asearch(self, query: str, use_internet: Optional[bool] = None,
                     max_subagents: Optional[int] = None,
                     memory_enabled: bool = True) -> Dict[str, Any]:
        """
        Async version of agentic RAG search.

        Args:
            query (str): The search query
            use_internet (Optional[bool]): Whether to use internet search
            max_subagents (Optional[int]): Maximum number of subagents to use
            memory_enabled (bool): Whether to use conversation memory

        Returns:
            Dict[str, Any]: Search results with agent reasoning
        """
        # Run the synchronous search in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.search, query, use_internet, max_subagents, memory_enabled
        )

    async def asearch_stream(self, query: str, use_internet: Optional[bool] = None,
                            max_subagents: Optional[int] = None,
                            memory_enabled: bool = True):
        """
        Async streaming version of agentic RAG search that yields progress updates.

        Args:
            query (str): The search query
            use_internet (Optional[bool]): Whether to use internet search
            max_subagents (Optional[int]): Maximum number of subagents to use
            memory_enabled (bool): Whether to use conversation memory

        Yields:
            Dict[str, Any]: Progress updates and final results
        """
        if not self.is_available or not self.agent:
            yield {
                'type': 'error',
                'message': ERROR_AGENTIC_SERVICE_UNAVAILABLE,
                'timestamp': datetime.now().isoformat()
            }
            return

        try:
            start_time = datetime.now()

            # Initial status
            yield {
                'type': 'status',
                'message': 'بدء البحث في المصادر الشيعية...' if self._is_arabic_query(query) else 'Starting search in Shia sources...',
                'timestamp': start_time.isoformat()
            }

            # Prepare messages with memory context
            yield {
                'type': 'status',
                'message': 'إعداد السياق والذاكرة...' if self._is_arabic_query(query) else 'Preparing context and memory...',
                'timestamp': datetime.now().isoformat()
            }
            messages = self._prepare_messages(query, memory_enabled)

            # Prepare search configuration
            config = self._prepare_search_config(use_internet, max_subagents)

            # Show which tools are available
            available_tools = []
            if _vector_store_service and _vector_store_service.is_available:
                available_tools.append('البحث الدلالي في الأحاديث' if self._is_arabic_query(query) else 'Semantic hadith search')
            else:
                available_tools.append('البحث في قاعدة البيانات' if self._is_arabic_query(query) else 'Database search')

            if _tavily_client:
                available_tools.append('البحث في الإنترنت' if self._is_arabic_query(query) else 'Internet search')

            yield {
                'type': 'tools',
                'message': f"الأدوات المتاحة: {', '.join(available_tools)}" if self._is_arabic_query(query) else f"Available tools: {', '.join(available_tools)}",
                'tools': available_tools,
                'timestamp': datetime.now().isoformat()
            }

            # Show agent activation
            yield {
                'type': 'agent_start',
                'message': 'تفعيل الوكلاء المتخصصين...' if self._is_arabic_query(query) else 'Activating specialized agents...',
                'timestamp': datetime.now().isoformat()
            }

            # Run the agent search in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.agent.invoke({
                    "messages": messages
                }, config=config)
            )

            end_time = datetime.now()

            # Show completion
            yield {
                'type': 'agent_complete',
                'message': 'اكتمل البحث بنجاح' if self._is_arabic_query(query) else 'Search completed successfully',
                'timestamp': end_time.isoformat()
            }

            # Format the result
            formatted_result = self._format_agent_result(
                result, query, start_time, end_time, memory_enabled
            )

            # Update memory if available
            if memory_enabled and self.memory:
                self._update_memory(query, formatted_result['answer'])

            # Final result
            yield {
                'type': 'result',
                'data': formatted_result,
                'timestamp': end_time.isoformat()
            }

        except Exception as e:
            logger.error(f"Error in agentic RAG streaming search: {e}")
            yield {
                'type': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _is_arabic_query(self, query: str) -> bool:
        """
        Check if the query is primarily in Arabic.

        Args:
            query (str): The query to check

        Returns:
            bool: True if query is primarily Arabic
        """
        arabic_chars = sum(1 for char in query if '\u0600' <= char <= '\u06FF')
        total_chars = sum(1 for char in query if char.isalpha())
        return total_chars > 0 and (arabic_chars / total_chars) > 0.5

    def _prepare_messages(self, query: str, memory_enabled: bool) -> List[Dict[str, str]]:
        """
        Prepare messages for the agent including memory context.

        Args:
            query (str): The user query
            memory_enabled (bool): Whether to include memory context

        Returns:
            List[Dict[str, str]]: Formatted messages
        """
        messages = []

        # Add memory context if enabled
        if memory_enabled and self.memory:
            try:
                memory_messages = self.memory.chat_memory.messages
                for msg in memory_messages[-10:]:  # Last 10 messages
                    if isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        messages.append({"role": "assistant", "content": msg.content})
            except Exception as e:
                logger.warning(f"Error loading memory: {e}")

        # Detect query language and prepare enhanced query
        query_language = detect_query_language(query)
        language_name = "Arabic" if query_language == 'ar' else "English"

        # Enhanced query with language instruction
        enhanced_query = f"{query}\n\n[SYSTEM INSTRUCTION: The user's query is in {language_name}. Please respond in {language_name} and use appropriate Islamic terminology in {language_name}.]"

        # Add current query
        messages.append({"role": "user", "content": enhanced_query})

        return messages

    def _prepare_search_config(self, use_internet: Optional[bool],
                             max_subagents: Optional[int]) -> Dict[str, Any]:
        """
        Prepare search configuration for the agent.

        Args:
            use_internet (Optional[bool]): Whether to use internet search
            max_subagents (Optional[int]): Maximum number of subagents

        Returns:
            Dict[str, Any]: Search configuration
        """
        config = {}

        # Configure internet usage
        if use_internet is not None:
            config['use_internet'] = use_internet

        # Configure subagent limits
        if max_subagents is not None:
            config['max_subagents'] = min(max_subagents, self.max_subagents)

        return config

    def _format_agent_result(self, result: Dict[str, Any], query: str,
                           start_time: datetime, end_time: datetime,
                           memory_enabled: bool) -> Dict[str, Any]:
        """
        Format the agent result into a standardized response.

        Args:
            result (Dict[str, Any]): Raw agent result
            query (str): Original query
            start_time (datetime): Search start time
            end_time (datetime): Search end time
            memory_enabled (bool): Whether memory was used

        Returns:
            Dict[str, Any]: Formatted result
        """
        try:
            # Extract the final answer from agent messages
            messages = result.get('messages', [])
            final_message = messages[-1] if messages else None

            # Handle AIMessage object properly
            if final_message and hasattr(final_message, 'content'):
                answer = final_message.content or 'No answer generated'
            elif isinstance(final_message, dict):
                answer = final_message.get('content', 'No answer generated')
            else:
                answer = 'No answer generated'
            # Extract tool usage information
            tool_usage = self._extract_tool_usage(messages)

            # Extract reasoning process
            reasoning_steps = self._extract_reasoning_steps(messages)

            return {
                'query': query,
                'answer': answer,
                'reasoning_steps': reasoning_steps,
                'tool_usage': tool_usage,
                'sources': self._extract_sources(messages),
                'metadata': {
                    'search_time_seconds': (end_time - start_time).total_seconds(),
                    'memory_enabled': memory_enabled,
                    'agent_available': True,
                    'tools_used': list(tool_usage.keys()),
                    'timestamp': end_time.isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error formatting agent result: {e}")
            return self._create_error_response(query, f"Error formatting result: {str(e)}")

    def _extract_tool_usage(self, messages: List[Any]) -> Dict[str, List[str]]:
        """
        Extract tool usage information from agent messages.

        Args:
            messages (List[Any]): Agent messages (can be LangChain message objects or dicts)

        Returns:
            Dict[str, List[str]]: Tool usage summary
        """
        tool_usage = {}

        for message in messages:
            # Handle LangChain message objects
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = getattr(tool_call, 'name', 'unknown')
                    tool_input = getattr(tool_call, 'args', {})

                    if tool_name not in tool_usage:
                        tool_usage[tool_name] = []

                    tool_usage[tool_name].append(str(tool_input))
            # Handle dictionary messages
            elif isinstance(message, dict) and 'tool_calls' in message:
                for tool_call in message['tool_calls']:
                    tool_name = tool_call.get('name', 'unknown')
                    tool_input = tool_call.get('args', {})

                    if tool_name not in tool_usage:
                        tool_usage[tool_name] = []

                    tool_usage[tool_name].append(str(tool_input))

        return tool_usage

    def _extract_reasoning_steps(self, messages: List[Any]) -> List[str]:
        """
        Extract reasoning steps from agent messages.

        Args:
            messages (List[Any]): Agent messages (can be LangChain message objects or dicts)

        Returns:
            List[str]: Reasoning steps
        """
        reasoning_steps = []

        for message in messages:
            # Handle LangChain message objects
            if hasattr(message, 'content'):
                content = message.content or ''
                # Determine message type from class name
                message_type = type(message).__name__.lower()
                is_assistant = 'ai' in message_type or 'assistant' in message_type
            # Handle dictionary messages
            elif isinstance(message, dict):
                content = message.get('content', '')
                role = message.get('role', '')
                is_assistant = role == 'assistant'
            else:
                continue

            # Look for assistant messages that contain reasoning
            if is_assistant and content:
                # Simple heuristic to identify reasoning steps
                if any(keyword in content.lower() for keyword in
                      ['thinking', 'analyzing', 'searching', 'considering', 'plan']):
                    reasoning_steps.append(content)

        return reasoning_steps

    def _extract_sources(self, messages: List[Any]) -> Dict[str, List[str]]:
        """
        Extract sources from agent messages.

        Args:
            messages (List[Any]): Agent messages (can be LangChain message objects or dicts)

        Returns:
            Dict[str, List[str]]: Sources by type
        """
        sources = {
            'hadith_sources': [],
            'shia_sources': []
        }

        for message in messages:
            # Handle LangChain message objects
            if hasattr(message, 'content'):
                content = message.content or ''
            # Handle dictionary messages
            elif isinstance(message, dict):
                content = message.get('content', '')
            else:
                continue

            # Look for hadith sources
            if 'hadith' in content.lower() and 'source:' in content.lower():
                # Extract hadith source information
                lines = content.split('\n')
                for line in lines:
                    if 'source:' in line.lower():
                        sources['hadith_sources'].append(line.strip())

            # Look for Shia sources
            if any(term in content.lower() for term in ['shia', "shi'a", 'shiite', 'imam']):
                lines = content.split('\n')
                for line in lines:
                    if 'url:' in line.lower() or line.strip().startswith('http'):
                        sources['shia_sources'].append(line.strip())

        return sources

    def _update_memory(self, query: str, answer: str) -> None:
        """
        Update conversation memory with the query and answer.

        Args:
            query (str): User query
            answer (str): Agent answer
        """
        try:
            if self.memory:
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(answer)
                logger.debug("Memory updated with query and answer")
        except Exception as e:
            logger.warning(f"Error updating memory: {e}")

    def _create_error_response(self, query: str, error_message: str) -> Dict[str, Any]:
        """
        Create an error response.

        Args:
            query (str): Original query
            error_message (str): Error message

        Returns:
            Dict[str, Any]: Error response
        """
        return {
            'query': query,
            'answer': f"I apologize, but I encountered an error while processing your request: {error_message}",
            'reasoning_steps': [],
            'tool_usage': {},
            'sources': {'hadith_sources': [], 'shia_sources': []},
            'metadata': {
                'search_time_seconds': 0,
                'memory_enabled': False,
                'agent_available': False,
                'tools_used': [],
                'error': error_message,
                'timestamp': datetime.now().isoformat()
            }
        }

    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        try:
            if self.memory:
                self.memory.clear()
                logger.info("Conversation memory cleared")
        except Exception as e:
            logger.error(f"Error clearing memory: {e}")

    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current memory state.

        Returns:
            Dict[str, Any]: Memory summary
        """
        try:
            if not self.memory:
                return {'available': False, 'message_count': 0}

            messages = self.memory.chat_memory.messages
            return {
                'available': True,
                'message_count': len(messages),
                'last_messages': [
                    {
                        'type': type(msg).__name__,
                        'content': msg.content[:100] + '...' if len(msg.content) > 100 else msg.content
                    }
                    for msg in messages[-5:]  # Last 5 messages
                ]
            }
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {'available': False, 'error': str(e)}

    def run_sync(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Synchronous wrapper for async operations.

        This method allows the service to be called from Django views
        without requiring async view functions.

        Args:
            query (str): The search query
            **kwargs: Additional arguments for search

        Returns:
            Dict[str, Any]: Search results
        """
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.search, query, **kwargs)
                    return future.result(timeout=self.max_runtime)
            else:
                # We can use asyncio.run
                return asyncio.run(self.asearch(query, **kwargs))
        except Exception as e:
            logger.error(f"Error in sync wrapper: {e}")
            return self.search(query, **kwargs)  # Fallback to sync version