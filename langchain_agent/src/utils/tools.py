import logging
import os
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

module_logger = logging.getLogger('langchain_agent')


def create_tavily_tool():
    """Create a Tavily search tool if API key is available."""
    if not TAVILY_API_KEY:
        return None

    tavily_search = TavilySearch(
        tavily_api_key=TAVILY_API_KEY,
        max_results=5,
        topic="general"
    )

    @tool
    def tavily_web_search(query: str) -> str:
        """Search the web for current information about events, news, weather, facts, and real-time data.

        Use this tool when you need to find:
        - Current events and news
        - Weather information
        - Recent facts and data
        - Information that changes over time
        - Things that happened recently

        Args:
            query: The search query to look up

        Returns:
            Formatted search results with relevant information
        """
        try:
            module_logger.info(f"[tavily_tool] Searching: {query[:100]}")
            response = tavily_search.invoke({"query": query})
            results = response.get("results", [])

            if not results:
                return "No search results found for that query."

            formatted = "Search results:\n\n"
            for i, result in enumerate(results[:3], 1):
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "")
                formatted += f"{i}. {title}\n{content}\n"
                if url:
                    formatted += f"Source: {url}\n"
                formatted += "\n"

            return formatted.strip()
        except Exception as e:
            module_logger.error(f"[tavily_tool] Error: {e}")
            return f"Search error: {str(e)}"

    return tavily_web_search
