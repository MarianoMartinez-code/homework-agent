from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

tavily = TavilySearch(
    max_results=5,
    topic="general",
)

