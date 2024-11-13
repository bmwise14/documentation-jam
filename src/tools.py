from pydantic import BaseModel, Field
from typing import List, Dict, Any
import requests
import json
from langchain_core.tools import BaseTool

################################################
class AcademicPaperSearchInput(BaseModel):
    topic: str = Field(..., description="The topic to search for academic papers on")
    max_results: int = Field(5, description="Maximum number of results to return")

class AcademicPaperSearchTool(BaseTool):
    args_schema: type = AcademicPaperSearchInput  # Explicit type annotation
    name: str = Field("academic_paper_search_tool", description="Tool for searching academic papers")
    description: str = Field("Queries an academic papers API to retrieve relevant articles based on a topic")

    def __init__(self, name: str = "academic_paper_search_tool", 
                 description: str = "Queries an academic paper API to retrieve relevant articles based on a topic"):
        super().__init__()
        self.name = name
        self.description = description

    def _run(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
        # Query an external academic API like arXiv, Semantic Scholar, or CrossRef
        search_results = self.query_academic_api(topic, max_results)
        # print(search_results)
        return search_results

    async def _arun(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
        raise NotImplementedError("Async version not implemented")

    def query_academic_api(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": topic,
            "limit": max_results,
            "fields": "title,abstract,authors,year,url"
        }
        
        response = requests.get(base_url, params=params)
        print(response)
        
        if response.status_code == 200:
            papers = response.json().get("data", [])
            formatted_results = [
                {
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                    "authors": [author.get("name") for author in paper.get("authors", [])],
                    "year": paper.get("year"),
                    "url": paper.get("url")
                }
                for paper in papers
            ]
            return formatted_results
        else:
            raise ValueError(f"Failed to fetch papers: {response.status_code} - {response.text}")

################################################