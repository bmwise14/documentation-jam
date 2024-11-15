from pydantic import BaseModel, Field
from typing import List, Dict, Any
import requests
import json
from langchain_core.tools import BaseTool

from bs4 import BeautifulSoup
import pymupdf4llm
# https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/index.html

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
        # testing = search_results[0]['text'][:100]

        return search_results

    async def _arun(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
        raise NotImplementedError("Async version not implemented")

    def query_academic_api(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
        base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": topic,
            "limit": 1, # max_results
            "fields": "title,abstract,authors,year,url,openAccessPdf",
            "openAccessPdf" : True
        }
        
        response = requests.get(base_url, params=params)
        print(response)
        
        if response.status_code == 200:
            papers = response.json().get("data", [])
            formatted_results = [
                {
                    "title"     : paper.get("title"),
                    "abstract"  : paper.get("abstract"),
                    "authors"   : [author.get("name") for author in paper.get("authors", [])],
                    "year"      : paper.get("year"),
                    "url"       : paper.get("url"),
                    "pdf"       : paper.get("openAccessPdf"),
                    "text"      : self.get_paper_content(paper['openAccessPdf']['url'])
                }
                for paper in papers
            ]

            return formatted_results
        else:
            raise ValueError(f"Failed to fetch papers: {response.status_code} - {response.text}")
    
    def get_paper_content(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            # Assuming we can extract text from the response, e.g., from a PDF or HTML
            return self.extract_text_from_response(response)  # Define or import this helper function as needed
        except Exception as e:
            print(f"Failed to retrieve content from {url}: {e}")
            return None
    
    def extract_text_from_response(self, response):
        # Detect the content type
        content_type = response.headers.get('Content-Type', '').lower()
        
        if 'application/pdf' in content_type:
            return self.extract_text_from_pdf(response.content)
        elif 'text/html' in content_type:
            return self.extract_text_from_html(response.text)
        elif 'text/plain' in content_type:
            return response.text
        else:
            print(f"Unsupported content type: {content_type}")
            return None

    def extract_text_from_pdf(self, pdf_content):
        # Write PDF content to a temporary file to use with pymupdf4llm
        with open("temp.pdf", "wb") as f:
            f.write(pdf_content)
        # Convert PDF to markdown text
        md_text = pymupdf4llm.to_markdown("temp.pdf")
        return md_text

    def extract_text_from_html(self, html_content):
        # Use BeautifulSoup to extract text from HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ')
        return text


# ################################################
# class GoogleScholarSearchInput(BaseModel):
#     topic: str = Field(..., description="The topic to search for academic papers on")
#     max_results: int = Field(5, description="Maximum number of results to return")

# class GoogleScholarSearchTool(BaseTool):
#     args_schema: type = GoogleScholarSearchInput  # Explicit type annotation
#     name: str = Field("academic_paper_search_tool", description="Tool for searching academic papers")
#     description: str = Field("Queries an academic papers API to retrieve relevant articles based on a topic")

#     def __init__(self, name: str = "academic_paper_search_tool", 
#                  description: str = "Queries an academic paper API to retrieve relevant articles based on a topic"):
#         super().__init__()
#         self.name = name
#         self.description = description

#     def _run(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
#         # Query an external academic API like arXiv, Semantic Scholar, or CrossRef
#         search_results = self.query_academic_api(topic, max_results)
#         # print(search_results)
#         return search_results

#     async def _arun(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
#         raise NotImplementedError("Async version not implemented")

#     def query_academic_api(self, topic: str, max_results: int) -> List[Dict[str, Any]]:
#         base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
#         params = {
#             "query": topic,
#             "limit": max_results,
#             "fields": "title,abstract,authors,year,url"
#         }
        
#         response = requests.get(base_url, params=params)
#         print(response)
        
#         if response.status_code == 200:
#             papers = response.json().get("data", [])
#             formatted_results = [
#                 {
#                     "title": paper.get("title"),
#                     "abstract": paper.get("abstract"),
#                     "authors": [author.get("name") for author in paper.get("authors", [])],
#                     "year": paper.get("year"),
#                     "url": paper.get("url")
#                 }
#                 for paper in papers
#             ]
#             return formatted_results
#         else:
#             raise ValueError(f"Failed to fetch papers: {response.status_code} - {response.text}")