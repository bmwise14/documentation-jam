from pydantic import BaseModel, Field
from typing import List, Dict, Any, Type
import requests
import json
from langchain_core.tools import BaseTool

from bs4 import BeautifulSoup
import pymupdf4llm
import sys
import os

from langchain_openai import ChatOpenAI
import prompts
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, ChatMessage



################################################
class AcademicPaperSearchInput(BaseModel):
    topic: str = Field(..., description="The topic to search for academic papers on")
    max_results: int = Field(20, description="Maximum number of results to return")

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
            "limit": max_results, # max_results
            "fields": "title,abstract,authors,year,openAccessPdf",
            "openAccessPdf" : True
        }
        try: 
            while True:
                try: 
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
                                "pdf"       : paper.get("openAccessPdf"),
                                # "text"      : self.get_paper_content(paper['openAccessPdf']['url'])
                            }
                            for paper in papers
                        ]

                        return formatted_results
                except:
                    # raise ValueError(f"Failed to fetch papers: {response.status_code} - {response.text}")
                    print((f"Failed to fetch papers: {response.status_code} - {response.text}. Trying Again..."))
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            sys.exit(0)  # Clean exit
            
    
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

################################################
class PaperAnalysisInput(BaseModel):
    paper_path: str = Field(..., description="Path to the paper file to analyze")

class PaperAnalysisTool(BaseTool):
    args_schema: type = PaperAnalysisInput  # Explicit type annotation
    name: str = Field("paper_analyzer", description="Tool for searching academic papers")
    description: str = Field("Analyzes academic papers and provides a structured analysis")
    
    def _run(self, paper_path: str) -> Dict:
        """Analyze a single academic paper"""
        try:
            # Convert PDF to markdown
            md_text = pymupdf4llm.to_markdown(paper_path)
            print(md_text)
            
            # Setup messages for analysis
            messages = [
                SystemMessage(content=prompts.analyze_paper_prompt),
                HumanMessage(content=md_text)
            ]
            
            # Analyze with GPT-4
            model = ChatOpenAI(model='gpt-4')
            response = model.invoke(messages, temperature=0.1)
            
            return {
                "paper": paper_path,
                "analysis": response.content
            }
            
        except Exception as e:
            return {
                "paper": paper_path,
                "error": str(e)
            }

    async def _arun(self, paper_path: str, prompt: str) -> Dict:
        """Async version of paper analysis"""
        raise NotImplementedError("Async analysis not implemented")


# state['papers'][-1].content


class PaperDownloaderInput(BaseModel):
    url: str = Field(..., description="The URL of the paper to download")

class PaperDownloaderTool(BaseTool):
    name: str = "paper_downloader"
    description: str = "Downloads academic papers from pdf URLs"
    args_schema: Type[BaseModel] = PaperDownloaderInput

    def _run(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Create a papers directory if it doesn't exist
            if not os.path.exists('papers'):
                os.makedirs('papers')
            
            # Generate a filename from the URL
            filename = f"papers/{url.split('/')[-1]}"
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            # Save the PDF
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            return filename
            
        except Exception as e:
            return f"Error downloading paper: {str(e)}"

    async def _arun(self, url: str) -> str:
        raise NotImplementedError("Async version not implemented")