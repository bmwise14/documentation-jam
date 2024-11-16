from dotenv import load_dotenv
_ = load_dotenv()

import requests
import ast
import psycopg
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import operator

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, ChatMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

import os
from uuid import uuid4

import prompts

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from tools import AcademicPaperSearchTool, PaperDownloaderTool, PaperAnalysisTool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.constants import Send
import pymupdf4llm

#############################################################
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

#############################################################
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]
    systematic_review_outline : str
    last_human_index : int
    papers : List[str]
    paper : Annotated[str, operator.add]
    analyses: Annotated[List[Dict], operator.add]  # Store analysis results
    combined_analysis: str  # Final combined analysis
    title: str
    
class Agent:
    def __init__(self, model, tools, checkpointer, temperature=0.1):
        self.temperature=temperature
        self.tools = {t.name: t for t in tools} if tools else {}
        self.model = model.bind_tools(tools) if tools else model
        
        graph = StateGraph(AgentState)
        graph.add_node("process_input", self.process_input)
        graph.add_node("planner", self.plan_node)
        graph.add_node("researcher", self.research_node)
        graph.add_node("search_articles", self.take_action)
        graph.add_node("article_decisions", self.decision_node)
        graph.add_node("download_articles", self.article_download)
        graph.add_node("paper_analyzer", self.paper_analyzer)
        graph.add_node("combine_analyses", self.combine_analyses)


        graph.add_edge("process_input", "planner")
        graph.add_edge("planner", "researcher")
        graph.add_edge("researcher", "search_articles")
        graph.add_edge("search_articles", "article_decisions")
        graph.add_edge("article_decisions", "download_articles")
        graph.add_edge("download_articles", 'paper_analyzer')
        graph.add_edge("paper_analyzer", "combine_analyses")
        graph.add_edge("combine_analyses", END)
        
        graph.set_entry_point("process_input") ## "llm"
        self.graph = graph.compile(checkpointer=checkpointer)


    def process_input(self, state: AgentState):
        messages = state.get('messages', [])
        # print("MESSAGES")
        # print(messages)
        last_human_index = len(messages) - 1
        for i in reversed(range(len(messages))):
            if isinstance(messages[i], HumanMessage):
                last_human_index = i
                break
        
        return {"last_human_index": last_human_index}
    
    def get_relevant_messages(self, state: AgentState) -> List[AnyMessage]:
        '''
        Don't get tool call messages for AI from history.
        Get state from everything up to the most recent human message
        '''
        messages = state['messages']
        filtered_history = []
        for message in messages:
            if isinstance(message, HumanMessage) and message.content!="":
                filtered_history.append(message)
            elif isinstance(message, AIMessage) and message.content!="" and message.response_metadata['finish_reason']=="stop":
                filtered_history.append(message)
        last_human_index = state['last_human_index']
        return filtered_history[:-1] + messages[last_human_index:]
    
    def plan_node(self, state: AgentState):
        print("PLANNER")
        relevant_messages = self.get_relevant_messages(state)
        messages = [SystemMessage(content=prompts.planner_prompt)] + relevant_messages
        response = self.model.invoke(messages, temperature=self.temperature)
        print(response)
        print()
        return {"systematic_review_outline" : [response]}
    
    def research_node(self, state: AgentState):
        print("RESEARCHER")
        review_plan = state['systematic_review_outline']
        messages = [SystemMessage(content=prompts.research_prompt)] + review_plan
        response = self.model.invoke(messages, temperature=self.temperature)
        print(response)
        print()
        return {"messages" : [response]}
    
    def decision_node(self, state: AgentState):
        print("DECISION-MAKER")
        review_plan = state['systematic_review_outline']
        relevant_messages = self.get_relevant_messages(state)
        messages = [SystemMessage(content=prompts.decision_prompt)] + review_plan + relevant_messages
        response = self.model.invoke(messages, temperature=self.temperature)
        print(response)
        print()
        return {"messages" : [response]}

    def article_download(self, state: AgentState):
        print("DOWNLOAD PAPERS")
        last_message = state["messages"][-1]

        try:
            # Handle different types of content
            if isinstance(last_message.content, str):
                urls = ast.literal_eval(last_message.content)
            else:
                urls = last_message.content

            filenames = []
            for url in urls:
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

                    filenames.append({"paper" : filename})
                    print(f"Successfully downloaded: {filename}")
                    
                except Exception as e:
                    print(f"Error downloading {url}: {str(e)}")
                    continue
            
            # Return AIMessage instead of raw strings
            return {
                "papers": [
                    AIMessage(
                        content=filenames,
                        response_metadata={'finish_reason': 'stop'}
                    )
                ]
            }
            
        except Exception as e:
            # Return error as AIMessage
            return {
                "messages": [
                    AIMessage(
                        content=f"Error processing downloads: {str(e)}",
                        response_metadata={'finish_reason': 'error'}
                    )
                ]
            }

    def paper_analyzer(self, state: AgentState):
        print("ANALYZE PAPERS")
        analyses=""
        for paper in state['papers'][-1].content:
            md_text = pymupdf4llm.to_markdown(f"./{paper['paper']}")
            messages = [
                SystemMessage(content=prompts.analyze_paper_prompt),
                HumanMessage(content=md_text)
            ]
            
            model = ChatOpenAI(model='gpt-4o')
            response = model.invoke(messages, temperature=0.1)
            print(response)
            analyses+=response.content
        return {
            "analyses": [analyses]
        }
    
    def combine_analyses(self, state: AgentState):
        print("COMBINER")
        review_plan = state['systematic_review_outline']
        analyses = state['analyses']
        messages = [SystemMessage(content=prompts.combine_prompt)] + review_plan + analyses
        model = ChatOpenAI(model='gpt-4o')
        response = model.invoke(messages, temperature=0.1)
        print(response)
        print()
        return {"messages" : [response]}

    def take_action(self, state: AgentState):
        ''' Get last message from agent state.
        If we get to this state, the language model wanted to use a tool.
        The tool calls attribute will be attached to message in the Agent State. Can be a list of tool calls.
        Find relevant tool and invoke it, passing in the arguments
        '''
        print("GET SEARCH RESULTS")
        last_message = state["messages"][-1]

        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return {"messages": state['messages']}
            
        results = []
        for t in last_message.tool_calls:
            print(f'Calling: {t}')

            if not t['name'] in self.tools: # check for bad tool name
                print("\n ....bad tool name....")
                result = "bad tool name, retry" # instruct llm to retry if bad
            else:
                # pass in arguments for tool call
                result = self.tools[t['name']].invoke(t['args'])

            # append result as a tool message
            results.append(ToolMessage(tool_call_id = t['id'], name=t['name'], content=str(result)))

        return {"messages" : results} # langgraph adding to state in between iterations
    
if __name__=="__main__":
    connection_kwargs = {"autocommit" : True, "prepare_threshold":0}
    HOST=os.getenv("DB_HOST")
    PORT=os.getenv("DB_PORT")
    USER=os.getenv("DB_USER")
    PASSWORD=os.getenv("DB_PASSWORD")
    DBNAME=os.getenv("DB_NAME")
    # DB_URI=f'postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}?sslmode=disable'
    DB_URI = f'postgresql://{USER}@{HOST}:{PORT}/{DBNAME}?sslmode=disable'
    print(DB_URI)

    papers_tool = AcademicPaperSearchTool()
    # download_tool = PaperDownloaderTool()
    # analysis_tool = PaperAnalysisTool()
    tools = [papers_tool]

    temperature=0.1
    model=ChatOpenAI(model='gpt-4o-mini') # gpt-4o-mini
    # llm = HuggingFaceEndpoint(
    #     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    #     task="text-generation",
    #     max_new_tokens=512,
    # )
    # model = ChatHuggingFace(llm=llm, verbose=False)
    thread_id = "testing"
    ##############
    with Connection.connect(DB_URI, **connection_kwargs) as conn:
        checkpointer=PostgresSaver(conn)
        # checkpointer.setup() # only when the DB is first created
        print(checkpointer)
        agent = Agent(model, tools, checkpointer=checkpointer, temperature=temperature)
        print(agent.graph.get_graph().print_ascii())
        agent_input = {"messages" : [HumanMessage(content="Diffusion model for music generation")]}
        thread_config = {"configurable" : {"thread_id" : thread_id}}
        result = agent.graph.invoke(agent_input, thread_config)
        response=result['messages'][-1].content
        print(response)

    ##############
    # checkpointer = MemorySaver()
    # agent = Agent(model, [], checkpointer=checkpointer, temperature=temperature, system=prompt)
    # agent_input = {"messages" : [HumanMessage(content="Hi, my name is Brad")]}
    # thread_config = {"configurable" : {"thread_id" : thread_id}}
    # result = agent.graph.invoke(agent_input, thread_config)
    # print(result['messages'][-1].content)

