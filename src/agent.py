from dotenv import load_dotenv
_ = load_dotenv()

import requests
import ast
import operator

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, ChatMessage
from langchain_openai import ChatOpenAI

import os
from uuid import uuid4

import prompts

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from tools import AcademicPaperSearchTool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import pymupdf4llm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import openai
from langgraph.types import Send
from langsmith import traceable
import markdown
from weasyprint import HTML

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
    papers : List[str] # Annotated[List[str], operator.add] ## papers downloaded - should I make this a static List[str]?
    analyses: Annotated[List[Dict], operator.add]  # Store analysis results

    title: str
    abstract : str
    introduction : str
    methods : str
    results : str
    conclusion : str
    references : str

    draft : Annotated[List[str], operator.add]
    revision_num : int
    num_articles : int
    max_revisions : int

@traceable  
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

        graph.add_node("write_abstract", self.write_abstract)
        graph.add_node("write_introduction", self.write_introduction)
        graph.add_node("write_methods", self.write_methods)
        graph.add_node("write_results", self.write_results)
        graph.add_node("write_conclusion", self.write_conclusion)
        graph.add_node("write_references", self.write_references)

        graph.add_node("aggregate_paper", self.aggregator)
        graph.add_node("critique_paper", self.critique)
        graph.add_node("revise_paper", self.paper_reviser)
        graph.add_node("final_draft", self.final_draft)

        ####################################
        graph.add_edge("process_input", "planner")
        graph.add_edge("planner", "researcher")
        graph.add_edge("researcher", "search_articles")
        graph.add_edge("search_articles", "article_decisions")
        graph.add_edge("article_decisions", "download_articles")
        # graph.add_edge("download_articles", 'paper_analyzer')
        graph.add_conditional_edges("download_articles", self.send_articles)

        graph.add_edge("paper_analyzer", "write_abstract")
        graph.add_edge("paper_analyzer", "write_introduction")
        graph.add_edge("paper_analyzer", "write_methods")
        graph.add_edge("paper_analyzer", "write_results")
        graph.add_edge("paper_analyzer", "write_conclusion")
        graph.add_edge("paper_analyzer", "write_references")

        graph.add_edge("write_abstract", "aggregate_paper")
        graph.add_edge("write_introduction", "aggregate_paper")
        graph.add_edge("write_methods", "aggregate_paper")
        graph.add_edge("write_results", "aggregate_paper")
        graph.add_edge("write_conclusion", "aggregate_paper")
        graph.add_edge("write_references", "aggregate_paper")

        graph.add_edge("aggregate_paper", 'critique_paper')
        
        graph.add_conditional_edges(
            "critique_paper", 
            self.exists_action, 
            {"final_draft": "final_draft", 
             "revise": "revise_paper", 
             True: "search_articles"} 
        )

        graph.add_edge("revise_paper", "critique_paper")
        graph.add_edge("final_draft", END)

        graph.set_entry_point("process_input") ## "llm"
        self.graph = graph.compile(checkpointer=checkpointer)


    def process_input(self, state: AgentState):
        num_articles = state.get('num_articles', 2)
        max_revision = 2
        messages = state.get('messages', [])
        # print("MESSAGES")
        # print(messages)
        last_human_index = len(messages) - 1
        for i in reversed(range(len(messages))):
            if isinstance(messages[i], HumanMessage):
                last_human_index = i
                break
        
        return {"last_human_index": last_human_index, "max_revisions" : max_revision, "revision_num" : 1, 'num_articles' : num_articles}
    
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
    
    def decision_node(self, state: AgentState):
        print("DECISION-MAKER")
        review_plan = state['systematic_review_outline']
        relevant_messages = self.get_relevant_messages(state)
        messages = [SystemMessage(content=prompts.decision_prompt.format(n=state["num_articles"]))] + review_plan + relevant_messages
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

                    filenames.append(filename)
                    print(f"Successfully downloaded: {filename}")
                    
                except Exception as e:
                    print(f"Error downloading {url}: {str(e)}")
                    continue
            
            # Return AIMessage instead of raw strings
            # [
            #         AIMessage(
            #             content=filenames,
            #             response_metadata={'finish_reason': 'stop'}
            #         )
            #     ]
            return {
                "papers": filenames
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

    def send_articles(self, state: AgentState):
        print("SEND ARTICLES FILENAMES")
        print(state['papers'])
        return [Send("paper_analyzer", {"paper" : p}) for p in state['papers']]
    
    def paper_analyzer(self, state: AgentState):
        print("ANALYZE PAPER")
        paper = state['paper']
        print(paper)
        try:
            md_text = pymupdf4llm.to_markdown(f"./{paper}")
            messages = [
                    SystemMessage(content=prompts.analyze_paper_prompt),
                    HumanMessage(content=md_text)
                ]
            model = ChatOpenAI(model='gpt-4o')
            response = model.invoke(messages, temperature=0.1)
        except Exception as e:
            response = AIMessage(
                    content=f"Error processing download: {str(e)}",
                    response_metadata={'finish_reason': 'error'}
                    )

        return {"analyses" : [response]}

        # analyses=""
        # for paper in state['papers'][-1].content:
        #     md_text = pymupdf4llm.to_markdown(f"./{paper['paper']}")
        #     messages = [
        #         SystemMessage(content=prompts.analyze_paper_prompt),
        #         HumanMessage(content=md_text)
        #     ]
            
        #     model = ChatOpenAI(model='gpt-4o')
        #     response = model.invoke(messages, temperature=0.1)
        #     print(response)
        #     analyses+=response.content
        # return {
        #     "analyses": [analyses]
        # }

    def write_abstract(self, state: AgentState):
        print("WRITE ABSTRACT")
        review_plan = state['systematic_review_outline']
        analyses = state['analyses']
        messages = [SystemMessage(content=prompts.abstract_prompt)] + review_plan + analyses
        model = ChatOpenAI(model='gpt-4o-mini')
        response = self._make_api_call(model, messages)
        print(response)
        print()
        return {"abstract" : [response]}
    
    def write_introduction(self, state: AgentState):
        print("WRITE INTRODUCTION")
        review_plan = state['systematic_review_outline']
        analyses = state['analyses']
        messages = [SystemMessage(content=prompts.introduction_prompt)] + review_plan + analyses
        model = ChatOpenAI(model='gpt-4o-mini')
        response = self._make_api_call(model, messages)
        print(response)
        print()
        return {"introduction" : [response]}
    
    def write_methods(self, state: AgentState):
        print("WRITE METHODS")
        review_plan = state['systematic_review_outline']
        analyses = state['analyses']
        messages = [SystemMessage(content=prompts.methods_prompt)] + review_plan + analyses
        model = ChatOpenAI(model='gpt-4o-mini')
        response = self._make_api_call(model, messages)
        print(response)
        print()
        return {"methods" : [response]}
    
    def write_results(self, state: AgentState):
        print("WRITE RESULTS")
        review_plan = state['systematic_review_outline']
        analyses = state['analyses']
        messages = [SystemMessage(content=prompts.results_prompt)] + review_plan + analyses
        model = ChatOpenAI(model='gpt-4o-mini')
        response = self._make_api_call(model, messages)
        print(response)
        print()
        return {"results" : [response]}

    def write_conclusion(self, state: AgentState):
        print("WRITE CONCLUSION")
        review_plan = state['systematic_review_outline']
        analyses = state['analyses']
        messages = [SystemMessage(content=prompts.conclusions_prompt)] + review_plan + analyses
        model = ChatOpenAI(model='gpt-4o-mini')
        response = self._make_api_call(model, messages)
        print(response)
        print()
        return {"conclusion" : [response]}
    
    def write_references(self, state: AgentState):
        print("WRITE REFERENCES")
        review_plan = state['systematic_review_outline']
        analyses = state['analyses']
        messages = [SystemMessage(content=prompts.references_prompt)] + review_plan + analyses
        model = ChatOpenAI(model='gpt-4o-mini')
        response = self._make_api_call(model, messages)
        print(response)
        print()
        return {"references" : [response]}

    def aggregator(self, state: AgentState):
        print("AGGREGATE")
        abstract = state['abstract'][-1].content
        introduction = state['introduction'][-1].content
        methods = state['methods'][-1].content
        results = state['results'][-1].content
        conclusion = state['conclusion'][-1].content
        references = state['references'][-1].content

        messages = [
                SystemMessage(content="Make a title for this systematic review based on the abstract. Write it in markdown."),
                HumanMessage(content=abstract)
            ]
        title = self.model.invoke(messages, temperature=0.1).content

        draft = title + "\n\n" + abstract + "\n\n" + introduction + "\n\n" + methods + "\n\n" + results + "\n\n" + conclusion + "\n\n" + references

        first_draft = AIMessage(
                    content=draft,
                    response_metadata={'finish_reason': 'error'}
                    )

        return {"draft" : [first_draft]}
    
    def critique(self, state:AgentState):
        print("CRITIQUE")
        draft = state["draft"][-1].content
        review_plan = state['systematic_review_outline']

        messages = [SystemMessage(content=prompts.critique_draft_prompt)] + review_plan + [draft]
        response = self.model.invoke(messages, temperature=self.temperature)
        print(response)

        # every critique is a call for revision
        return {'messages' : [response], "revision_num": state.get("revision_num", 1) + 1}

    def paper_reviser(self, state: AgentState):
        print("REVISE PAPER")
        critique = state["messages"][-1].content
        draft = state["draft"][-1].content

        messages = [SystemMessage(content=prompts.revise_draft_prompt)] + [critique] + [draft]
        response = self.model.invoke(messages, temperature=self.temperature)
        print(response)

        return {'draft' : [response]}

    def exists_action(self, state: AgentState):
        '''
        Determines whether to continue revising, end, or search for more articles
        based on the critique and revision count
        '''
        print("DECIDING WHETHER TO REVISE, END, or SEARCH AGAIN")    
        
        if state["revision_num"] > state["max_revisions"]:
            return "final_draft"
        
        # # Get the latest critique
        critique = state['messages'][-1]
        print(critique)
        
        # Check if the critique response has any tool calls
        if hasattr(critique, 'tool_calls') and critique.tool_calls:
            # The critique suggests we need more research
            return True
        else:
            # No more research needed, proceed with revision
            return "revise"
    
    def final_draft(self, state: AgentState):
        print("FINAL DRAFT")

        # Convert markdown to HTML
        html = markdown.markdown(state['draft'][-1].content)
        # Generate PDF from HTML
        HTML(string=html).write_pdf("papers/final_draft.pdf")
                
        return {"draft" : state['draft']}

    def _make_api_call(self, model, messages, temperature=0.1):
        @retry(
            retry=retry_if_exception_type(openai.RateLimitError),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            stop=stop_after_attempt(5)
        )
        def _call():
            try:
                return model.invoke(messages, temperature=temperature)
            except openai.RateLimitError as e:
                print(f"Rate limit reached. Waiting before retry... ({e})")
                raise
        return _call()

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
    tools = [papers_tool]

    temperature=0.1
    model=ChatOpenAI(model='gpt-4o-mini') # gpt-4o-mini
    # llm = HuggingFaceEndpoint(
    #     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    #     task="text-generation",
    #     max_new_tokens=512,
    # )
    # model = ChatHuggingFace(llm=llm, verbose=False)
    thread_id = "test21"
    ##############
    with Connection.connect(DB_URI, **connection_kwargs) as conn:
        checkpointer=PostgresSaver(conn)
        # checkpointer.setup() # only when the DB is first created
        print(checkpointer)
        agent = Agent(model, tools, checkpointer=checkpointer, temperature=temperature)
        print(agent.graph.get_graph().print_ascii())
        agent_input = {"messages" : [HumanMessage(content="diffusion models for music generation")], "num_articles" : 3}
        thread_config = {"configurable" : {"thread_id" : thread_id}}
        result = agent.graph.invoke(agent_input, thread_config)
        print("FINAL PAPER")
        paper=result['draft'][-1].content
        print(paper)