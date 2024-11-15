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
from tools import AcademicPaperSearchTool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


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
    summary: str
    last_human_index : int
    title: str
    
class Agent:
    def __init__(self, model, tools, checkpointer, temperature=0.1, system=""):
        self.system = system
        self.temperature=temperature
        self.tools = {t.name: t for t in tools} if tools else {}
        self.model = model.bind_tools(tools) if tools else model
        
        graph = StateGraph(AgentState)
        graph.add_node("process_input", self.process_input)
        graph.add_node("llm", self.call_openai)
        graph.add_node("summarize", self.summarize_conversation)
        graph.add_node("paper_analyzer", self.analyze_paper)

        graph.add_edge("process_input", "llm")
        
        if tools:
            graph.add_node("action", self.take_action)
            graph.add_conditional_edges(
              "llm", ## where the node starts
              self.exists_action, ## function to determine where to go after LLM executes
              {True: "action", False : "summarize"} # map response to the function to next node to go to 
            )
            graph.add_edge("action", "paper_analyzer")
            graph.add_edge("paper_analyzer", "llm")
        else:
            graph.add_edge('llm', 'summarize')
            
        graph.add_edge("summarize", END)
        
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
                
        # Check if a title already exists
        if not state.get('title'):
            # Find the first human message
            first_human_message = next((msg for msg in messages if isinstance(msg, HumanMessage)), None)
            
            if first_human_message:
                # Create a title based on the first human message
                title_query = [
                    SystemMessage(content="Make a short, concise title based on the following query:"),
                    first_human_message
                ]
                title_response = self.model.invoke(title_query, temperature=0.01)
                title = title_response.content
            else:
                title = "New Conversation"
        else:
            # If a title already exists, keep it
            title = state['title']
        
        return {"last_human_index": last_human_index, "title" : title}

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
        
    def summarize_conversation(self, state: AgentState):
        messages = state['messages']
        existing_summary = state.get("summary", "")

        if existing_summary:
            summary_prompt = (
                f"""This is a summary of the content of the conversation to date: {existing_summary}
                Extend the summary by taking into account any unseen messages above.
                Prioritize gathering facts of the conversation that you can reference later for long-term memory:"""
            )
        else:
            summary_prompt = (
                "Create a summary of the content of the conversation above. "
                "Prioritize gathering facts of the conversation that you can reference later for long-term memory:"
            )

        summary_messages = messages + [HumanMessage(content=summary_prompt)]

        try:
            summary_response = self.model.invoke(summary_messages, temperature=0.01)
        except Exception as e:
            summary_messages = [HumanMessage(content="Too much context from all the messages before")] + [HumanMessage(content=summary_prompt)]
            # print(summary_messages)
            summary_response = self.model.invoke(summary_messages, temperature=0.01)
        
        return {
            "summary": summary_response.content,
            "messages": [messages[-1]],  # Only keep the last message
            "last_human_index": 0
        }
    
    def call_openai(self, state: AgentState):
        '''All nodes and edges will take this in'''
        relevant_messages = self.get_relevant_messages(state) # get Human/AI history and only all other messages since the last human message
        
        messages = [SystemMessage(content=self.system)] + relevant_messages
        print("FINAL MESSAGES FOR AGENT:", messages)
        try:
            print("TRY MODEL INVOCATION")
            response = self.model.invoke(messages, temperature=self.temperature)
            print(response)
            return {"messages" : [response]}
        except Exception as e:
            print("CONTEXT TOO BIG")
            finality_message = [HumanMessage(content="Please Reply: 'I retrieved too much information in my searches and couldn't fit it in my context window. Try again or try tweaking your question to make the search space smaller. I apologize for the inconvenience.'")]
            response = self.model.invoke(finality_message, temperature=0.01)
            return {"messages" : [response]}
        
    def analyze_paper(self, state: AgentState):
        # last_message = state["messages"][-1]
        # print(last_message)
        messages = state['messages']
        last_human_index = state['last_human_index']
        last_messages = messages[last_human_index:]

        messages = [SystemMessage(content=prompts.paper_prompt)] + last_messages

        print(messages)
        llmmodel=ChatOpenAI(model='gpt-4o')
        try:
            print("INVOKE PAPER ANALYZER")
            response = llmmodel.invoke(messages, temperature=self.temperature)
            print(response)
            return {"messages" : [response]}
        except Exception as e:
            print("CONTEXT TOO BIG")
            finality_message = [HumanMessage(content="Please Reply: 'I retrieved too much information in my searches and couldn't fit it in my context window. Try again or try tweaking your question to make the search space smaller. I apologize for the inconvenience.'")]
            response = self.model.invoke(finality_message, temperature=0.01)
            print(response)
            return {"messages" : [response]}

    def take_action(self, state: AgentState):
        ''' Get last message from agent state.
        If we get to this state, the language model wanted to use a tool.
        The tool calls attribute will be attached to message in the Agent State. Can be a list of tool calls.
        Find relevant tool and invoke it, passing in the arguments
        '''
        last_message = state["messages"][-1]
        # if not isinstance(last_message, AnyMessage) or not last_message.tool_calls: # AIMessage
        #     return {"messages": state['messages']}
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
    
    def exists_action(self, state:AgentState):
        '''
        takes in result after the LLM is called and return True/False if no tool calls vs calls exist. 
        Should we take an action or should we not?
        '''
        if not self.tools:
            return False
        
        result = state['messages'][-1]
        return len(result.tool_calls) > 0
    
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

    prompt = prompts.agent_prompt
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
        agent = Agent(model, tools, checkpointer=checkpointer, temperature=temperature, system=prompt)
        print(agent.graph.get_graph().print_ascii())
        agent_input = {"messages" : [HumanMessage(content="Search 1 Attention and Transformers article. Give me details")]}
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

