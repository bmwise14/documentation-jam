import gradio as gr
import requests
import json
import pandas as pd
import os
from dotenv import load_dotenv

import prompts
from langchain_openai import ChatOpenAI
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
import psycopg
from agent import Agent
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, ChatMessage
from typing import TypedDict, Annotated, List, Dict, Any
from tools import AcademicPaperSearchTool

_ = load_dotenv()
######################################################
connection_kwargs = {"autocommit" : True, "prepare_threshold":0}
HOST=os.getenv("DB_HOST")
PORT=os.getenv("DB_PORT")
USER=os.getenv("DB_USER")
PASSWORD=os.getenv("DB_PASSWORD")
DBNAME=os.getenv("DB_NAME")
DB_URI = f'postgresql://{USER}@{HOST}:{PORT}/{DBNAME}?sslmode=disable'
print(DB_URI)

model=ChatOpenAI(model='gpt-4o-mini')
# llm = HuggingFaceEndpoint(
#     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
#     task="text-generation",
#     max_new_tokens=512,
# )
# model = ChatHuggingFace(llm=llm, verbose=False)
print(model)
papers_tool = AcademicPaperSearchTool()
tools = [papers_tool]

######################################################
def execute_sql(query: str) -> List[Dict[str, Any]]:
    conn_params = {
        "host": HOST,
        "dbname": DBNAME,
        "user": USER,
        "password": PASSWORD,
        "port": PORT  
    }
    conn = psycopg.connect(**conn_params)
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        results = [dict(zip(columns, row)) for row in rows]        
    finally:
        # Close the cursor and connection
        cursor.close()
        conn.close()

    return results

######################################################
def query_agent(query, chat_history, thread_id=None):
    prompt = prompts.agent_prompt
    temperature=0.1
    ##############
    with Connection.connect(DB_URI, **connection_kwargs) as conn:
        checkpointer=PostgresSaver(conn)
        # checkpointer.setup() # only when the DB is first created
        print(checkpointer)
        agent = Agent(model, tools, checkpointer=checkpointer, temperature=temperature, system=prompt)
        print(agent.graph.get_graph().print_ascii())
        agent_input = {"messages" : [HumanMessage(content=query)]}
        thread_config = {"configurable" : {"thread_id" : thread_id}}
        result = agent.graph.invoke(agent_input, thread_config)
        response=result['messages'][-1].content
    
    chat_history.append((query, response))
    return "", chat_history
    
def get_conversation_history(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    with Connection.connect(DB_URI, **connection_kwargs) as conn:
        checkpoints = PostgresSaver(conn).get(config)
        history = []
        user_message = None
        ai_message = None
        for message in checkpoints['channel_values']['messages']:
            if isinstance(message, HumanMessage) and message.content != "":
                user_message = message.content
            if isinstance(message, AIMessage) and message.content != "" and message.response_metadata['finish_reason'] == "stop":
                ai_message = message.content
            # Add to history only when both user and AI message are available
            if user_message and ai_message:
                history.append((user_message, ai_message))
                user_message, ai_message = None, None  # Reset for the next pair

        return history
    
def summary(thread_id):
    query = f"SELECT DISTINCT(thread_id) FROM checkpoints WHERE thread_id='{thread_id}';"
    print(query)
    res = execute_sql(query)
    print(res)
    if len(res)==0:
        return "thread doesn't exist"
    final_res = []
    for data in res:
        config = {"configurable": {"thread_id": data['thread_id']}}
        with Connection.connect(DB_URI, **connection_kwargs) as conn:
            checkpoints = PostgresSaver(conn).get(config)
            if 'summary' in checkpoints['channel_values']:
                summary = checkpoints['channel_values']['summary']
                return summary
            else:
                return "Error: Unable to fetch summary"

def get_title(thread_id):
    query = f"SELECT DISTINCT(thread_id) FROM checkpoints WHERE thread_id='{thread_id}';"
    res = execute_sql(query)
    if len(res)==0:
        return "thread does not exist"
    
    final_res = []
    for data in res:
        config = {"configurable": {"thread_id": data['thread_id']}}
        with Connection.connect(DB_URI, **connection_kwargs) as conn:
            checkpoints = PostgresSaver(conn).get(config)
            print(checkpoints)
            if 'title' in checkpoints['channel_values']:
                title = checkpoints['channel_values']['title']
                return title
            else:
               return "no title"

def update_thread_info(thread_id):
    title = get_title(thread_id)
    return title

with gr.Blocks(title="LangGraph Agent",) as demo:
    with gr.Row():
        gr.Markdown(
        """
        # LangGraph Agent
        """
    )
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row():
                title_output = gr.Textbox(label="Conversation Title", show_label=True, interactive=False)
            chatbot = gr.Chatbot(label="LangGraph RAG API - Proof of Concept", show_label=False)
            with gr.Row():
                msg = gr.Textbox(lines=1, show_label=False, placeholder="Enter your query and press ENTER", 
                                 container=False, scale=2)
                thread = gr.Textbox(lines=1, placeholder="Keep thread ID to help keep track of the conversation", show_label=False)
                clear_btn = gr.ClearButton([chatbot], value="Clear Chat")
                load_history_btn = gr.Button("Load Conversation History")

            summary_btn = gr.Button("Get Conversation Summary")
            summary_output = gr.Textbox(label="Conversation Summary", show_label=True, interactive=False)

    # Event handlers
    query_event = msg.submit(query_agent, inputs=[msg, chatbot, thread], outputs=[msg, chatbot])
    thread.change(
        update_thread_info,
        inputs=[thread],
        outputs=[title_output]
    )
    summary_btn.click(summary, inputs=thread, outputs=summary_output)
    load_history_btn.click(get_conversation_history, inputs=thread, outputs=chatbot)

demo.launch(share=False,server_name="0.0.0.0",server_port=5001, debug=True)
