# Systematic Review of Scientific Articles

## Overview 
An advanced academic paper review system that automates the creation of systematic literature reviews. This solution uses a directed graph architecture to orchestrate a complex workflow, transforming research topics into comprehensive review papers through autonomous planning, research, analysis, and writing stages.

• Use Generative AI to compile papers from a given topic and get a systematic overview of their contents. 

• Find similarities and dissimilarities in the literature

• Be able to gain understanding information from various research domains, through meta-analysis or systematic review. 

## Implementation 
Implement a directed graph workflow that orchestrates systematic review generation through sequential stages of planning, research, paper selection, content analysis, parallel section writing, automated critique, and revision cycles, using LangGraph state management for content generation and Semantic Scholar API for paper retrieval.


## Overview of Components

<img src="assets/systematic_review_graph.png" alt="drawing" width="400" style="display: block; margin-left: auto; margin-right: auto;"/>

1. **Initial Stages**
- `_start_`: Beginning point of the process
- `process_input`: Initial data processing stage
- `planner`: Strategy development phase
- `researcher`: Research coordination phase

2. **Article Management**
- `search_articles`: Article search and identification
- `article_decisions`: Evaluation and selection of articles
- `download_articles`: Retrieval of selected articles
- `paper_analyzer`: In-depth analysis of papers

3. **Writing Components**
- `write_abstract`: Abstract composition
- `write_conclusion`: Conclusion development
- `write_introduction`: Introduction creation
- `write_methods`: Methodology documentation
- `write_references`: Reference compilation
- `write_results`: Results documentation

4. **Final Stages**
- `aggregate_paper`: Combining all sections
- `critique_paper`: Critical review phase
- `revise_paper`: Revision process
- `final_draft`: Final document preparation
- `_end_`: Process completion

## Getting Started

### Optional Keys for .env
```
OPENAI_API_KEY="<YOUR_KEY>"
LLAMA_INFERENCE_KEY="<YOUR_KEY>"
```

If using PostgresSaver(), otherwise MemorySaver()
```
DB_HOST="localhost" # whatever IP
DB_PORT="<PORT>"
DB_USER="<POSTGRES_USER>"
DB_PASSWORD="<PASSWORD>" # if needed
DB_NAME="<DATABASE_NAME>"
```

For Traceability
```
LANGCHAIN_API_KEY="<LANGSMITH_KEY>" # if needed
LANGCHAIN_TRACING_V2=true # if needed
```

### Requirements
```
pip install -r minimal_reqs.txt
```

### Run a test
```
python agent.py
```

### Run on Your Own
```python
DB_URI = f'postgresql://{USER}@{HOST}:{PORT}/{DBNAME}?sslmode=disable'
papers_tool = AcademicPaperSearchTool()
tools = [papers_tool]

temperature=0.1
model=ChatOpenAI(model='gpt-4o-mini') # gpt-4o-mini
thread_id = "test_thread"
topcic = "diffusion models for music generation"
##############
with Connection.connect(DB_URI, **connection_kwargs) as conn:
    checkpointer=PostgresSaver(conn)
    # checkpointer.setup() # only when the DB is first created
    print(checkpointer)
    agent = Agent(model, tools, checkpointer=checkpointer, temperature=temperature)
    print(agent.graph.get_graph().print_ascii())
    agent_input = {"messages" : [HumanMessage(content=topic)], "num_articles" : 8}
    thread_config = {"configurable" : {"thread_id" : thread_id}}
    result = agent.graph.invoke(agent_input, thread_config)
    print("FINAL PAPER")
    paper=result['draft'][-1].content
    print(paper)
```