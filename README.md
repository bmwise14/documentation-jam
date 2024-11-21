# Systematic Review of Scientific Articles

## Overview 
An advanced academic paper review system that automates the creation of systematic literature reviews. This solution uses a directed graph architecture to orchestrate a complex workflow, transforming research topics into comprehensive review papers through autonomous planning, research, analysis, and writing stages.

## Implementation 
Implement a directed graph workflow that orchestrates systematic review generation through sequential stages of planning, research, paper selection, content analysis, parallel section writing, automated critique, and revision cycles, using GPT-4o for content generation and Semantic Scholar API for paper retrieval.


## What's the Goal?
• Use Generative AI to compile papers from a given topic and get a systematic overview of their contents. 

• Find similarities and dissimilarities in the literature?

• Be able to gain understanding information from various research domains, through meta-analysis or systematic review. 

## Overview of Components

| ![Systematic Review Graph](assets/systematic_review_graph.png) | **Initial Stages**<br>- `_start_`: Beginning point of the process<br>- `process_input`: Initial data processing stage<br>- `planner`: Strategy development phase<br>- `researcher`: Research coordination phase<br><br>**Article Management**<br>- `search_articles`: Article search and identification<br>- `article_decisions`: Evaluation and selection of articles<br>- `download_articles`: Retrieval of selected articles<br>- `paper_analyzer`: In-depth analysis of papers<br><br>**Writing Components**<br>- `write_abstract`: Abstract composition<br>- `write_conclusion`: Conclusion development<br>- `write_introduction`: Introduction creation<br>- `write_methods`: Methodology documentation<br>- `write_references`: Reference compilation<br>- `write_results`: Results documentation<br><br>**Final Stages**<br>- `aggregate_paper`: Combining all sections<br>- `critique_paper`: Critical review phase<br>- `revise_paper`: Revision process<br>- `final_draft`: Final document preparation<br>- `_end_`: Process completion |
|---|---|
