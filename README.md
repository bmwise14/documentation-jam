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

<div style="display: flex; align-items: flex-start; gap: 20px;">
    <img src="assets/systematic_review_graph.png" alt="Systematic Review Graph" width="500"/>
    <div>
        <h3>Initial Stages</h3>
        <ul>
            <li><code>_start_</code>: Beginning point of the process</li>
            <li><code>process_input</code>: Initial data processing stage</li>
            <li><code>planner</code>: Strategy development phase</li>
            <li><code>researcher</code>: Research coordination phase</li>
        </ul>
        <h3>Article Management</h3>
        <ul>
            <li><code>search_articles</code>: Article search and identification</li>
            <li><code>article_decisions</code>: Evaluation and selection of articles</li>
            <li><code>download_articles</code>: Retrieval of selected articles</li>
            <li><code>paper_analyzer</code>: In-depth analysis of papers</li>
        </ul>
        <h3>Writing Components</h3>
        <ul>
            <li><code>write_abstract</code>: Abstract composition</li>
            <li><code>write_conclusion</code>: Conclusion development</li>
            <li><code>write_introduction</code>: Introduction creation</li>
            <li><code>write_methods</code>: Methodology documentation</li>
            <li><code>write_references</code>: Reference compilation</li>
            <li><code>write_results</code>: Results documentation</li>
        </ul>
        <h3>Final Stages</h3>
        <ul>
            <li><code>aggregate_paper</code>: Combining all sections</li>
            <li><code>critique_paper</code>: Critical review phase</li>
            <li><code>revise_paper</code>: Revision process</li>
            <li><code>final_draft</code>: Final document preparation</li>
            <li><code>_end_</code>: Process completion</li>
        </ul>
    </div>
</div>
