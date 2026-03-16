### Langchain Basics
## Setting up ollama with Langchain
# !pip install langchain
# !pip install -U langchain-ollama

from langchain_ollama import ChatOllama 

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2:latest",
    temperature=0.5,
    max_tokens=250
)
response = llm.invoke("Hi, How are you doing ?")
print(response.content)

for chuck in llm.stream("Hi How are you doing ?"):
    print(chuck.content, end="", flush=True)

'''
- Chunk = a word/phrase coming from the model one by one instead of the full sentence at once
- llm.stream() -> does not return the full response at once, it streams the output gradually
- flush=True means: Print the output immediately without waiting for the buffer to fill.
- temperature = Controls randomness in the model's responses.
- Lower values (close to 0) → more deterministic, focused answers. Good for tasks where accuracy and reliability matters. 
- Higher values (close to 1) → more creative, varied outputs. Responses vary more between runs.
- 0.5 is a balanced middle ground.
- max_tokens: the maximum number of tokens the LLM can generate in its output.
- 1 token ≈ 0.75 words in English, So 250 tokens ≈ ~ 180 words
'''

## Understanding Prompt Template
from langchain_core.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template("Hi how are you, I am {name}")
print(llm.invoke(prompt_template.invoke({'name': 'Kiran'})).content)

"""
- Template → Fill values → Send to LLM → Get text output
- A prompt template is a reusable prompt with placeholders that allows dynamic data injection before sending it to an LLM.
- A prompt is a fixed instruction sent to an LLM, while a prompt template is a reusable, parameterized prompt that allows dynamic 
values to be injected at runtime.

# Prompt engineering best practices
Memory Trick 🧠: CLEAR → Context, Limits, Examples, Audience, Response format
Be specific – Clearly state what you want to avoid ambiguous responses.
Provide context – Add background so the model understands the scenario.
Define a role – Assign a role to guide the style and depth of the response.
Structure the output – Ask for bullets, steps, tables, or sections.
Use prompt templates – Create reusable prompts with dynamic placeholders.
Use examples (few-shot) – Show sample inputs and outputs to improve accuracy.
Set constraints – Limit length, tone, or format of the response.
Ask step-by-step – Request logical or sequential explanations when needed.
Keep prompts focused – One task per prompt gives better results.
Iterate and refine – Improve prompts based on model output.
Mention edge cases – Explicitly ask for negative or failure scenarios.
Control response style – Specify tone (interview-ready, simple, technical).

Eg. You are a Senior QA Automation Engineer with strong LLM testing experience.
I am building a pytest-based API automation framework where LLM-generated responses are evaluated for quality instead of exact matches.

# Common Mistakes with Prompt Templates (QA / LLM Testing)
Vague placeholders
Missing input validation
Hardcoded values
Overloaded templates
Undefined output format
Edge cases ignored
Prompt drift
No prompt versioning
Lack of determinism controls
Business logic inside prompts
No negative prompt testing
Token limit issues
No localization handling
Conflicting system vs user instructions
No prompt regression testing

# Prompt Template Test Cases (QA / LLM Testing)
1. Placeholder Validation
Verify all required placeholders are present
Verify no unused placeholders exist

2. Happy Path
Valid input values produce expected response
Output follows defined format

3. Missing Input
Omit required placeholder
Validate graceful error or fallback behavior

4. Empty Input
Pass empty string to placeholder
Verify response handling and clarity

5. Invalid Input Type
Pass number/list instead of string
Validate error handling or safe response

6. Boundary Input Length
Very long input text
Verify no truncation or broken output

7. Special Characters
Input with symbols, emojis, SQL/HTML
Ensure prompt is not corrupted

8. Output Format Validation
Validate JSON / bullet / step structure
Ensure mandatory sections exist

9. Deterministic Output
Run prompt multiple times
Validate consistency (temperature fixed)

10. Negative Prompt
Ambiguous or conflicting instructions
Verify safe and meaningful response

11. Prompt Injection
Attempt instruction override
Verify system rules are enforced

12. Token Limit
Prompt near token limit
Validate truncation handling

13. Localization
Non-English input
Validate grammar and intent preservation

14. Regression
Compare output before and after prompt changes
Detect behavior drift

15. Model Version Compatibility
Execute prompt across model versions
Validate output stability

Interview One-Liner 🎯
Prompt template test cases cover input validation, output structure, determinism, security, regression, and model compatibility.

"""

# Langchain Chaining mechanism with parsers
from langchain_core.output_parsers import JsonOutputParser
prompt_template = PromptTemplate.from_template("What is the role of {type} in software Testing ? give me the response in JSON format ONLY.")
chain = prompt_template | llm | JsonOutputParser()
result = chain.invoke({'type': 'AI'})
print(result)

"""
JsonOutputParser forces the response to be valid JSON, like to convert free-text LLM output into structured, testable data.
| operator builds a pipeline (Prompt → LLM → Parser)
invoke() fills the placeholder and executes the chain
Result is a Python dictionary, not plain text

# What is a LangChain Chain?
A chain is a pipeline of connected components.
Each component's output becomes the next component's input.
Used to orchestrate prompts, models, parsers, and tools.
Simple Flow: Input → PromptTemplate → LLM → OutputParser → Result

# Why Chains Matter (QA View)
To build reusable, maintainable, and testable LLM workflows.
A chain adds structure, validation, and post-processing to LLM calls.
If one step fails in chain, the chain execution fails at that step, making error isolation easier.
LCEL: LangChain Expression Language used to declaratively compose chains.
Chain follows a fixed flow; agent dynamically decides next actions.
Chains can be used in CI/CD as they are ideal for automated prompt and LLM regression testing.

# LangChain Chain - Test Cases (Short, QA Notes)
Input validation
Missing / invalid inputs
Prompt formatting correctness
Placeholder substitution
Happy-path execution
Output format validation
Output parser failure handling
Deterministic output check
Multiple-run consistency
Token limit handling
Error propagation between steps
Model version compatibility
Regression after prompt changes
Performance / response time
Security / prompt injection
Integration with external tools
"""

### Working with External Docs
# Working with documents in python
# !pip install langchain-community

import os
from langchain_community.document_loaders import (
    UnstructuredPDFLoader # reads PDF files and extracts text into LangChain Document objects.
)

docs_folder = "./Docs"
documents = []

for filename in os.listdir(docs_folder):  # Loops through every file inside ./Docs
    filepath = os.path.join(docs_folder, filename) # Creates the full path for each PDF file
    loader = UnstructuredPDFLoader(filepath)   # Initializes the PDF loader for the current file
    documents.extend(loader.load()) # Loads the document and adds it to the documents list

    # documents contains text content + metadata for all PDFs in the ./Docs folder.

# Chunking the text into smaller parts
from langchain_text_splitters import RecursiveCharacterTextSplitter # Splits text into smaller chunks based on character count and overlap

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50) # Configures the text splitter to create chunks of 300 characters 
# with 50 characters overlapping between chunks, Overlap helps retain context between chunks
chunks = splitter.split_documents(documents) # Splits all loaded documents into smaller chunks

for i, chunk in enumerate(chunks):          # Loops through each chunk and prints its content
    print(f"------ Chunck {i + 1} ------ ") # Prints the chunk number
    print(chunk.page_content)               # Prints the actual text content of the chunk
    print("------------------------------")

"""
# Why Chunking Is Important (QA / Interview)
    Prevents token limit issues
    Improves retrieval accuracy in RAG
    Maintains context using overlap
    Enables faster and more reliable LLM responses

## Chunking Test Cases
    1. Chunk size limit validation
    2. Chunk overlap validation
    3. No data loss between chunks
    4. No duplicate content beyond overlap
    5. Order of chunks preserved
    6. Metadata retained per chunk
    7. Empty / null document handling
    8. Very small document handling
    9. Very large document handling
    10. Special characters & formatting
    11. Page boundary handling (PDFs)
    12. Performance on large datasets
    13. Token limit compliance
    14. Regression after chunk config change

## Best Chunk Size for RAG
    * Text documents: 300-500 characters
    * Technical / dense content: 200-300 characters
    * FAQs / simple text: 500-800 characters
    * Chunk overlap: 10-20% of chunk size
    * Goal: Balance context retention and retrieval accuracy
    * The best chunk size for RAG typically ranges from 200-500 characters with 10-20% overlap, depending on content density.
"""

### Embedding and Vector
"""
Embedding: It refers to the process of converting complex data (like words or categories) into dense numerical vectors 
    that capture their meaning and relationships and which LLM understand.
Vector: A vector is a numerical representation of data used to measure similarity in AI systems.
Vector Store: A specialized database designed to store and manage these high-dimensional vectors efficiently,
    enabling fast similarity searches and retrieval based on vector comparisons.
"""

#!pip install chromadb

from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

embedding = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(chunks, embedding, persist_directory="qa_db")
retriever = db.as_retriever() 

"""
OllamaEmbeddings- Converts text into numerical vectors using Ollama models
Chroma: A vector database used to store and search embeddings efficiently
Creates a Chroma vector store from document chunks using Ollama embeddings
Converts the Chroma vector store into a retriever object for querying
"""

## RetrievalQA Chain
"""
RetrievalQA is a powerful chain in LangChain that combines two key components
    RetrievalQA = Retriever + Language Model (LLM)
It enables a system to:
- Retrieve relevant information from a document store (like a vector database).
- Answer a user's question using that information via a language model.
- Retriver: Finds the most relevant document chunks based on the user's question
- LLM: Reads those chunks and generates a natural-language answer
"""
"""
# Whole flow for rag systems
question -> read documents -> UnstructuredPDFLoader -> RecursiveCharacterTextSplitter -> Converted to Chunks -> OllamaEmbeddings 
  -> Converted to Numerical Vectors -> Chroma -> Stored vectors in Chroma Vector Store -> RetrievalQA 
      -> Retrived relevant chunks based on the questions -> LLM -> Answer
"""
from langchain_classic.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm = llm, retriever=retriever) 
question = "Explain the system architecture of Netflix"
print(qa_chain.run(question))

### AI Agents & Tools
"""
AI Agent: An LLM-driven system that autonomously decides actions using tools.
Tool: A callable function that performs a defined task for an agent or LLM.
An LLM answers a question, while an agent completes a goal by planning and using tools.
Agent → uses LLM (reasoning & planning)
Agent → uses Tools (actions)
"""
#%pip install --upgrade --quiet  wikipedia
from langchain_classic.agents import initialize_agent, AgentType, load_tools

tools = load_tools(["wikipedia"], llm=llm)                          # Loads the Wikipedia tool for the agent to use
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
for chunk in agent.stream("What is Tools in AI Agent ?"):
    print(chunk)

"""
# Zero-Shot React Description mode:
- “Zero-Shot” → The agent doesn't need examples; it decides what to do based on the description of tools.
- “React” → It reasons step by step, deciding when to call a tool and when to answer directly.
- “Description” → It relies on the tool's description to know how to use it.
"""

# Building a Simple Agent
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
# HumanMessage tells the LLM “this message is from a human user.

@tool
def add_numbers(a: int, b: int) -> int:
    "Adding two given numbers and returns a result" 
    # This short string describes what the tool does. Agents use this description to decide when and how to call the tool.
    return int(a) + int(b)

@tool
def substract_numbers(a: int, b: int) -> int:
    "Subtract two given numbers and returns a result"
    return int(a) - int(b)

@tool
def multiply_numbers(a: int, b: int) -> int:
    "Multiply two given numbers and returns a result"
    return int(a) * int(b)

tools = [add_numbers, substract_numbers, multiply_numbers]

# Creating an AI Agent
agent = create_agent(tools= tools, model=llm)

query = "Whats the 100 less of 50 and what will be the sum of 20 and 90 and how about double of the above answer"
result = agent.invoke({"messages": [HumanMessage(content=query)]})
print(result["messages"][-1].content)

#- HumanMessage(content=query) wraps your text into a chat message format that the agent understands.
#- this line extracts and prints the final response text from your agent’s output.

## Playwright Toolkit
"""
Playwright Toolkit is a set of tools that allows AI agents to interact with web browsers programmatically.
It enables agents to perform actions like navigating web pages, clicking buttons, filling forms, and extracting data from websites.
This toolkit is useful for automating web-based tasks, testing web applications, and scraping data from the internet.
* I use pytest as the test runner and Playwright for browser automation. 
* On top of that, I integrate an AI agent using LangChain and a local LLM. 
* The agent can understand natural language, decide which browser actions to perform, navigate the page, and extract data. 
* Pytest assertions then validate the AI output, so it works like a normal automated test but with added intelligence and better 
resilience to UI changes.
* This approach combines traditional test automation with AI-driven decision-making, making tests more adaptable and easier to maintain.
"""

## MCP (Model Context Protocol) Server
"""
- MCP is a standard protocol that lets LLMs securely connect to external tools, data sources, and services in a consistent way.
- Instead of building custom integrations for every tool, MCP provides a common interface that models can use to access external context.
- MCP standardizes tool access but does not perform reasoning or execution.
- Q: Is MCP a model? --> No, it's a protocol, not a model.
- Q: Does MCP replace agents? --> No, agents use MCP to access tools safely.
- Q: Does MCP reduce hallucination? --> Indirectly, by providing grounded and controlled context.
- Function calling is a model-specific way to invoke tools, 
    while MCP is a standardized, secure protocol that decouples tools from models and enables scalable agent architectures.

# Why MCP Was Introduced
Tool integrations were hard-coded and inconsistent
Security and permission control was weak
Tool reuse across models was difficult
Debugging and testing were complex

# MCP Architecture
User → Agent → LLM
              ↓
             MCP
              ↓
        Tools / Data / APIs

#Role Breakdown
Agent → decides what to do
LLM → reasons and selects tools
MCP → securely connects to tools
Tools → perform actions

## MCP Test Cases (QA Focused)
# Functional Test Cases
    * Tool discovery via MCP
    * Correct tool metadata exposure
    * Valid input schema enforcement
    * Correct output schema parsing
    * Tool invocation success
    * Multiple tool handling
---
# Negative Test Cases
    * Invalid tool input rejection
    * Unauthorized tool access
    * Tool timeout handling
    * Tool unavailable scenarios
    * Malformed MCP response
---
# Security Test Cases
    * Permission enforcement
    * Prompt injection attempts
    * Data leakage prevention
    * Cross-tool access isolation
---
# Integration & Regression
    * Agent → MCP → Tool flow validation
    * Tool version change compatibility
    * Model version compatibility
    * Backward compatibility tests
---
# Performance & Reliability
    * Tool response latency
    * Concurrent MCP requests
    * Retry and fallback behavior

**A-L-M-T**
    * **Agent** decides
    * **LLM** reasons
    * **MCP** connects
    * **Tool** executes
"""
# - This code creates a simple MCP server using FastMCP. 
# - The server exposes an add_numbers function as a tool that can be called by an AI client. 
# - When the server runs, it registers this tool under the name simple-calculator. 
# - The configuration file tells the client how to start this MCP server by running the Python script, 
#   so the AI can connect to it and use the calculator tool to add two numbers.
# # Question: What is the sum of 50 and 70

from mcp.server.fastmcp import FastMCP
mcp = FastMCP("simple-calculator") # Initializes an MCP server with the name "simple-calculator"
@mcp.tool() # Registers the function as a tool in the MCP server
def add_numbers(a: int, b: int) -> int:
    """Adding two numbers and return the results"""
    return int(a) + int(b)

if __name__ == "__main__":
    mcp.run()
# - This block ensures that mcp.run() is executed only when the Python file is run directly, not when it is imported into another file.
# Content of cloude_desktop_config: 
{
    "mcpServers": {
        "simple-calculator": {
        "command": "python",
        "args": ["C:\\Users\\Lenovo\\Downloads\\MCP_Server\\simple-calculator.py"]
        }
    }
}
"""
## FastMCP: It is a lightweight, developer-friendly implementation of MCP.
It allows you to quickly set up an MCP server to expose tools for AI agents to use.

# What FastMCP Provides
    Easy tool registration
    Schema-based inputs/outputs
    Built-in request handling
    Faster development & testing

### FastMCP Test Cases (QA Focused, Short)
# Functional
1. Tool registration validation
2. Tool discovery via MCP
3. Input schema validation
4. Output schema validation
5. Successful tool invocation
6. Multiple tool handling

# Negative
7. Invalid input rejection
8. Missing required fields
9. Unsupported tool request
10. Tool execution failure handling

# Security
11. Permission enforcement
12. Unauthorized tool access
13. Prompt injection attempts
14. Data leakage prevention

# Integration
15. Agent → MCP → FastMCP flow
16. LLM compatibility testing
17. Backward compatibility

# Performance & Reliability
18. Tool response latency
19. Concurrent requests handling
20. Retry & timeout behavior
"""