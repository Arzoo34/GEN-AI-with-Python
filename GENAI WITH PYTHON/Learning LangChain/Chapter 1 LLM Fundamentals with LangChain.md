#  INTRODUCTION TO LANGCHAIN

> [!info]
> **LangChain** is an open source framework that simplifies building applications using LLMs.

- Helps developers connect LLMs with external data, tools and workflows.
- Available in both **Python** and **JavaScript**.
- Simplifies chaining LLMs together for reusable and efficient workflows.
- Streamlines the process of building LLM-powered applications.

---

#  KEY COMPONENTS

## 1. Chain

Define a sequence of steps where each step uses an LLM, processes data or calls tools.

- **Simple Chains:** Use one step.
- **Multi-step Chains:** Combine multiple actions.

---

## 2. Prompt Management

Helps design and manage prompts using templates, making it easier to control:

- Input
- Output formats
- Model behavior

---

## 3. Agents

LLM-driven components that decide which action to take, such as:

- Calling APIs
- Using tools
- Performing actions based on input and predefined capabilities

---

## 4. Vector Database

Stores data as vectors to enable similarity search, helping retrieve relevant information for tasks such as:

- Document Search
- RAG (Retrieval-Augmented Generation)

---

## 5. Models

Supports multiple LLMs like:

- OpenAI
- Hugging Face
- Others

Allowing flexibility in choosing the best model.

---

#  IMPLEMENTATION

## Step 1: Install the Dependencies

We will install all the required dependencies for our model.

### **langchain**

The core LangChain framework (chains, prompts, tools, memory, etc.).

### **langchain-google-genai**

LangChain integration for accessing Google Gemini models through the Gemini API.

```bash
pip install langchain langchain-google-genai
```

---

## Step 2: Import Libraries

We will import all the required libraries.

### **ChatGoogleGenerativeAI**

Enables interaction with Google Gemini models through LangChain.

### **PromptTemplate**

Defines structured prompts with placeholders.

### **StrOutputParser**

Ensures model response is returned as clean string text.

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

---

## Step 4: Initialize the Gemini Model

- **model:** Specifies which Gemini model to use.
- **temperature = 0.7:** Controls creativity (0 = more deterministic, 1 = more creative).
- **google_api_key = api_key:** Authenticates access to Gemini.

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=api_key
)
```

---

## Step 5: Run a Simple Prompt

We will check by running a simple prompt.

- **.invoke():** Sends prompt to the LLM and returns text output.

```python
prompt = "Suggest me a skill that is in demand?"

response = llm.invoke(prompt)

print("Suggested Skill:\n", response)
```

---
---
tags: [LangChain, LLM, AI]
---

# Prompt Template

> [!abstract]
> **Used in LLM interface.**

Uses Python's **f-string** syntax where placeholders are written as:

```python title="Placeholder Syntax"
{variable_name}
```

---

### Syntax

```python title="PromptTemplate Example"
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("""
Answer the question based on the context below.
Context: {context}
Question: {question}
Answer:
""")

prompt = template.invoke({
    "context": "...",
    "question": "Which model providers offer LLMs?"
})
```

---

# ChatPromptTemplate

> [!abstract]
> **Used with chat model interface.**

Composed from a list of message templates, each having:

| Role | Purpose |
|------|----------|
| System | Gives instructions |
| Human | User input |
| AI | Model response (optional) |

```python title="ChatPromptTemplate Example"
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "Answer the question based on the context below."),
    ("human", "Context: {context}"),
    ("human", "Question: {question}"),
])

prompt = template.invoke({
    "context": "...",
    "question": "..."
})
```

---

# LLM Chains

> [!info]
> An LLM Chain is a structured sequence of connected components that use LLMs alongside other tools and data sources to perform complex tasks more efficiently.

---

## Features

| Component | Description |
|-----------|-------------|
| Prompt Templates | Structure prompts |
| Memory | Retains conversation history |
| Retrieval | Fetches external knowledge |
| Agents | Makes dynamic decisions |
| Vector Store | Stores embeddings |
| APIs | Integrates external services |

---

## Architecture

![[Pasted image 20260703140644.png]]

---

# Key Components of LLM Chains

> [!summary]

### 1. LLMs

The central generative model trained on extensive datasets to understand and generate human-like language.

---

### 2. Prompt Templates

Define how user inputs are formatted and structured to guide LLMs effectively.

---

### 3. Chains

Sequence of actions where each step involves querying an LLM, transforming data or interacting with external tools.

---

### 4. Memory Management

Allows the LLM chain to retain context across interactions, useful for conversational agents requiring coherence over multiple turns.

---

### 5. Data Retrieval (Retrievers) and Vector Stores

External databases or vector stores enable Retrieval-Augmented Generation (RAG).

---

### 6. Agents

Autonomous components that decide dynamically which tasks to perform.

---

### 7. Output Parsers and Postprocessors

Components that format, filter or enrich the LLM's raw output.

---

### 8. Error Handlers and Quality Controllers

Modules ensuring robustness by handling runtime errors and validating outputs.

---

# Working of LLM Chain

```text
User Query
     │
     ▼
Input Processing
     │
     ▼
Vector Embedding & Retrieval
     │
     ▼
Prompt Generation
     │
     ▼
LLM Invocation
     │
     ▼
Post Processing
     │
     ▼
Final Response
```

---

## Workflow

| Step | Description |
|------|-------------|
| User Query Input | User submits a query |
| Input Processing | Cleans and transforms input |
| Vector Retrieval | Retrieves relevant documents |
| Prompt Generation | Creates enriched prompt |
| LLM Invocation | Generates response |
| Postprocessing | Formats and validates output |
| Response Delivery | Returns final response |

---

# LLM Chain Example

## Code

```python
import textwrap
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

os.environ["OPENAI_API_KEY"] = "openai_api_key"

llm = OpenAI(temperature=0.7)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short discription about {topic}."
)

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run("LLM Chains")

print("\n--------------Generated Response------------\n")
print(response)
print("\n---------------------------------------------\n")
```

> [!note]
> This example creates a simple **LLM Chain** using an **OpenAI** model and a **PromptTemplate**. The chain takes a topic as input and generates a short description.

---

# Output

```
LLM-Chain
```

---

# Types of LLM Chains

Let's see the types of LLM Chains.

---

## 1. Simple Sequential Chains

These are the most straightforward type where a single LLM call or a series of LLM calls happen one after the other.

Each step takes input from the user or the output of the previous step and produces output consumed by the next.

Useful for basic workflows such as question answering with prompt templates or text summarization.

---

## 2. Multi-Step Chains (Complex Sequential Chains)

These chains involve multiple LLM calls sequenced to address subtasks stepwise.

They break a large or complex problem into smaller chunks that are handled independently.

Common in tasks like document analysis where text chunking, retrieval and summarization happen in stages.

Also called **"prompt chaining"** or **"pipeline chaining"** where intermediate outputs feed subsequent prompts.

---

## 3. Router Chains

Router chains have a controlling or routing mechanism that dynamically decides which sub-chain or model to invoke based on input characteristics.

Useful when different inputs require different processing pipelines or models.

**Example:** An input classifier routes queries to specialized expert LLM sub-chains like legal, medical or technical assistants.

---

## 4. Agent-Based Chains

These use intelligent agents that autonomously decide on actions, calls to LLMs and interactions with external APIs or tools.

Agents can dynamically split tasks, call multiple LLMs, query databases or invoke other services in an adaptive fashion.

Agents offer reasoning, decision-making and orchestration capabilities, enabling flexible and context-aware behavior.

Useful for complex applications like conversational AI with tool usage, automated research assistants or multi-modal input processing.

---

## 5. Retrieval-Augmented Generation (RAG) Chains

These chains enhance the LLM's understanding and output quality by integrating retrieval from external knowledge bases or vector stores.

The chain converts a query into a vector embedding, fetches relevant documents and includes them in the prompt for generation.

Improves factual accuracy and domain adaptation by grounding LLMs in real, up-to-date data.

Widely used in semantic search, question answering and knowledge-intensive tasks.

---

## 6. Chain of Thought (CoT) / Reasoning Chains

These chains focus on eliciting explicit reasoning or intermediate steps from the LLM.

The model is prompted to **"think step by step"** or generate intermediate reasoning steps before the final answer.

Leads to improved performance on complex problems like math, logic or multi-step instructions.

Can be manual (prompt engineering) or autonomous (model generates reasoning).

---

## 7. Chain-of-Agents Collaboration (CoA)

A novel multi-agent approach where multiple LLM-based agents collaborate by communicating with each other through natural language.

Each agent specializes in a task or subtask and they share outputs to solve long-context or complex workflows collectively.

This framework is training-free and outperforms traditional single LLM or retrieval models on tasks like long document summarization and code completion.

---

# LLM Chains vs LLM Agents

Let's see some key differences between LLM Chains and LLM Agents.

# Types of LLM Chains

| Type | Description | Common Applications |
|------|-------------|---------------------|
| **Simple Sequential Chains** | Single LLM call or a series of LLM calls executed one after another. | Question answering, text summarization |
| **Multi-Step Chains** | Multiple LLM calls broken into smaller subtasks where each step feeds the next. | Document analysis, pipeline workflows |
| **Router Chains** | Routes the input to different sub-chains or models depending on the query. | Domain-specific assistants |
| **Agent-Based Chains** | Uses intelligent agents that decide which actions or tools to invoke. | Conversational AI, research assistants |
| **RAG Chains** | Combines retrieval from vector stores with generation. | Semantic search, QA systems |
| **Chain of Thought (CoT)** | Encourages step-by-step reasoning before generating the final answer. | Math, logic, reasoning tasks |
| **Chain-of-Agents (CoA)** | Multiple specialized agents collaborate to solve complex tasks. | Long document summarization, code generation |

---

# Use Cases of LLM Chains

### Chatbot Assistants

Enhanced chatbots that connect to proprietary data stores and maintain conversation memory, delivering context-aware responses.

---

### Document Analysis and Summarization

Breaking down large documents into chunks processed separately and then combined for analysis or summary, overcoming LLM token limits.

---

### Semantic Search

Retrieving precise, context-relevant documents or information based on vector similarity, improving search quality over traditional keyword search.

---

### Question Answering (QA) Systems

Integrating numerous data types and sources to accurately handle domain-specific questions.

---

### Personalized User Experiences

Providing tailored content and advice based on user history and preferences stored in memory or external databases.

---
