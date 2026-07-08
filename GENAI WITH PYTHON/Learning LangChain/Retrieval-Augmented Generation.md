# Why do we need RAG Pipeline?

If you ask an out-of-box LLM a highly specific question about your private data, it has two options:
1. It will tell you : "I don't know, I don not have access to your files".
2. It will hallucinate : make up highly confident, completely false answer.

## The old way : Fine-tuning

Before RAG, developers thought the only way to teach an LLM about custom data was to "fine-tune" it (retrain its neural network wights using custom documents).
1. The problem: Fine-tuning is incredibly expensive, takes hours or days, requires massive machine-learning expertise, and is completely static. 

## The modern way : RAG (The open-book exam)

- RAG takes the exact opposite approach. Instead of forcing Gemini to memorize your documents, RAG turns the interaction into an open-book exam.
- When use asks a question, your system automatically flips through your private documents, find the exact paragraph containing the answer, glues that paragraph directly onto the user's question, and says : "Hey Gemini, read this specific text block and use it to answer this question."

- **The takeaway:** RAG gives you the accuracy of custom, real-time data combined with the creative writing and analytical power of a pre-trained model like Gemini, without changing a single weight in the neural network.

## What is RAG?

Retrieval-Augmented Generation (RAG) is an architectural pattern in AI engineering where an external knowledge source (like a database or document repository) is attached to a Large Language Model (LLM) to improve the accuracy, relevance, and truthfulness of the responses.
 ### 1**Retrieval:** When a user asks a question, the system searches (retrieves ) relevant text documents or database entries matching that question.
 #### 2. **Augmented:** The system takes the user's original question and adds (augments) the retrieve document text directly into the prompt payload.
 #### **3. Generation:** The combined packet is handed to the LLM (like Gemini) to output (generate) a final, highly accurate answer based only on that evidence.


## The Complete RAG Architecture 

A production RAG architecture is built upon two distinct system layers: The Ingestion Pipeline (The Data Preparation Stage) and The Retrieval & Generation Pipeline (The Live Runtime Stage).

## Phase 1 : The Ingestion Pipeline (Done Offline)

#### 1. Documents :
Your raw files (PDFs, Word Docs, Markdown files) are read from your hard drive into your application memory.

#### 2. Chunking (Node parsing):
Large documents are chopped into smaller, overlapping text fragments (chunks). This is done because sending a 500-page book to Gemini for a 1-sentence is inefficient and costly. Chunks are typically 500 to 1000 characters long.

#### 3. Embedding:
Each individual text chunk is sent over the network to an embedding model (like Google's text-embedding-004). The model converts the text into a 768-dimenisonal array of coordinates representing the semantic meaning.

#### 4. Vector Store Indexing:
The text chunks, their metadata, and their 768-dimensional vector arrays are written onto your disk into a specialized database (like ChromaDB) and organized into an optimized search graph.

## Phase 2: The Retrieval and Generation Pipeline

This loop executes in real-time every single time a user types a query into your app chat box:
#### 1. User Query:
The user inputs a question (what is our company's server CPU limit?)
#### 2. Query Embedding:
Your local script instantly sends just that question text to Google's embedding model to get its matching 768-dimensional coordinate.

#### 3. Vector Retrieval:
Your local Vector Database matches the question's coordinate array against all the document chunk coordinate array stored on your disk using distance geometry. It instantly returns the top K (usually 3 to 5) most similar text chunks.

#### 4. Prompt Augmentation:
Your orchestration framework takes the user's raw question and seamlessly wraps it inside a foundational context template layout:


 ```
 You are a secure corporate assistant. Use ONLY the following facts to answer the user's question:

[Retrieved Chunk 1: Project CodeShield limits container CPU to 0.5 cores.]
[Retrieved Chunk 2: Docker configurations require version 24.0.]

User Question: What is our company's server CPU limit?
Answer:
 ```

#### 5. Generation Network call : 
This augmented string payload is sent across the internet to Gemini. Gemini reads the context instructions, extracts the fact, formats it into clear human syntax and streams the correct answer back to your console.


- **The Vector DB (ChromaDB):** Sits completely on **your machine**. It does the mechanical heavy lifting of slicing files and searching coordinates locally.
    
- **The LLM (Gemini):** Sits on **Google's cloud infrastructure**. It does zero searching; it simply acts as an advanced reading comprehension engine that translates your raw database chunks into a beautifully articulated answer.


##  Where is this used in the real world?

- **AI Customer Support Agents:** Reading through thousands of product manuals to give users exact troubleshooting steps instantly.
    
- **Legal and Financial Analysis:** Uploading a 200-page contract or earnings report and asking, _"What are the hidden liability clauses or compliance risks?"_
    
- **Academic/Research Tools:** Querying massive medical journals or code documentation libraries without reading through thousands of pages manually.

```python
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.google import Gemini
from llama_index.embeddings.google import GeminiEmbedding

# -------------------------------------------------------------------------
# 1. GLOBAL CONFIGURATION & SETTINGS BOUNDARY
# -------------------------------------------------------------------------
# Set your API token for cloud authentication
os.environ["GOOGLE_API_KEY"] = "your_actual_gemini_api_key_here"

# Configure LlamaIndex global settings to use Gemini for both thinking and embedding
# This ensures that whenever LlamaIndex needs an engine, it uses Google's infrastructure.
Settings.llm = Gemini(model="models/gemini-2.5-flash", temperature=0.2)
Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004")

# -------------------------------------------------------------------------
# 2. THE INGESTION PHASE (Local Machine Execution)
# -------------------------------------------------------------------------
print("--- Step 1: Ingesting Data from Local Drive ---")

# SimpleDirectoryReader scans your local folder and reads the files into memory
documents = SimpleDirectoryReader("./my_documents").load_data()

print(f"[INFO] Loaded {len(documents)} document file(s) into local RAM.")

# VectorStoreIndex takes the text, pushes it to Google's embedding API behind the scenes,
# receives the 768-dimensional coordinates, and builds an in-memory vector database.
print("[INFO] Generating embeddings and building vector index mapping...")
index = VectorStoreIndex.from_documents(documents)
print("[SUCCESS] Local Knowledge Base Index compiled successfully.\n")

# -------------------------------------------------------------------------
# 3. THE RETRIEVAL & GENERATION PHASE (Local -> Cloud Loop)
# -------------------------------------------------------------------------
# We assemble a Query Engine out of our database index.
query_engine = index.as_query_engine()

# Define a query testing information that Gemini could NOT know from its base training
user_question = "What are the exact container resource limits configured for Project CodeShield?"
print(f"--- Step 2: Executing RAG for Query: '{user_question}' ---")

# Execution Breakdown under the hood:
# A. LlamaIndex asks text-embedding-004 to convert the question into a vector.
# B. LlamaIndex matches it against the 'company_info.txt' vector chunks locally.
# C. It pulls the text regarding CPU/RAM limits.
# D. It crafts an augmented prompt payload: 'Context: [Project CodeShield limits...] Question: [What are the limits?]'
# E. It transmits this payload to Gemini.
response = query_engine.query(user_question)

# -------------------------------------------------------------------------
# 4. UNPACKING THE GROUNDED RESPONSE
# -------------------------------------------------------------------------
print("\n--- Step 3: Gemini Grounded Output ---")
print(response)
```

Look carefully at how clean the abstraction is, but keep track of who did what work:

1. **`SimpleDirectoryReader`** executed **locally**, looking at your file system paths.
    
2. **`VectorStoreIndex.from_documents`** triggered an automated network pipeline: converting your local file blocks into payloads, getting vector arrays from Google's embedding servers, and saving the structural matrix back onto your engine memory.
    
3. **`query_engine.query`** intercepted your natural language phrase, executed a localized similarity search score computation to grab the exact paragraph match, packaged it into a safe system prompt, and leveraged Gemini's text engine over the internet to yield your perfectly accurate, non-hallucinated answer.