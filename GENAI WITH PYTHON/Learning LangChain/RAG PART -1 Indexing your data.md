## 1.The Problem: Limitations of Simple LLM Chatbots

- LLMs are trained on public data up to a **knowledge cutoff** date.
    
- Two major gaps:
    
    - **Private data**: information not publicly available (e.g., internal company documents).
        
    - **Current events**: anything after the training cutoff.
        
- Without this knowledge, the model will **hallucinate** (produce false or misleading answers).
    
- Changing the prompt alone won’t fix it, because the model still lacks the necessary facts.

## 2. The Goal: Picking Relevant Context for LLMs

- If you have only 1–2 pages of private/current text, just paste it into every prompt.
    
- The real challenge is **quantity** – you have more information than fits in the context window.
    
- You must choose the **most relevant subset** of your documents for each question.
    
- Solution in two steps:
    
    1. **Indexing** – preprocess documents so they can be searched efficiently.
        
    2. **Retrieval** – fetch the best matching pieces and use them as context for the LLM.

## 3. The Four Key Steps of Ingestion (Indexing)

To go from a raw PDF to a searchable knowledge base:

1. **Extract text** from the document.
    
2. **Split** text into manageable chunks.
    
3. **Convert** chunks into numbers (embeddings) that capture meaning.
    
4. **Store** those number representations in a special database for fast retrieval.

## 4. Embeddings: Converting Text to Numbers

### 4.1 Sparse Embeddings (Bag-of-Words)

- Counts word occurrences.
    
- Simple, useful for keyword search and basic classification.
    
- **Limitation**: No understanding of meaning – “sunny day” and “bright skies” look completely different, even though they are similar in meaning.

### 4.2 LLM-Based (Dense / Semantic) Embeddings

- Learned by models that capture **semantics** (meaning).
    
- Output a long list of floating-point numbers (e.g., 100–2000 dimensions).
    
- Called **dense** because most values are non-zero.
    
- Different embedding models produce incompatible embeddings – never mix them.
    
- **Key property**: Texts with similar meanings have vectors that are close together in space.

### 4.3 Measuring Similarity

- **Cosine similarity**: measures the angle between vectors.
    
    - 1 = identical meaning, 0 = unrelated, -1 = opposite.
        
- Distance/angle between vectors indicates semantic closeness.
    
- This allows a computer to find “which document chunks are most relevant to my question”.

### 4.4 Other Uses of Embeddings

- Search, clustering, classification, recommendation, anomaly detection.
    
- Vector arithmetic (e.g., `king – man + woman ≈ queen`).
    
- Multimodal embeddings (images, video, sound).

## 5. Step 1: Converting Documents into Text (Document Loaders)

- LangChain provides **document loaders** for many formats and sources.
    
- General usage pattern:
    
    1. Choose the loader for your file type.
        
    2. Instantiate it with the file path/URL.
        
    3. Call `.load()` → returns a list of `Document` objects (text + metadata).
        

### Examples:

- **Plain text**: `TextLoader`
    
- **HTML/Web**: `WebBaseLoader` (needs `beautifulsoup4`)
    
- **PDF**: `PyPDFLoader` (needs `pypdf`)
    
- Also CSV, JSON, Markdown, Slack, Notion, etc.

```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("./test.pdf")
pages = loader.load()   # each page becomes a Document
```

## 6. Step 2: Splitting Text into Chunks (Text Splitters)

- A full PDF may be far too long for an LLM/embedding model (context window limit).
    
- We need to break it into **chunks** that are small enough but still meaningful.
    
- LangChain’s main tool: **`RecursiveCharacterTextSplitter`**
    

### How `RecursiveCharacterTextSplitter` works:

- Uses a list of separators in priority order (paragraphs → lines → spaces).
    
- Tries to keep chunks smaller than `chunk_size` (e.g., 1000 characters).
    
- Overlaps chunks (`chunk_overlap`, e.g., 200 characters) to retain context.
    
- Returns a list of `Document` objects, each with metadata including position.

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splitted_docs = splitter.split_documents(docs)
```
### Language‑specific splitting:

- Can split Python, Markdown, HTML, etc., using language separators.
    
- Preserves code blocks, functions, and other structural elements.

 ```python
 python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
 ```
 - `create_documents` takes a list of raw strings and an optional metadata list.

## 7. Step 3: Generating Text Embeddings

- LangChain’s `Embeddings` class wraps embedding models (OpenAI, Cohere, Hugging Face, etc.).
    
- Two main methods:
    
    - `.embed_documents(list_of_strings)` – for documents
        
    - `.embed_query(single_string)` – for queries
        
- Embed many documents at once for efficiency.
```python
from langchain_openai import OpenAIEmbeddings
model = OpenAIEmbeddings()
embeddings = model.embed_documents(["Hi there!", "Oh, hello!"])
# Returns a list of vectors (list of floats)
```
## 8. Step 4: Storing Embeddings in a Vector Store

- A **vector store** is a database specialized in storing vectors and performing fast similarity searches (e.g., cosine similarity).
    
- Supports CRUD operations plus semantic search.
    
- Many providers exist; LangChain supports lots of them.

## 9. Tracking Changes to Your Documents (Indexing API)

- When source documents change, you need to re‑index without duplicates or stale data.
    
- LangChain provides an `index` function plus a `RecordManager` to track document hashes and write times.
    

**Cleanup modes**:

- `None`: no automatic cleanup (manual).
    
- `incremental`: replaces old versions of the same source; leaves unrelated docs untouched.
    
- `full`: deletes any existing docs not present in the current indexing run (full re‑sync).
    

**Workflow**:

1. Create a `RecordManager` (e.g., with Postgres) and call `create_schema()`.
    
2. Use `index(docs, record_manager, vectorstore, cleanup="incremental", source_id_key="source")`.
    
3. Subsequent runs with the same source IDs will skip unchanged documents, update changed ones, and (in `full` mode) remove deleted ones.
    

---

## 10. Indexing Optimization Strategies

Basic chunking + embedding may not handle tables, images, or multi‑hop questions well. Three advanced techniques:

### 10.1 MultiVectorRetriever

- **Problem**: A table chunked by text loses the whole table context.
    
- **Solution**: Store **multiple representations** of a document.
    
    - Generate a **summary** of the original (e.g., table summary), embed only the summary.
        
    - Keep the **full original** in a separate document store (keyed by a shared ID).
        
    - At query time, the summary is matched; the full original is returned as context to the LLM.
        
- LangChain’s `MultiVectorRetriever` combines a vector store and a document store.
    

### 10.2 RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

- **Problem**: Some questions require tiny facts; others require high‑level concepts spanning many documents.
    
- **Solution**: Recursively cluster document chunks, summarize each cluster, and embed those summaries. Repeat, building a **tree** of summaries at different levels of abstraction.
    
- Index all levels together, so both narrow and broad questions can be answered.
    

### 10.3 ColBERT

- **Problem**: Compressing an entire chunk into a single embedding can lose nuance.
    
- **Solution**: Use **token‑level embeddings**. Compute similarity between every query token and every document token, sum the maximum matches. This yields a fine‑grained relevance score.
    
- Example library: `ragatouille` with the `colbert-ir/colbertv2.0` model.
    
- Can be wrapped as a LangChain retriever.