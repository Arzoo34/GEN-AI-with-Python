# PHASE 1: UNDERSTANDING THE NATIVE API LAYER

## What is an API Layer?

Think of an API as a waiter in a restaurant:

- **You** (the developer) = Customer
- **API** = Waiter
- **LLM Model** (like GPT-4, Gemini) = Kitchen

You don't go directly to the kitchen. You tell the waiter what you want, and they bring it back. Similarly, you don't interact directly with the AI model's complex code—you send requests through an API.

> **An API (Application Programming Interface)** is a set of rules, protocols, and definitions that allows two software programs or systems to communicate and exchange data with each other.

---

## What are Native SDKs?

**SDK** = Software Development Kit (a toolbox for developers)

**Native SDK** = The official toolbox provided by the company that made the AI model.

```
Framework (LangChain, LlamaIndex)
        ↓
Native SDK (google-genai, openai, anthropic)
        ↓
Raw HTTP Requests
```

---

# Lesson 2: Setting Up Your Environment

## Step 1: Install Python

Ensure Python 3.8+ is installed.

```bash
python --version
```

---

## Step 2: Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv llm_env

# Activate it (Windows)
llm_env\Scripts\activate
```

---

## Step 3: Install the SDKs

```bash
pip install google-genai
pip install openai
pip install anthropic
pip install python-dotenv
```

---

## Step 4: Get API Keys

- Google AI Studio: https://makersuite.google.com/app/apikey
- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

---

## Step 5: Store API Keys

Create a `.env` file.

```env
GOOGLE_API_KEY=your_google_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

---

# Lesson 3: Your First API Calls

## Google GenAI (Gemini)

![Google Gemini API Call](assets/gemini%20api%20call%20image%201.png)

### What's happening behind the scenes?

1. Your code packages the prompt.
2. Sends the request to Google's servers.
3. Gemini processes the request.
4. Response is generated.
5. Your application displays the response.

---

## OpenAI (GPT)

![OpenAI Chat Completion API](assets/Pasted%20image%2020260627135259.png)

### Understanding the Message Structure

```python
messages = [
    {"role": "system", "content": "You are a pirate"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Ahoy matey!"},
    {"role": "user", "content": "How are you?"}
]
```

> Every API call must include the **entire conversation history**.

---

## Anthropic (Claude)

![Claude API](assets/Pasted%20image%2020260627135351.png)

---

# Lesson 4: Understanding Hyperparameters

Hyperparameters determine how the model behaves.

---

## 1. Temperature (0.0 - 2.0)

![Temperature](assets/Pasted%20image%2020260627135955.png)

### When to use Temperature

### Code Generation

```python
temperature = 0.1
```

### Factual Q&A

```python
temperature = 0.2
```

### Creative Writing

```python
temperature = 0.8
```

### Brainstorming

```python
temperature = 0.9
```

### General Chat

```python
temperature = 0.5
```

---

## 2. Top-p (Nucleus Sampling)

### Quality Filter

![Top-p](assets/Pasted%20image%2020260627140600.png)

---

## 3. Top-k

### Limit Filter

![Top-k](assets/Pasted%20image%2020260627140723.png)

---

## 4. Frequency Penalty

### Prevent Repetition

![Frequency Penalty](assets/Pasted%20image%2020260627140855.png)

---

## 5. Presence Penalty

### Encourage New Topics

![Presence Penalty](assets/Pasted%20image%2020260627141005.png)

---

# Hyperparameter Configurations

## Coding Assistant

```python
coding_config = {
    "temperature": 0.1,
    "top_p": 0.1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

---

## Creative Writer

```python
creative_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5
}
```

---

## Factual Assistant

```python
factual_config = {
    "temperature": 0.2,
    "top_p": 0.5,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
```

---

## Brainstorming

```python
brainstorm_config = {
    "temperature": 1.0,
    "top_p": 1.0,
    "frequency_penalty": 1.0,
    "presence_penalty": 1.0
}
```

---

## Chatbot

```python
chat_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.3,
    "presence_penalty": 0.3
}
```

# Streaming and Async in LLMs

## Why do we need streaming?

Traditional (Non-Streaming) Approach:
┌─────────┐         ┌──────────┐         ┌─────────┐
│  Your   │────────>│   LLM    │────────>│  Your   │
│  App    │ Request │  Server  │ Response│  App    │
└─────────┘         └──────────┘         └─────────┘
               Processing...
               time - 30 sec
                    
You wait 30 seconds seeing NOTHING, then suddenly get everything.

**The User Experience Problem:**

- **No feedback**: User stares at a blank screen for 30 seconds
    
- **Perceived slowness**: Even if 30 seconds is reasonable, it FEELS slow
    
- **Anxiety**: "Did it crash? Is it working?"
    
- **No early exit**: Can't stop generation if going wrong direction
    

**The Technical Problem:**

- **Memory pressure**: Entire response must be held in memory
    
- **No progressive processing**: Can't start processing partial results
    
- **Poor resource utilization**: Server holds connection open with no data flowing

## The solution : STREAMING ARCHITECTURE

Streaming Approach:
┌─────────┐         ┌──────────┐
│  Your   │────────>│   LLM    │
│  App    │ Request │  Server  │
└─────────┘         └──────────┘
     ▲                   │
     │    Token 1: "The" │
     │◄──────────────────┘
     │    Token 2: "quick"
     │◄──────────────────┘
     │    Token 3: "brown"
     │◄──────────────────┘
     │    Token 4: "fox"
     │◄──────────────────┘
     │       ...continues...

# Conceptual understanding of LLM text generation
def llm_generate(prompt):
    """
    LLMs don't generate entire text at once.
    They predict ONE token at a time, autoregressively.
    """
    context = prompt
    generated_text = ""
    
    while not done:
        # 1. Model predicts next token probabilities
        probabilities = model.predict(context)
        
        # 2. Sample one token based on probabilities
        next_token = sample_token(probabilities)
        
        # 3. Add token to context for next prediction
        context = context + next_token
        generated_text = generated_text + next_token
        
        # 4. Check if we should stop
        if next_token == END_TOKEN or len(generated_text) >= max_tokens:
            done = True
    
    return generated_text  # Returns everything at once

# Streaming version - yields tokens as they're generated
def llm_stream(prompt):
    context = prompt
    
    while not done:
        probabilities = model.predict(context)
        next_token = sample_token(probabilities)
        context = context + next_token
        
        yield next_token  # 🔑 Yield token immediately instead of accumulating
        
        if next_token == END_TOKEN:
            done = True

## Python Generators (yield)

- **Python generators** are functions that produce a sequence of values lazily (on-demand) rather than computing all values at once, using the **`yield`** keyword instead of `return`.

- This approach offers **low memory usage** by storing only the current state and value, making it ideal for **large datasets**, **infinite sequences**, and **data pipelines**.

-  **Pause and Resume**: When `yield` is encountered, the function pauses, saves its local state (variables, instruction pointer), and returns the value. Calling `next()` resumes execution from where it left off.

- **Bidirectional Communication**: Generators can receive data from the caller using the `.send()` method, which passes a value into the generator that becomes the result of the `yield` expression

  ## The iterator protocol : what yield actually does

  def simple_generator():
     print("generator started")
     yield 1  # pause here, return 1
     print("resumed after yield 1")
     yield 2   # pause here, return 2
     print("generator finished")
     
  gen = simple_generator()
  print("generator object":{gen})

   first = next(gen)
   print({first})  # output : 1
   second = next(gen)
   print({second})  # output 2


## Generator state machine visualization:

State 1: CREATED
    │
    │ next()
    ▼
State 2: RUNNING (executing code until yield)
    │
    │ yield value
    ▼
State 3: SUSPENDED (paused at yield, value returned)
    │
    │ next()
    ▼
State 2: RUNNING (resumes after yield)
    │
    │ ...eventually...
    ▼
State 4: FINISHED (StopIteration raised)


## Understanding Asynchronous Programming

### The core problem: I/O is slow
import time

#### Synchronous: Tasks execute one after another
def download_file(url):
    print(f"Starting download from {url}")
    time.sleep(2)  # Simulating network delay
    print(f"Finished {url}")
    return f"Data from {url}"

#### This takes 10 seconds total (5 files × 2 seconds each)
def download_all_sync():
    urls = ['url1', 'url2', 'url3', 'url4', 'url5']
    results = []
    for url in urls:
        results.append(download_file(url))
    return results

start = time.time()
download_all_sync()
print(f"Synchronous time: {time.time() - start:.2f} seconds")
#### Output: 10.00 seconds

## The Event Loop - Heart of Async

The Event Loop is like a restaurant manager:

Instead of: Waiter takes order, goes to kitchen, WAITS for food,
           comes back, serves, then takes next order...

Async is like: Waiter takes order, sends to kitchen, IMMEDIATELY
              takes next order while first order cooks...

┌─────────────────────────────────────────┐
│         EVENT LOOP (Manager)            │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌───────┐│
│  │ Task 1:  │  │ Task 2:  │  │Task 3:││
│  │ Download │  │ Process  │  │Write  ││
│  │ File A   │  │ Data     │  │File   ││
│  └──────────┘  └──────────┘  └───────┘│
│       │              │             │   │
│       └──────────────┴─────────────┘   │
│              Cooperative                │
│              Multitasking              │
└─────────────────────────────────────────┘


## async 

import asyncio
#### Async version: Tasks run concurrently
async def download_file_async(url):
    print(f"Starting download from {url}")
    await asyncio.sleep(2)  # Non-blocking sleep!
    print(f"Finished {url}")
    return f"Data from {url}"

async def download_all_async():
    urls = ['url1', 'url2', 'url3', 'url4', 'url5']
    
    # Create all tasks at once
    tasks = [download_file_async(url) for url in urls]
    
    # Wait for all to complete
    results = await asyncio.gather(*tasks)
    return results

start = time.time()
asyncio.run(download_all_async())
print(f"Asynchronous time: {time.time() - start:.2f} seconds")
#### Output: ~2.00 seconds (not 10!)
