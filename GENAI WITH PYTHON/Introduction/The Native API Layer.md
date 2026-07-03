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

```python
# PART 1: IMPORTS
import os  # For reading environment variables
from dotenv import load_dotenv  # For loading .env file
import google.generativeai as genai  # Google's official SDK

# PART 2: LOAD ENVIRONMENT VARIABLES
load_dotenv()  # This reads your .env file
api_key = os.getenv('GOOGLE_API_KEY')  # Gets the API key
# os.getenv() is like saying "find this variable in the environment"

# PART 3: CONFIGURE THE CLIENT
genai.configure(api_key=api_key)  
# This sets up the connection to Google's servers
# Think of it as dialing the phone number

# PART 4: CHOOSE YOUR MODEL
model = genai.GenerativeModel('gemini-1.5-pro')
# 'gemini-1.5-pro' is like choosing which chef cooks your meal
# Other options: 'gemini-1.5-flash' (faster, cheaper), 'gemini-pro' (older version)

# PART 5: CREATE A PROMPT
prompt = "Explain quantum computing in simple terms"
# This is what you're asking the model

# PART 6: GENERATE A RESPONSE
response = model.generate_content(prompt)
# This sends your request and waits for a response

# PART 7: EXTRACT THE TEXT
print(response.text)
# .text extracts just the text portion of the response
```

### What's happening behind the scenes?

1. Your code packages the prompt.
2. Sends the request to Google's servers.
3. Gemini processes the request.
4. Response is generated.
5. Your application displays the response.

---

## OpenAI (GPT)

```python
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI uses a different approach - create a client object
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# The client is your permanent connection to OpenAI

# Make an API call
response = client.chat.completions.create(
    model="gpt-4o-mini",  # Model name (cheapest GPT-4 variant)
    messages=[  # List of messages in the conversation
        {
            "role": "system",  # Instructions for the AI
            "content": "You are a helpful assistant that explains things simply."
        },
        {
            "role": "user",  # The user's message
            "content": "Explain quantum computing in simple terms"
        }
    ],
    # Why messages is a list? Because ChatGPT remembers conversation history!
)

# Navigate the response object
print(response.choices[0].message.content)
# response.choices = list of possible responses (usually 1)
# [0] = first response
# .message = the message object
# .content = the actual text
```

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

```python
import anthropic
from dotenv import load_dotenv
import os

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

response = client.messages.create(
    model="claude-3-haiku-20240307",  # Fastest Claude model
    max_tokens=1000,  # Maximum length of response
    system="You are a helpful assistant that explains things simply.",  # System prompt
    messages=[
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms"
        }
    ]
)

print(response.content[0].text)
# Anthropic returns content as a list of blocks
# [0] = first content block
# .text = the text of that block
```

---

# Lesson 4: Understanding Hyperparameters

Hyperparameters determine how the model behaves.

---

## 1. Temperature (0.0 - 2.0)

```python
# LOW TEMPERATURE (0.0 - 0.3): Very focused, predictable
# Like a strict teacher who always gives the same answer
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Complete: Roses are ___"}],
    temperature=0.1
)
# Will almost always say: "Roses are red"

# HIGH TEMPERATURE (0.7 - 1.0): Creative, diverse
# Like an artist who thinks outside the box
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Complete: Roses are ___"}],
    temperature=0.9
)
# Might say: "Roses are poetry in motion"
# Or: "Roses are nature's valentines"
# Or: "Roses are silent symphonies"
```

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
```python
# Think of the AI having a bag of words it could choose next
# Top_p controls how many words are in that bag based on probability

# TOP_P = 0.1: Only consider words in top 10% probability
# Very narrow selection - boring but safe
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "The sky is"}],
    temperature=1.0,  # Keep temperature high to see top_p effect
    top_p=0.1
)
# Only considers: "blue", "the", "a" (few options)
# Almost always says: "blue"

# TOP_P = 0.9: Consider words in top 90% probability
# Wide selection - more varied responses
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "The sky is"}],
    temperature=1.0,
    top_p=0.9
)
# Considers: "blue", "beautiful", "limitless", "cloudy", "falling", etc.
# More varied completions
```


---

## 3. Top-k

### Limit Filter

```python
# Top_k simply says: "Only consider the top K most likely words"

# TOP_K = 1: Only the most likely word
# Extremely deterministic
response = genai.GenerativeModel('gemini-1.5-pro').generate_content(
    "The sky is",
    generation_config={
        "top_k": 1
    }
)
# Always picks the single most likely word

# TOP_K = 40: Consider top 40 words
response = genai.GenerativeModel('gemini-1.5-pro').generate_content(
    "The sky is",
    generation_config={
        "top_k": 40
    }
)
# More variety from a larger pool
```

---

## 4. Frequency Penalty

### Prevent Repetition

```python
# POSITIVE VALUES: Penalize words already used
# Negative values: Encourage repetition

# HIGH FREQUENCY PENALTY (1.0): Avoid repetition strongly
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a paragraph about dogs"}],
    frequency_penalty=2.0  # Maximum penalty for repeating
)
# Will use diverse vocabulary, avoids repeating "dog" too much
# "Canines are loyal companions. These four-legged friends..."

# NEGATIVE FREQUENCY PENALTY (-1.0): Encourage repetition
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a paragraph about dogs"}],
    frequency_penalty=-1.0
)
# "Dogs are great. Dogs are loyal. Dogs are friendly. Dogs..."
```

---

## 5. Presence Penalty

### Encourage New Topics

```python
# POSITIVE VALUES: Talk about new things
# Negative values: Stay on current topic

# HIGH PRESENCE PENALTY (1.0): Change topics frequently
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me about space"}],
    presence_penalty=2.0
)
# "Space is vast. Speaking of vast, oceans are deep. 
#  Ocean creatures like whales... Have you tried whale watching?"

# NEGATIVE PRESENCE PENALTY (-1.0): Stick to the topic
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me about space"}],
    presence_penalty=-1.0
)
# "Space is vast. It contains stars. Stars are massive. 
#  Our Sun is a star. The solar system..."
```

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
```python
┌─────────┐         ┌──────────┐         ┌─────────┐
│  Your   │────────>│   LLM    │────────>│  Your   │
│  App    │ Request │  Server  │ Response│  App    │
└─────────┘         └──────────┘         └─────────┘
               Processing...
               time - 30 sec
```

                    
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

```python
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

```

# Conceptual understanding of LLM text generation
```python
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
```


# Streaming version - yields tokens as they're generated
```python
def llm_stream(prompt):
    context = prompt
    
    while not done:
        probabilities = model.predict(context)
        next_token = sample_token(probabilities)
        context = context + next_token
        
        yield next_token  # 🔑 Yield token immediately instead of accumulating
        
        if next_token == END_TOKEN:
            done = True
```


## Python Generators (yield)

- **Python generators** are functions that produce a sequence of values lazily (on-demand) rather than computing all values at once, using the **`yield`** keyword instead of `return`.

- This approach offers **low memory usage** by storing only the current state and value, making it ideal for **large datasets**, **infinite sequences**, and **data pipelines**.

-  **Pause and Resume**: When `yield` is encountered, the function pauses, saves its local state (variables, instruction pointer), and returns the value. Calling `next()` resumes execution from where it left off.

- **Bidirectional Communication**: Generators can receive data from the caller using the `.send()` method, which passes a value into the generator that becomes the result of the `yield` expression

  ## The iterator protocol : what yield actually does
  ```python
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
  ```

## Generator state machine visualization:
```python
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

```



## Understanding Asynchronous Programming

### The core problem: I/O is slow
```python
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
```




## The Event Loop - Heart of Async

The Event Loop is like a restaurant manager:

Instead of: Waiter takes order, goes to kitchen, WAITS for food,
           comes back, serves, then takes next order...

Async is like: Waiter takes order, sends to kitchen, IMMEDIATELY
              takes next order while first order cooks...
```python
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

```


## async 
```python
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
```

