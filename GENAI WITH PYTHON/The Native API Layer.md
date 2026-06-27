## PHASE 1: UNDERSTANDING THE NATIVE API LAYER

### What is an API Layer?

Think of an API as a waiter in a restaurant:

- **You** (the developer) = Customer
- **API** = Waiter
- **LLM Model** (like GPT-4, Gemini) = Kitchen

You don't go directly to the kitchen. You tell the waiter what you want, and they bring it back. Similarly, you don't interact directly with the AI model's complex code—you send requests through an API.

##### **An API (Application Programming Interface)** is a set of rules, protocols, and definitions that allows two software programs or systems to communicate and exchange data with each other

### What are "Native SDKs"?

**SDK** = Software Development Kit (a toolbox for developers)

**Native SDK** = The official toolbox provided by the company that made the AI model

Framework (like LangChain) → Higher level, more abstraction
        ↓
Native SDK (like google-genai) → Direct interaction, less bloat
        ↓
Raw HTTP Requests → Most basic level


---

## **Lesson 2: Setting Up Your Environment**

### Step 1: Install Python

First, ensure you have Python 3.8+ installed:

bash

python --version  # Should show 3.8 or higher

### Step 2: Create a Virtual Environment

bash
---- Create a virtual environment (isolated Python space)
python -m venv llm_env
----Activate it
----On Windows:
llm_env\Scripts\activate
### Step 3: Install the Three Major SDKs

bash

pip install google-genai
pip install openai
pip install anthropic
pip install python-dotenv  # For managing API keys securely

### Step 4: Get Your API Keys

1. **Google AI Studio**: Visit [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
    
2. **OpenAI**: Visit [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
    
3. **Anthropic**: Visit [https://console.anthropic.com/](https://console.anthropic.com/)
    

### Step 5: Securely Store API Keys

Create a `.env` file:

env

GOOGLE_API_KEY=your_google_key_here
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

## **Lesson 3: Your First API Calls - Complete Breakdown**

Let's understand every single line of code:

### Google GenAI (Gemini) 

![Architecture](assets/ gemini api call image 1.png)

**What's happening behind the scenes?**

1. Your code packages your prompt into a request
    
2. Sends it over the internet to Google's servers
    
3. The AI model processes your prompt
    
4. The response travels back to your code
    
5. You extract and display the text
    

### OpenAI (GPT) - Detailed Explanation

![[Pasted image 20260627135259.png]]

**Understanding the Message Structure:**

python

messages = [
    {"role": "system", "content": "You are a pirate"},  # Sets AI's personality
    {"role": "user", "content": "Hello"},  # First user message
    {"role": "assistant", "content": "Ahoy matey!"},  # AI's response
    {"role": "user", "content": "How are you?"}  # Next user message
]
##### Each new call should include the ENTIRE history!

### Anthropic (Claude) - Detailed Explanation

![[Pasted image 20260627135351.png]]


## **Lesson 4: Understanding Hyperparameters - The Complete Guide**

### What are Hyperparameters?

Hyperparameters are like the "personality settings" for AI. They control how creative, focused, or predictable the AI's responses are.

### **1. Temperature (0.0 to 2.0)**

![[Pasted image 20260627135955.png]]
##### WHEN TO USE TEMPERATURE:
### CODE GENERATION : low temp
temperature = 0.1 # you want exact, correct code
### FACTUAL Q&A: low temp
temperature = 0.2  # you want consistent, accurate answers
### CREATIVE WRITING : High temp
temperature = 0.8  # you want unique, creative content
### BRAINSTORMING: high temp
temperature = 0.9  # you want diverse, unexpected answers
### BALANCED : medium temp
temperature = 0.5  # good default for general use

## 2. Top_p (Nucleus Sampling) - 0.0 to 1.0

### The Quality Filter
![[Pasted image 20260627140600.png]]

## 3. Top_k - Only available in some models

### The Limit Filter
![[Pasted image 20260627140723.png]]

## 4. Frequency Penalty -  -2.0 to 2.0
### The do not repeat yourself setting
![[Pasted image 20260627140855.png]]

## 5. Presence Penalty  -2.0 to 2.0

### the stay on topic vs explore new topics setting
![[Pasted image 20260627141005.png]]
### **Complete Hyperparameter Combinations for Different Use Cases**

python

### 1. CODING ASSISTANT: Precise, consistent
coding_config = {
    "temperature": 0.1,
    "top_p": 0.1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
### 2. CREATIVE WRITER: Diverse, surprising
creative_config = {
    "temperature": 0.9,
    "top_p": 0.95,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5
}
### 3. FACTUAL ASSISTANT: Accurate, focused
factual_config = {
    "temperature": 0.2,
    "top_p": 0.5,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}
### 4. BRAINSTORMING: Maximum diversity
brainstorm_config = {
    "temperature": 1.0,
    "top_p": 1.0,
    "frequency_penalty": 1.0,
    "presence_penalty": 1.0
}
### 5. CHATBOT: Balanced
chat_config = {
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.3,
    "presence_penalty": 0.3
}