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
