# Memory Management in Stateless APIs

## The Core problem : Why do we even need to manage memory?

- The Gemini API is stateless.

  ##### What does STATELESS means?
  
-  Every single request is treated like a brand-new.
- Google Gemini does not store your previous questions on it sever between your API calls. If you write:
- - **Turn 1 (You):** "My name is Arzoo." → _Gemini responds: "Nice to meet you, Arzoo!"_
- **Turn 2 (You):** "What is my name?" → _Gemini responds: "I don't have access to your personal information, what is your name?"_

  Because Gemini instantly forgets everything the moment it sends a response, **you (the developer)** must build a system on your local server to remember the conversation and feed it back to Gemini every single time.

## Where is it used in the real world?
Any production AI application you interact with uses these exact patterns. if these memory patterns are not implemented correctly, the software breaks down instantly:

- **Customer Support Chatbots**: The bots uses Buffer or Window memory to track the chat context.
- **Coding Companions (like VS Code Copilot):** When you highlight fix this the system uses memory to pass the code files you were working on a few minutes ago alongside your current question.
- **Long-Term Enterprise AI Assistant:** Tools that manage projects data over weeks or months use Summary or database-backed memory so they don't lose track of overarching goals while keeping computing bills low.

## The Anatomy of a Chat Payload 

LLM does not just receive a string of text. It receives an array (a list) of structured objects called Messages. Each message has a specific role.

When LlamaIndex or LangChain sends a packet over the network to Gemini, it converts your conversation into a structured timeline that looks exactly like a JSON block:

```JSON
[
  { "role": "system", "content": "You are a helpful assistant." },
  { "role": "user", "content": "My name is Arzoo." },
  { "role": "assistant", "content": "Nice to meet you, Arzoo! How can I help you today?" },
  { "role": "user", "content": "What is my name?" }
]
```

Every single time you press enter, your local computer compiles this entire growing history list and sends it over the internet. Gemini reads the whole script from the top down, figures out the pattern, and generates the next assistant response.

## The 3 Core Memory Strategies:

As conversation grows from 4 to 100 turns, that JSON list becomes massive. This is where memory management becomes a trade-off between Accuracy and Cost/Speed.

### Strategy 1 : Buffer Memory (remember everything)
- **How it works:** Your local server acts as a continuous tape recorder. Every single text message sent by the user and received by Gemini is appended to a list in your computer's RAM or database. When a new question is asked, the _entire_ tape recorder history is sent over the network.
    
- **Why it's important:** It provides absolute perfect accuracy. The model remembers every tiny detail, nuance, and code snippet mentioned at the very beginning of the session.
    
- **Where it fails:** High cost and crash risk. LLMs charge you by the token (words/characters). If your history has 10,000 words, you pay for those 10,000 words _every single time_ the user types a new sentence. Eventually, the list becomes longer than Gemini's maximum context capacity, and the API will return a `400 Bad Request / Context Window Exceeded` error, crashing your app.

### Strategy 2: Window Memory (The "Sliding Window" Strategy)

- **How it works:** Instead of sending the whole tape recording, your local server only looks at the last $K$ exchanges (where 1 exchange = 1 user message + 1 assistant response).
    
- **The Mechanics:** If you set a window size of $K=2$, your local Python code actively slices the list. If there are 10 messages in your database, your code deletes the oldest 6 messages from the payload copy and sends _only_ the latest 4 messages to Gemini.
    
- **Why it's important:** It keeps your API bills perfectly stable and predictable. Your performance stays lightning-fast because the network payload never grows past a specific size.
    
- **Where it fails:** Total amnesia. If the user says "My API key is X" in turn 1, and you are on turn 10, that information has completely slid out of the window. Gemini will have no clue what API key they are talking about.
    

### Strategy 3: Summary Memory (The "Executive Summary" Strategy)

- **How it works:** This is a hybrid approach that uses a two-step process to compress information.
    
- **The Mechanics:**
    
    1. As the conversation happens, your local server watches the history.
        
    2. When the text gets too long, your server makes a secret background API call to Gemini, saying: _"Here are the last 20 messages. Compress them into a 2-sentence summary of what the user wants."_
        
    3. Your local server saves that summary string to your database and deletes the raw 20 messages.
        
    4. The next time the user asks a question, your server sends: `System: [Summary String] + User: [New Question]`.
        
- **Why it's important:** It allows for infinitely long conversations. You can talk to the bot for three days straight, and it will never crash or break your budget.
    
- **Where it fails:** Loss of detail. While Gemini will remember the _topic_ you were talking about, it will lose exact technical details like variable names, syntax formats, or specific numbers because they were washed away during the summary process.
    

## WINDOW MEMORY SYSTEM Using Python and Gemini

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# -------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP (Local Machine)
# -------------------------------------------------------------------------
# Set your API key so your local script can authenticate with Google's servers.
os.environ["GOOGLE_API_KEY"] = "your_actual_gemini_api_key_here"

# Initialize our connection config for Gemini.
# We use gemini-2.5-flash as our fast, efficient chat engine.
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# -------------------------------------------------------------------------
# 2. LOCAL MEMORY STORAGE (Your Server's RAM)
# -------------------------------------------------------------------------
# This list acts as our database. It will store EVERY single message 
# exchanged during this console session. It resides entirely in your local RAM.
FULL_CHAT_HISTORY = []

# Define the sliding window size (K). 
# K=2 means we will only ever send the last 2 rounds of chat to Gemini.
# (1 round = 1 Human message + 1 AI response, so 2 rounds = 4 messages total)
WINDOW_SIZE = 2

# -------------------------------------------------------------------------
# 3. THE INTERACTION & PRUNING CORE
# -------------------------------------------------------------------------
def chat_with_memory(user_query: str):
    # --- STEP A: Store new input locally ---
    # Convert the user's string into a structured LangChain Message object
    user_msg = HumanMessage(content=user_query)
    FULL_CHAT_HISTORY.append(user_msg)
    
    # --- STEP B: Local Window Slicing (The Magic) ---
    # Calculate how many messages represent our 'K' window limit.
    max_messages_to_send = WINDOW_SIZE * 2
    
    # Slice the list locally. If FULL_CHAT_HISTORY has 10 items, 
    # pruned_history will only copy the final 4 items.
    if len(FULL_CHAT_HISTORY) > max_messages_to_send:
        pruned_history = FULL_CHAT_HISTORY[-max_messages_to_send:]
    else:
        pruned_history = FULL_CHAT_HISTORY.copy()
        
    # Inject a baseline identity (System Message) at the absolute front of the payload
    # so Gemini always knows how to behave, regardless of what gets deleted.
    network_payload = [SystemMessage(content="You are a helpful, concise AI companion.")] + pruned_history
    
    # --- STEP C: The Network Event (Local -> Cloud) ---
    # Print a local debug line so you can see exactly how much context we are sending over the wire
    print(f"\n[DEBUG] Total messages in local DB: {len(FULL_CHAT_HISTORY)}")
    print(f"[DEBUG] Messages being sent to Gemini over the network: {len(network_payload)}")
    
    # Send the pruned list across the internet to Google's API endpoint.
    response = model.invoke(network_payload)
    
    # --- STEP D: Store AI response locally ---
    # Extract the string answer from the Gemini response metadata packet and append it to our local tape
    ai_msg = AIMessage(content=response.content)
    FULL_CHAT_HISTORY.append(ai_msg)
    
    return response.content

# -------------------------------------------------------------------------
# 4. RUNNING THE CHAT LOOP (The Console Interface)
# -------------------------------------------------------------------------
print("Chat session started with Gemini! Type 'exit' to quit.\n")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'exit':
        break
        
    reply = chat_with_memory(user_input)
    print(f"Gemini: {reply}")
```

## Running a Trace Simulation (What Actually Happens)

If you run this script and type these exact sentences back-to-back, look at how the memory behaves under the hood:

### Turn 1:

- **You type:** `Hi, my secret password is 'BANANA'.`
    
- **Local Processing:** `FULL_CHAT_HISTORY` now has `[User: Hi, secret password is BANANA]`.
    
- **Network Call:** Sends 1 System Message + 1 User Message to Gemini.
    
- **Gemini Responds:** `Got it! I will remember that secret password.`
    
- **Local Processing:** `FULL_CHAT_HISTORY` appends the AI message. (Total in DB: 2)
    

### Turn 2:

- **You type:** `I love eating fruits.`
    
- **Local Processing:** Appends to DB. (Total in DB: 3)
    
- **Network Call:** Sends System + 3 Chat Messages. (Well within our $K=2$ window limit of 4 messages).
    
- **Gemini Responds:** `Fruits are healthy! What is your favorite one?`
    
- **Local Processing:** Appends to DB. (Total in DB: 4)
    

### Turn 3: (The Slicing Trigger)

- **You type:** `What was that secret password I told you in the beginning?`
    
- **Local Processing:** Appends to DB. (Total in DB: 5)
    
- **The Slicing Execution:** The code detects that `len(FULL_CHAT_HISTORY)` is 5, which is greater than our limit of 4 (`WINDOW_SIZE * 2`). It completely chops off Turn 1 from the payload copy!
    
- **Network Call:** The network payload compiled by your computer looks exactly like this:
    
    JSON
    
    ```JSON
    [
      {"role": "system", "content": "You are a helpful, concise AI companion."},
      {"role": "assistant", "content": "Fruits are healthy! What is your favorite one?"},
      {"role": "user", "content": "What was that secret password I told you in the beginning?"}
    ]
    ```
    
- **Gemini's Evaluation:** Look closely at that list. **The word 'BANANA' is completely missing from the data packet.** Gemini reads this, looks through its context, and sees no mention of a password.
    
- **Gemini Responds:** `I'm sorry, but you haven't mentioned a secret password yet in our conversation.`


- ##### This code proves that Gemini is not actively "remembering" or storing your values inside its neural network cloud. It only knows what you choose to put into that `network_payload` list right before your computer hits the internet send button. By changing the `WINDOW_SIZE` variable in your local script, you control exactly how much data leaves your machine.

# SUMMARY MEMORY (ConversationSummaryMemory)

## The core Architecture :

```
                  +--------------------------+
                  |  User types a new question|
                  +--------------------------+
                               ↓
+--------------------------------------------------------------+
| LOCAL SERVER: Compiles [Current Summary] + [New Question]     |
+--------------------------------------------------------------+
                               ↓ (Network Call 1)
+--------------------------------------------------------------+
| GEMINI CLOUD: Reads the summary context, answers the query  |
+--------------------------------------------------------------+
                               ↓ (Returns to Local)
+--------------------------------------------------------------+
| LOCAL SERVER: Sends [Old Summary] + [Last Turn Text] to      |
|               Gemini to generate a brand-new updated summary |
+--------------------------------------------------------------+
                               ↓ (Returns to Local)
+--------------------------------------------------------------+
| LOCAL DATABASE: Overwrites the old summary string with the   |
|                 new one. Ready for the next turn.            |
+--------------------------------------------------------------+
```

## Implementing Summary Memory from Scratch

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# -------------------------------------------------------------------------
# 1. ENVIRONMENT SETUP (Local Machine)
# -------------------------------------------------------------------------
os.environ["GOOGLE_API_KEY"] = "your_actual_gemini_api_key_here"

# Initialize our primary chat model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# -------------------------------------------------------------------------
# 2. LOCAL STATE STORAGE (Your Server's RAM / Database Simulation)
# -------------------------------------------------------------------------
# Instead of storing an infinite list of messages, we only store a single string 
# that updates dynamically after every conversation round.
CURRENT_CONVERSATION_SUMMARY = "The user has just started the session."

# -------------------------------------------------------------------------
# 3. BACKGROUND MEMORY COMPRESSION LOGIC
# -------------------------------------------------------------------------
def update_summary(old_summary: str, last_user_input: str, last_ai_response: str) -> str:
    """
    This function acts as the 'Data Librarian'. It makes a hidden, secondary 
    network call to Gemini to compress the chat logs.
    """
    print("\n[DEBUG] Running background compression call to update summary...")
    
    compilation_prompt = (
        f"Progressively summarize the conversation. Update the current summary "
        f"with the new lines of conversation provided.\n\n"
        f"Current Summary:\n{old_summary}\n\n"
        f"New Lines of Conversation:\n"
        f"User: {last_user_input}\n"
        f"AI: {last_ai_response}\n\n"
        f"New Summary (Write a concise paragraph capturing all critical details like names, rules, or keys):"
    )
    
    # We pass the instruction payload directly to Gemini over the network
    updated_summary_packet = model.invoke([HumanMessage(content=compilation_prompt)])
    return updated_summary_packet.content

# -------------------------------------------------------------------------
# 4. THE INTERACTION CORE
# -------------------------------------------------------------------------
def chat_with_summary_memory(user_query: str):
    global CURRENT_CONVERSATION_SUMMARY
    
    # --- STEP A: Construct the Grounded Payload ---
    # We build a custom system message that holds our rolling summary
    context_system_message = SystemMessage(
        content=f"You are a helpful AI assistant. Here is a summary of what has happened "
                f"so far in this conversation for your reference:\n{CURRENT_CONVERSATION_SUMMARY}"
    )
    
    current_question = HumanMessage(content=user_query)
    
    # Pack them up to send over the network
    network_payload = [context_system_message, current_question]
    
    print(f"\n[DEBUG] Current Summary size: {len(CURRENT_CONVERSATION_SUMMARY.split())} words.")
    print(f"[DEBUG] Sending Context + Question to Gemini...")
    
    # --- STEP B: Main Network Call ---
    response = model.invoke(network_payload)
    ai_answer = response.content
    
    # --- STEP C: Asynchronous Summary Overwrite ---
    # Before we finish, we update our local RAM state with a fresh summary 
    # that includes this current exchange.
    CURRENT_CONVERSATION_SUMMARY = update_summary(
        old_summary=CURRENT_CONVERSATION_SUMMARY,
        last_user_input=user_query,
        last_ai_response=ai_answer
    )
    
    return ai_answer

# -------------------------------------------------------------------------
# 5. RUNNING THE CHAT LOOP 
# -------------------------------------------------------------------------
print("Chat session started using Summary Memory! Type 'exit' to quit.\n")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'exit':
        break
        
    reply = chat_with_summary_memory(user_input)
    print(f"\nGemini: {reply}")
    print(f"--- [LOCAL DATABASE CURRENT STATE] ---\n{CURRENT_CONVERSATION_SUMMARY}\n---------------------------------------")
```
## Running a Trace Simulation (What Actually Happens)

Let's test this strategy with the exact same scenarios that broke our Window Memory:

### Turn 1:

- **You type:** `Hi, my secret password is 'BANANA'.`
    
- **Local Machine Action:** Compiles `System: [The user has just started the session.] + User: [Hi, my secret password is 'BANANA'.]` and sends it over the wire.
    
- **Gemini Responds:** `Got it. I'll remember that your secret password is 'BANANA'.`
    
- **Background Update:** The code immediately invokes `update_summary()`. Gemini reviews the exchange and generates a new summary string.
    
- **Local State Overwrite:** `CURRENT_CONVERSATION_SUMMARY` becomes: _"The user introduced themselves and stated that their secret password is 'BANANA'. The AI acknowledged this security detail."_
    

### Turn 2: (10 Turns Later...)

Imagine you talk about fruits, coding, and sports for 10 straight turns. The raw logs would be thousands of words long. But with our code, those raw logs are continuously deleted, and only the paragraph summary remains.

- **You type:** `What was that secret password I told you in the beginning?`
    
- **Local Machine Action:** Compiles the payload:
    
    Plaintext
    
    ```
    System Message: You are a helpful AI assistant. Here is a summary of what has happened so far... [The user introduced themselves and stated that their secret password is 'BANANA'. The AI acknowledged this... they discussed coding problems...]
    User Message: What was that secret password I told you in the beginning?
    ```
    
- **Network Call:** Transmits this beautifully condensed packet over the network.
    
- **Gemini reads the packet:** It looks at the system context message, extracts the fact from the summary text, and synthesizes the correct answer.
    
- **Gemini Responds:** `Your secret password is 'BANANA'.`
