## The Core Problem : Why do we need Autonomous Agents?

Out-of-the-box LLMs face three fundamental restrictions that prevent them from operating in enterprise software environments:
#### 1. Mathematical Incompetence:
They predict words based on statistics; they cannot calculate complex algorithmic equations or process structured formulas reliably.
#### 2. Total Isolation:
They do not know today's date, real-time stock prices, or what is currently sitting inside your private SQL database.
#### 3. Linear Processing:
Standard chat frameworks evaluate a prompt once and output an answer. They cannot pause, evaluate if their answer is wrong, change their strategy and try again.

## The Agentic Fix:

1. An AI Agent is an architecture that wraps an LLM inside a continuous execution loop. 
2. Instead of instantly spitting out text, the LLM is given access to a toolbox of Python functions and an orchestration framework that allows it to say : "I do not know the answer yet, but I am choosing to execute tool X to find out".

## Topic 1: Function Calling / Tool Use

### Why is it important and where it is used?

Function calling is the mechanical foundation of all agents. It is used whenever a AI needs to write data to an external API, check real-time systems (like tracking an e-commerce shipping order ID), or compute verifiable arithmetic.

### The Structural Mechanics:

You do not know Gemini raw access to you computer's terminal. Instead, you write a standard Python function, and your orchestration framework reads the function's structural definition and transforms it into a machine-readable string metadata schema (a JSON Schema).
- When you pass this schema over the network to the Gemini, you are telling the model: "Here is a capability you can request".

### Step-by-Step Code Implementation 

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

# 1. ENVIRONMENT SETUP
os.environ["GOOGLE_API_KEY"] = "your_actual_gemini_api_key"

# 2. DEFINE THE TOOL (Exposing Docstrings as JSON Schemas)
# The @tool decorator tells LangChain to read the function name, parameter types, 
# and the docstring description. This text is what Gemini reads to understand WHEN to use it.
@tool
def calculate_mining_hazard_index(gas_ppm: float, temperature_c: float) -> float:
    """
    Calculates the localized mining safety risk multiplier based on environmental telemetry.
    Args:
        gas_ppm: The concentration of combustible gas in Parts Per Million.
        temperature_c: The ambient temperature measured in Celsius.
    Returns:
        A floating-point risk index scale from 0.0 (Safe) to 10.0 (Extreme Danger).
    """
    # Local hardcoded mathematical execution
    base_risk = (gas_ppm * 0.002) + (temperature_c * 0.1)
    return min(10.0, max(0.0, base_risk))

# 3. BIND THE TOOL TO THE GEMINI BRAIN
# We configure our model and pass the tool layout into the .bind_tools network array.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
model_with_tools = llm.bind_tools([calculate_mining_hazard_index])

# 4. EXECUTION GATEWAY
user_prompt = "Sensor readings from Level 4 shaft just came in: Gas is at 1200 ppm and temperature is 38 degrees Celsius. What's the hazard index?"
print("--- Step 1: Evaluating Prompt Intent with Gemini ---")

# Network Call: Sends prompt + tool description schema to Google Cloud
ai_message = model_with_tools.invoke(user_prompt)

# Inspecting the model output payload structural architecture
print(f"Raw Text Content returned: '{ai_message.content}'") 
print(f"Tool Calls Intent Packet: {ai_message.tool_calls}")
```

#### OUTPUT:

```
--- Step 1: Evaluating Prompt Intent with Gemini ---
Raw Text Content returned: ''
Tool Calls Intent Packet: [{'name': 'calculate_mining_hazard_index', 'args': {'gas_ppm': 1200.0, 'temperature_c': 38.0}, 'id': 'call_abcd1234'}]
```


# TOPIC 2 : The ReAct Framework (Reasoning + Acting)

### Why is it important & where it is used?

The ReAct (Reasoning + Acting) paradigm allows an AI to solve complex, multi-step problems by forcing it to alternate between explicit logical thoughts and concrete actions. It runs an internal loop:
#### Thought -> Action -> Observation -> Repeat

## Hand-coding a manual ReAct Loop in Python

```python
# Assuming 'model_with_tools' and 'calculate_mining_hazard_index' from above are available

def run_manual_react_agent(user_question: str):
    # A local state history array tracking our loop progress
    messages_timeline = [
        {"role": "user", "content": user_question}
    ]
    
    print("\n--- Starting Local ReAct Execution Loop ---")
    
    # We restrict the loop to a max of 3 iterations to prevent runaway infinite API bills
    for step in range(3):
        print(f"\n[LOOP STEP {step + 1}]: Contacting Gemini Cloud...")
        response = llm.bind_tools([calculate_mining_hazard_index]).invoke(messages_timeline)
        
        # Scenario A: The model does not want to call a tool. It has a final answer.
        if not response.tool_calls:
            print("[AGENT CONCLUSION]: Final text response synthesized.")
            return response.content
            
        # Scenario B: The model requests a tool action
        tool_call = response.tool_calls[0]
        print(f"[THOUGHT]: Model wants to execute tool: '{tool_call['name']}' with args: {tool_call['args']}")
        
        # Execution local matching boundary
        if tool_call['name'] == 'calculate_mining_hazard_index':
            # Run the actual local python function using the extracted arguments
            observation_result = calculate_mining_hazard_index.invoke(tool_call['args'])
            print(f"[OBSERVATION]: Local function executed. Resulting Output = {observation_result}")
            
            # Append the entire transcript back into history so Gemini can evaluate the result
            messages_timeline.append(response) # Add model's tool call intent
            messages_timeline.append({
                "role": "tool",
                "content": str(observation_result),
                "tool_call_id": tool_call['id'],
                "name": tool_call['name']
            })

# Execute the ReAct loop
final_verdict = run_manual_react_agent("Is it safe to deploy team members to an area reporting 2200 ppm gas and 42C temp?")
print(f"\n[FINAL RESPONSE]: {final_verdict}")
```

# TOPIC 3 : Stateful Graphs (LangGraph)

### Why it is important & Where it is used
While writing loops manually using for statement works for basic systems ,it becomes unmanageable when your agent needs complex behavior - like conditional branch forks, looping back to validation nodes if an evaluation fails or interacting with multiple agents simultaneously.

#### LangGraph models your agentic application as a Stateful Graph Network built of three primitives:
#### 1. State:
A central data structure (a dictionary) representing your current memory snapshot.
#### 2. Nodes:
Plain Python functions that accept the State, do some work (like calling Gemini or saving a file), and return updates to the state.
#### 3. Edges/ Conditional Edges:
The directional routing lines that govern where the execution moves next based on the state.

## Practical Implementation of LangGraph with Gemini

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage

# 1. DEFINE THE CENTRAL ARCHITECTURAL STATE
# 'add_messages' is a reducer function that tells LangGraph to always append 
# new messages to the timeline list instead of overwriting them.
class AgentGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 2. DEFINE THE NODE FUNCTIONS
def call_gemini_node(state: AgentGraphState):
    """Executes the LLM reasoning layer."""
    messages = state["messages"]
    # Provide system context parameters at runtime
    model_agent = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1).bind_tools([calculate_mining_hazard_index])
    response = model_agent.invoke(messages)
    return {"messages": [response]}

# 3. DEFINE THE CONDITIONAL ROUTING EDGE
def route_next_step(state: AgentGraphState):
    """Evaluates whether to proceed to tool execution or terminate."""
    last_message = state["messages"][-1]
    # If the model generated tool calls, route directly to the 'tools' execution node
    if last_message.tool_calls:
        return "execute_tools"
    # Otherwise, exit the loop and return the final answer to the user
    return "terminate"

# 4. CONSTRUCT THE COMPILED STATE GRAPH
workflow = StateGraph(AgentGraphState)

# Add our processing blocks
workflow.add_node("llm_reasoning", call_gemini_node)
workflow.add_node("execute_tools", ToolNode([calculate_mining_hazard_index]))

# Establish layout connectivity
workflow.set_entry_point("llm_reasoning")

# Add the conditional routing gate
workflow.add_conditional_edges(
    "llm_reasoning",
    route_next_step,
    {
        "execute_tools": "execute_tools",
        "terminate": END
    }
)

# Connect the output of the tools node back to the LLM so it can read the results
workflow.add_edge("execute_tools", "llm_reasoning")

# Compile into an executable application binary pipeline
compiled_agent = workflow.compile()

# 5. INITIALIZE AND EXECUTE
inputs = {"messages": [("user", "Run a diagnostic check on 500 ppm gas and 25C temp")]}
for output in compiled_agent.stream(inputs):
    print(output)
```

