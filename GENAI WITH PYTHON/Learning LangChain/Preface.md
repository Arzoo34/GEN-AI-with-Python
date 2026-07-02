## Introduction

#### Let’s break down the term large language model:
- Large refers to the size of these models in terms of training data and parameters used during the learning process. For example, OpenAI’s GPT-3 model contains 175 billion parameters, which were learned from training on 45 terabytes of text data.
- Language model refers to a computer algorithm trained to receive written text (in English or other languages) and produce output also as written text (in the same language or a different one).
- These are neural networks, a type of ML model which resembles a stylized conception of the human brain, with the final output resulting from the combination of the individual outputs of many simple mathematical functions, called neurons, and their interconnections
- Large language models are instances of big, general-purpose language models that are trained on vast amounts of text.
- The driving engine behind LLMs’ predictive power is known as the transformer neural network architecture
- The transformer architecture enables models to handle sequences of data, such as sentences or lines of code, and make predictions about the likeliest next word(s) in the sequence
- Transformers are designed to understand the context of each word in a sentence by considering it in relation to every other word. This allows the model to build a comprehensive understanding of the meaning of a sentence, paragraph, and so on (in other words, a sequence of words) as the joint meaning of its parts in relation to each other.

## Instruction-Tuned LLMs

Researchers have made pretrained LLMs easier to use by further training (additional training applied on top of the long and costly training described in the previous section), also known as fine-tuning them on the following:
#### Task-specific datasets
These are datasets of pairs of questions/answers manually assembled by research‐ ers, providing examples of desirable responses to common questions that end users might prompt the model with. For example, the dataset might contain the following pair: Q: What is the capital of England? A: The capital of England is London. Unlike the pretraining datasets, these are manually assembled, so they are by necessity much smaller: 
#### Reinforcement learning from human feedback (RLHF)
Through the use of RLHF methods, those manually assembled datasets are aug‐ mented with user feedback received on output produced by the model. For example, user A preferred The capital of England is London to London is the capital of England as an answer to the earlier question.

## DIALOGUE-TUNED LLMs

Models tailored for dialogue or chat purposes are a further enhancement of instruction-tuned LLMs. Done through :

#### Dialogue Datasets
The manually assembled fine-tuning datasets are extended to include more exam‐ ples of multiturn dialogue interactions, that is, sequences of prompt-reply pairs.

#### Chat Format
The input and output formats of the model are given a layer of structure over freeform text, which divides text into parts associated with a role (and option‐ ally other metadata like a name).

## Fine-Tuned LLMs

 - Fine-tuned LLMs are created by taking base LLMs and further training them on a proprietary dataset for a specific task.
 - Technically, instruction-tuned and dialoguetuned LLMs are fine-tuned LLMs, but the term “fine-tuned LLM” is usually used to describe LLMs that are tuned by the developer for their specific task.

## PROMPTING

### 1. Zero-Shot Prompting
The first and most straightforward prompting technique consists of simply instructing the LLM to perform the desired task.

### 2. Chain-of-Thought
- A very useful iteration is to further instruct the model to take the time to think. This technique has been found to increase performance on a variety of tasks.
- Called chain-of-thought (CoT) prompting, this is usually done by prepending the prompt with instructions for the LLM to describe how it could arrive at the answer

### 3. Retrieval-Augmented Generation
Retrieval-augmented generation (RAG) consists of finding relevant pieces of text, also known as context, such as facts you’d find in an encyclopedia and including that context in the prompt. The RAG technique can (and in real applications should) be combined with CoT, but for simplicity we’ll use these techniques one at a time here.

### 4. Few-Shot Prompting 
This consists of providing the LLM with examples of other questions and the correct answers, which enables the LLM to learn how to perform a new task without going through additional training or fine-tuning.
##### Static few-shot prompting
The most basic version of few-shot prompting is to assemble a predetermined list of a small number of examples that you include in the prompt.
##### Dynamic few-shot prompting
If you assemble a dataset of many examples, you can instead pick the most relevant examples for each new query


## LangChain and why it's important
LangChain was one of the earliest open source libraries to provide LLM and prompting building blocks and the tooling to reliably combine them into larger applications.
