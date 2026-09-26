# UzPost AI Chatbot

## Multi-Agent RAG-Based Intelligent Assistant for UzPost

UzPost AI Chatbot is an AI-powered customer support system designed for **Uzbekistan Post (UzPost)**.

The system combines **hybrid information retrieval, semantic search, keyword search, reranking, intent classification, conversational context, and LLM-based response generation** to provide accurate answers about postal services, tariffs, parcel tracking, locations, delivery services, and other UzPost-related topics.

The system supports:

* 🇺🇿 Uzbek
* 🇷🇺 Russian
* 🇬🇧 English

The primary focus is on reliable Uzbek-language customer support and integration with UzPost backend services.

---

## Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [Architecture](#architecture)
* [AI Pipeline](#ai-pipeline)
* [Intent Routing](#intent-routing)
* [Hybrid RAG](#hybrid-rag)
* [Knowledge Base](#knowledge-base)
* [Embedding Model](#embedding-model)
* [Vector Database](#vector-database)
* [Retrieval Pipeline](#retrieval-pipeline)
* [Reranking](#reranking)
* [LLM Integration](#llm-integration)
* [Conversation Memory](#conversation-memory)
* [Caching](#caching)
* [Backend Integration](#backend-integration)
* [API](#api)
* [Response Flow](#response-flow)
* [Configuration](#configuration)
* [Running the Project](#running-the-project)
* [Performance Considerations](#performance-considerations)
* [Security and Rate Limiting](#security-and-rate-limiting)
* [Project Structure](#project-structure)
* [Example Conversations](#example-conversations)
* [Future Improvements](#future-improvements)

---

# Overview

Traditional FAQ chatbots rely on predefined answers and keyword matching.

UzPost AI Chatbot uses a **retrieval-augmented generation (RAG)** architecture instead.

```text
User Question
      │
      ▼
Intent Detection
      │
      ▼
Query Processing
      │
      ▼
Hybrid Retrieval
 ┌────┼──────────────┐
 │    │              │
 ▼    ▼              ▼
Dense BM25        ColBERT
 │    │              │
 └────┼──────────────┘
      ▼
Candidate Fusion
      │
      ▼
Reranking
      │
      ▼
Context Construction
      │
      ▼
LLM
      │
      ▼
Final Answer
```

For operational questions, the chatbot can also route the request to dedicated backend logic instead of relying only on retrieved documents.

For example:

```text
"Posilkam qayerda?"
        │
        ▼
   Tracking Intent
        │
        ▼
UzPost Tracking API
        │
        ▼
Current Shipment Status
```

---

# Key Features

* Multi-language AI assistant
* Uzbek-first conversational experience
* Multi-agent architecture
* Intent-based routing
* Hybrid RAG
* Dense vector retrieval
* BM25 keyword retrieval
* ColBERT-based retrieval/reranking
* BGE-M3 embeddings
* Qdrant vector database
* Redis-based conversational state
* LLM-powered answer generation
* Backend API integration
* Parcel tracking
* Tariff information
* Location lookup
* Service information
* Context-aware follow-up questions
* Entity memory
* Intent history
* Response caching
* Rate limiting
* FastAPI backend
* Async request processing
* Local and cloud LLM support

---

# Architecture

```text
                         ┌──────────────────┐
                         │   Web / Mobile   │
                         │   / Telegram     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    FastAPI       │
                         │    API Layer     │
                         └────────┬─────────┘
                                  │
                                  ▼
                     ┌────────────────────────┐
                     │ Conversation Manager   │
                     │                        │
                     │ History                │
                     │ Entity Memory          │
                     │ Intent History         │
                     │ Pause / Resume         │
                     └───────────┬────────────┘
                                 │
                                 ▼
                       ┌───────────────────┐
                       │ Intent Router      │
                       └─────────┬─────────┘
                                 │
             ┌───────────────────┼────────────────────┐
             │                   │                    │
             ▼                   ▼                    ▼
        Tracking Agent      Price Agent         Location Agent
             │                   │                    │
             │                   │                    │
             ▼                   ▼                    ▼
        UzPost API          Calculator          Location Search
             │
             │
             └───────────────────┐
                                 │
                                 ▼
                         ┌────────────────┐
                         │    RAG Agent   │
                         └───────┬────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
                  BGE-M3       BM25        ColBERT
                    │            │            │
                    └────────────┼────────────┘
                                 ▼
                              Qdrant
                                 │
                                 ▼
                             Reranker
                                 │
                                 ▼
                               LLM
                                 │
                                 ▼
                           Final Answer
```

---

# AI Pipeline

The main processing pipeline is:

```text
User Message
     │
     ▼
Language / Query Understanding
     │
     ▼
Intent Classification
     │
     ▼
Entity Extraction
     │
     ▼
Conversation Context
     │
     ▼
Agent Routing
     │
     ├── Tracking
     ├── Price
     ├── Location
     ├── FAQ / RAG
     └── Off-topic
             │
             ▼
      Retrieval / Backend API
             │
             ▼
         Context
             │
             ▼
            LLM
             │
             ▼
        Final Response
```

---

# Intent Routing

The chatbot does not send every request directly to the LLM.

Instead, user requests are classified and routed to specialized processing paths.

Current intent categories include:

```text
tracking
price
location
faq
offtopic
```

This architecture reduces unnecessary LLM usage and allows deterministic backend logic to handle operational queries.

---

## Tracking Intent

Example:

```text
Where is my parcel?
```

or:

```text
Posilkаm qayerda?
```

Flow:

```text
User
 ↓
Tracking Intent
 ↓
Tracking Agent
 ↓
UzPost Tracking API
 ↓
Tracking Result
 ↓
LLM / Response Formatter
 ↓
User
```

---

## Price Intent

Example:

```text
How much does it cost to send a parcel to Samarkand?
```

The system can use tariff information and calculation logic instead of relying solely on generated text.

---

## Location Intent

Example:

```text
Where is the nearest post office?
```

The location flow can process location-related information separately from general FAQ retrieval.

---

## FAQ / RAG Intent

General questions are routed to the RAG pipeline.

Example:

```text
What documents are required to send an international parcel?
```

Flow:

```text
Question
   ↓
Embedding
   ↓
Dense Retrieval
   +
BM25 Retrieval
   +
ColBERT
   ↓
Candidate Documents
   ↓
Reranking
   ↓
LLM
   ↓
Answer
```

---

## Off-topic Intent

Questions unrelated to UzPost are handled separately.

Example:

```text
Who won the football match yesterday?
```

The chatbot should not unnecessarily search the UzPost knowledge base for unrelated questions.

---

# Hybrid RAG

The chatbot uses multiple retrieval approaches instead of relying on a single vector similarity search.

The main retrieval components are:

```text
BGE-M3 Dense Retrieval
        +
BM25 Keyword Retrieval
        +
ColBERT / Late Interaction Retrieval
        ↓
Candidate Fusion
        ↓
Reranking
        ↓
Final Context
```

This is especially useful for Uzbek postal terminology, addresses, service names, tariff codes, and technical terms where exact keyword matching can be as important as semantic similarity.

---

# Knowledge Base

The knowledge base contains structured and unstructured information related to UzPost.

Typical information includes:

* Postal services
* Tariffs
* Delivery conditions
* Parcel rules
* EMS services
* International shipping
* Domestic shipping
* Branch information
* Service categories
* Frequently asked questions
* Operational instructions
* Postal terminology

Documents are processed before being inserted into the retrieval system.

---

# Document Processing

A typical ingestion pipeline is:

```text
Source Documents
      │
      ▼
Text Extraction
      │
      ▼
Cleaning
      │
      ▼
Chunking
      │
      ▼
Metadata Generation
      │
      ▼
BGE-M3 Embedding
      │
      ▼
Qdrant
```

Each chunk can contain metadata such as:

```json
{
  "source": "uzpost_document.pdf",
  "category": "tariff",
  "language": "uz",
  "service": "EMS"
}
```

This metadata can later be used for filtering and retrieval.

---

# Embedding Model

The primary embedding model is:

```text
BAAI/bge-m3
```

BGE-M3 is used to generate semantic representations of documents and user queries.

The configured vector dimension is:

```text
1024
```

The system can use BGE-M3 for:

* Dense retrieval
* Sparse retrieval
* ColBERT-style retrieval

This allows the retrieval system to capture both semantic similarity and lexical relationships.

---

# Vector Database

The chatbot uses:

```text
Qdrant
```

Qdrant stores the document vectors and associated metadata.

Example collections used by the system include:

```text
uzpost_chunks_v1
uzpost_services_v1
uzpost_categories_v1
```

A legacy / consolidated knowledge collection may also be used depending on the deployment version.

---

# Retrieval Pipeline

The retrieval stage combines several retrieval strategies.

## Dense Retrieval

The user query is converted into a BGE-M3 embedding.

```text
User Query
    ↓
BGE-M3
    ↓
1024-dimensional vector
    ↓
Qdrant
    ↓
Semantic candidates
```

Dense retrieval is useful when the wording of the question differs from the wording used in the source document.

---

## BM25 Retrieval

BM25 provides lexical matching.

It is particularly useful for:

* Service names
* Postal codes
* Exact terminology
* Addresses
* Product names
* Technical terms

The implementation uses:

```text
BM25Okapi
```

through the `rank_bm25` package.

---

## ColBERT

ColBERT-style retrieval provides fine-grained query-document interaction.

The architecture can therefore combine:

```text
Semantic similarity
+
Keyword similarity
+
Token-level interaction
```

instead of depending on one retrieval signal.

---

# Candidate Fusion

The results from different retrieval methods are combined before final context selection.

Conceptually:

```text
Dense Results
     │
     ├─────────┐
BM25 Results  │
     │        │
     ├────────┤
ColBERT      │
Results      │
     │        │
     └────────┘
          │
          ▼
      Candidate Pool
          │
          ▼
        Reranker
          │
          ▼
    Top Relevant Chunks
```

This improves robustness when a query contains both semantic and exact-match information.

---

# Reranking

Initial retrieval generates a candidate set.

A second stage is then used to determine which chunks are most relevant.

```text
Query
  │
  ▼
Initial Retrieval
  │
  ▼
Top-N Candidates
  │
  ▼
Reranking
  │
  ▼
Top-K Context
```

This reduces the amount of irrelevant information passed to the LLM.

---

# LLM Integration

The system supports multiple LLM backends.

Historically, cloud inference has been handled through:

```text
Groq
```

with models such as:

```text
Llama 3.x
```

The architecture also supports local inference for selected workloads.

This allows the system to separate:

```text
Cloud LLM
```

and:

```text
Local LLM
```

depending on latency, cost, privacy, and workload requirements.

---

# Agent Architecture

The chatbot uses an agent-oriented architecture built around:

```text
LangChain
+
LangGraph
```

Agents can be separated by responsibility.

Example:

```text
base_agent.py
tracking/agent.py
calculator/agent.py
location/handler.py
```

This allows each agent to have its own:

* Prompt
* Tools
* Processing logic
* Validation
* Backend integrations
* State handling

---

# Conversation Memory

The chatbot maintains conversational context.

The state manager can store:

* Recent conversation history
* Intent history
* Extracted entities
* Current conversation state
* Follow-up information

A Redis-based state layer is used for scalable temporary state management.

---

## History

The system keeps a limited recent conversation window rather than sending the entire conversation to the LLM.

This helps control:

* Token usage
* Latency
* Memory consumption
* Prompt size

---

# Entity Memory

The chatbot can preserve important entities from previous messages.

For example:

```text
User:
How much is delivery to Samarkand?

Assistant:
...

User:
And how long does it take?
```

The second question can be interpreted using the previously established context.

Conceptually:

```text
Current Query
     +
Previous Intent
     +
Known Entities
     ↓
Resolved Query
```

---

# Follow-up Questions

The system supports contextual follow-up processing.

For example:

```text
User:
How much does EMS cost?

Assistant:
Which destination city?

User:
Samarkand.
```

The system can use the previous intent and current entity information to continue the same task instead of starting a completely new conversation.

---

# Caching

The chatbot uses response caching for frequently repeated requests.

Different cache lifetimes can be used depending on intent.

Example configuration:

```text
Location cache → 24 hours
FAQ cache      → 6 hours
```

Caching reduces:

* LLM requests
* Vector database queries
* API calls
* Response latency

---

# Redis

Redis is used for temporary application state.

Possible state:

```text
Conversation history
Intent state
Entity memory
Session data
Cache
```

Conceptual architecture:

```text
FastAPI
   │
   ├── Redis → conversation state
   │
   ├── Qdrant → knowledge vectors
   │
   └── LLM → response generation
```

---

# Backend Integration

The chatbot can combine AI-generated responses with deterministic backend services.

This is especially important for information that should come from current operational systems.

Examples:

```text
Tracking
Tariff calculation
Branch locations
Service availability
```

Instead of asking the LLM to invent an answer:

```text
User
 ↓
Intent
 ↓
Backend API
 ↓
Real data
 ↓
Response
```

This architecture helps separate:

```text
Knowledge-based questions
```

from:

```text
Real-time operational questions
```

---

# API

The application is implemented using:

```text
FastAPI
```

The API layer is responsible for:

* Request validation
* Session handling
* Chat processing
* Rate limiting
* Agent orchestration
* Response generation

A typical request conceptually looks like:

```json
{
  "message": "Posilkam qayerda?",
  "session_id": "abc123"
}
```

The exact request/response schema depends on the deployed API version.

---

# Response Flow

A typical chatbot request:

```text
POST /chat
      │
      ▼
Validate Request
      │
      ▼
Load Conversation State
      │
      ▼
Classify Intent
      │
      ▼
Route to Agent
      │
      ├───────────────┐
      │               │
      ▼               ▼
Backend Tool         RAG
      │               │
      │          ┌────┼────┐
      │          ▼    ▼    ▼
      │        Dense BM25 ColBERT
      │          │    │    │
      │          └────┼────┘
      │               ▼
      │           Reranking
      │               │
      └───────┬───────┘
              ▼
             LLM
              │
              ▼
        Final Response
              │
              ▼
       Save Conversation
```

---

# Rate Limiting

The chatbot includes request rate limiting to protect the API and AI infrastructure.

The configured application can limit requests per client/session.

This is particularly important because a single chatbot request can trigger:

```text
Intent classification
+
Vector search
+
Reranking
+
LLM inference
```

and therefore consume considerably more resources than a conventional REST request.

---

# Configuration

Configuration is environment-driven.

A typical configuration can include:

```env
QDRANT_URL=http://localhost:6333

REDIS_URL=redis://localhost:6379

GROQ_API_KEY=...

GROQ_MODEL=...

EMBEDDING_MODEL=BAAI/bge-m3

RATE_LIMIT=20
```

Actual environment variable names depend on the deployed project version.

---

# Local AI Inference

The architecture can also be deployed with local LLM inference.

Conceptually:

```text
FastAPI
   │
   ▼
Agent Orchestrator
   │
   ▼
Local LLM Server
   │
   ▼
GPU
```

This is useful when:

* Data should remain inside the infrastructure
* API costs need to be reduced
* High request volume is expected
* A dedicated GPU is available

The same architecture can therefore support both:

```text
Cloud inference
```

and:

```text
Local inference
```

without changing the overall RAG design.

---

# GPU Acceleration

The embedding and retrieval pipeline can be accelerated using NVIDIA GPUs.

A GPU-enabled deployment is particularly useful for:

* BGE-M3 embedding generation
* Large-scale document ingestion
* Reranking
* Local LLM inference

The application can keep frequently used models loaded in GPU memory to avoid repeated model initialization.

---

# Performance Strategy

The chatbot is designed to minimize unnecessary expensive operations.

The main optimization principles are:

```text
1. Intent routing
        ↓
2. Avoid RAG when not required
        ↓
3. Avoid LLM when deterministic logic is sufficient
        ↓
4. Cache repeated requests
        ↓
5. Limit conversation history
        ↓
6. Retrieve only relevant context
        ↓
7. Reuse loaded models
```

---

# Production-Oriented Design

The system separates responsibilities into several layers:

```text
API Layer
     │
     ▼
Conversation Layer
     │
     ▼
Agent Layer
     │
     ▼
Retrieval / Tool Layer
     │
     ▼
AI / Backend Services
```

This makes it easier to independently scale:

* API workers
* LLM inference
* Qdrant
* Redis
* Background ingestion
* External UzPost APIs

---

# Knowledge Ingestion

A typical document ingestion process:

```text
PDF / DOCX / HTML / Structured Data
             │
             ▼
        Text Extraction
             │
             ▼
        Data Cleaning
             │
             ▼
           Chunking
             │
             ▼
       Metadata Creation
             │
             ▼
          BGE-M3
             │
             ▼
          Qdrant
```

The ingestion pipeline should preserve important metadata such as:

```text
document
category
service
language
source
section
```

This makes later retrieval and filtering more precise.

---

# Example Conversations

## Tariff Question

```text
User:
How much does it cost to send a parcel?

Assistant:
The delivery price depends on the destination,
service type, parcel parameters, and selected tariff.
```

The actual answer can be generated from the current UzPost knowledge base and tariff data.

---

## Tracking

```text
User:
Where is my parcel?

Assistant:
Please provide your tracking number.
```

After receiving the tracking number:

```text
User:
EE123456789UZ

Assistant:
[Tracking information retrieved from the backend]
```

---

## Location

```text
User:
Where is the nearest UzPost office?

Assistant:
[Location agent processes the request and returns
relevant branch information.]
```

---

## Contextual Follow-up

```text
User:
How much is EMS delivery?

Assistant:
The price depends on the destination.
Which city are you sending the parcel to?

User:
Samarkand.

Assistant:
[Continues the EMS pricing flow using the
previous intent and new destination entity.]
```

---

# Multi-Agent Design

The main advantage of the agent architecture is separation of responsibilities.

```text
                    ┌───────────────┐
                    │ User Request  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Intent Router │
                    └───────┬───────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
    Tracking            Calculator          Location
      Agent                Agent               Agent
        │                   │                   │
        ▼                   ▼                   ▼
    Tracking API       Tariff Logic        Location Data

                            │
                            ▼
                       ┌─────────┐
                       │   RAG   │
                       └────┬────┘
                            │
                            ▼
                           LLM
```

This prevents every request from becoming a generic LLM prompt.

---

# Why Hybrid Retrieval?

A single retrieval method has limitations.

For example, a user may ask:

```text
"EMS jo'natmasi qancha turadi?"
```

while the knowledge base contains:

```text
EMS international postal service tariff
```

Dense retrieval can recognize semantic similarity.

However, exact terms such as:

```text
EMS
tariff
service code
postal category
```

can benefit from keyword retrieval.

Therefore:

```text
Dense Retrieval
+
BM25
+
ColBERT
```

provides complementary retrieval signals.

---

# Data Flow

```text
                    User
                     │
                     ▼
                 FastAPI
                     │
                     ▼
             Conversation State
                     │
                     ▼
              Intent Detection
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
   Tracking        Price        Location
       │             │             │
       └─────────────┼─────────────┘
                     │
                     ▼
                   FAQ?
                     │
                     ▼
                  RAG
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      BGE-M3       BM25        ColBERT
        │            │            │
        └────────────┼────────────┘
                     ▼
                  Reranker
                     │
                     ▼
                  Context
                     │
                     ▼
                    LLM
                     │
                     ▼
                  Response
                     │
                     ▼
              Conversation State
```

---

# Project Structure

A representative project structure:

```text
UZPOST_AI_BOT/
│
├── main.py
├── config.py
├── requirements.txt
├── .env
│
├── agents/
│   ├── base_agent.py
│   ├── tracking/
│   │   └── agent.py
│   ├── calculator/
│   │   └── agent.py
│   └── location/
│       └── handler.py
│
├── rag/
│   ├── embeddings/
│   ├── retrieval/
│   ├── reranking/
│   └── pipeline/
│
├── services/
│   ├── tracking.py
│   ├── pricing.py
│   └── location.py
│
├── memory/
│   ├── conversation.py
│   └── redis_state.py
│
├── prompts/
│   └── ...
│
├── ingestion/
│   └── ...
│
└── static/
    └── ...
```

The exact directory structure may differ between deployments.

---

# Main Technologies

| Component                  | Technology        |
| -------------------------- | ----------------- |
| API                        | FastAPI           |
| Agent Framework            | LangChain         |
| Agent Orchestration        | LangGraph         |
| Embeddings                 | BGE-M3            |
| Vector Database            | Qdrant            |
| Keyword Retrieval          | BM25Okapi         |
| Late Interaction Retrieval | ColBERT           |
| State / Cache              | Redis             |
| LLM                        | Groq / Local LLM  |
| Backend Language           | Python            |
| Async API                  | FastAPI / asyncio |

---

# Core AI Components

## BGE-M3

Used for multilingual semantic representation and retrieval.

```text
BAAI/bge-m3
```

---

## Qdrant

Used for vector storage and similarity search.

```text
Qdrant
```

---

## BM25

Used for lexical retrieval:

```text
BM25Okapi
```

---

## LangChain

Used for:

* LLM integration
* Tools
* Agent components
* Prompt management

---

## LangGraph

Used for:

* Agent orchestration
* Conditional routing
* Stateful workflows
* Multi-step processing

---

## Redis

Used for:

* Conversation state
* Temporary memory
* Cache
* Session-related information

---

# Deployment Architecture

A production deployment can be organized as:

```text
                    Internet
                       │
                       ▼
                 Reverse Proxy
                       │
                       ▼
                 FastAPI API
                       │
        ┌──────────────┼───────────────┐
        │              │               │
        ▼              ▼               ▼
      Redis          Qdrant          LLM
        │                              │
        │                         ┌────┴────┐
        │                         │         │
        │                       Cloud     Local
        │                       LLM       LLM
        │
        ▼
 Conversation State

FastAPI
   │
   ├── UzPost APIs
   ├── Tracking
   ├── Pricing
   └── Location
```

---

# Security Considerations

For production deployment, the following should be configured:

* API authentication
* Restricted CORS
* Rate limiting
* API key protection
* Secret management through environment variables
* Request validation
* Input size limits
* Logging and monitoring
* Redis authentication
* Qdrant access control
* HTTPS
* Reverse proxy

API keys should never be committed to Git.

Example:

```env
GROQ_API_KEY=your-secret-key
```

should remain in `.env` and be excluded from version control.

---

# Monitoring

Important production metrics include:

```text
Requests / second
Average response latency
LLM latency
RAG latency
Embedding latency
Qdrant latency
Cache hit rate
Intent distribution
Error rate
Token usage
Rate-limit events
```

For AI quality monitoring:

```text
Retrieval relevance
Answer correctness
Hallucination rate
Fallback rate
User feedback
Intent classification accuracy
```

---

# Future Improvements

Potential future improvements include:

* Persistent conversation analytics
* Advanced Uzbek language reranking
* Better Uzbek-specific embeddings
* Fine-tuned intent classifier
* Streaming responses
* Voice input
* Uzbek speech-to-text integration
* Uzbek text-to-speech integration
* Call-center AI agent integration
* Tool-calling improvements
* Automated knowledge-base ingestion
* Document OCR pipeline
* Better citation / source attribution
* RAG evaluation framework
* Automated benchmark dataset
* Distributed inference
* GPU-based local LLM serving
* Advanced observability
* Conversation quality analytics

---

# Voice AI Integration

The architecture can be extended to support a call-center agent:

```text
Customer Voice
      │
      ▼
    STT
      │
      ▼
UzPost AI Agent
      │
 ┌────┼─────────┐
 ▼    ▼         ▼
RAG  Tools   Backend APIs
      │
      ▼
    LLM
      │
      ▼
    TTS
      │
      ▼
Customer Voice
```

This allows the same knowledge and agent architecture to be reused for both:

```text
Text Chatbot
```

and:

```text
Voice Call Center Agent
```

---

# Design Principles

The system follows several important principles:

### 1. Use deterministic logic where possible

Tracking and tariff calculations should use backend data rather than relying entirely on generative output.

### 2. Use RAG for knowledge

General postal information should be retrieved from the verified knowledge base.

### 3. Use the LLM for language understanding and generation

The LLM should synthesize retrieved information instead of inventing unsupported facts.

### 4. Preserve conversation context

Follow-up questions should be interpreted using previous intent and entities.

### 5. Minimize expensive inference

Caching, routing, and deterministic tools reduce unnecessary LLM calls.

### 6. Keep retrieval modular

Dense retrieval, BM25, ColBERT, and reranking should remain independently replaceable.

---

# End-to-End Example

```text
User:
"Chilonzorda qaysi pochta bo'limi ochiq?"

        │
        ▼

Intent Classification
        │
        ▼

Location Intent
        │
        ▼

Location Agent
        │
        ▼

UzPost Location Data
        │
        ▼

Relevant Branches
        │
        ▼

LLM Response Formatting
        │
        ▼

User receives answer
```

For a knowledge-base question:

```text
User:
"Xalqaro posilka yuborish uchun nimalar kerak?"

        │
        ▼

FAQ Intent
        │
        ▼

BGE-M3
        │
        ├── Dense Retrieval
        ├── BM25
        └── ColBERT
                 │
                 ▼
             Reranking
                 │
                 ▼
          Relevant Documents
                 │
                 ▼
                LLM
                 │
                 ▼
             Final Answer
```

---

# Conclusion

UzPost AI Chatbot is designed as a modular AI customer-support platform rather than a simple question-answering bot.

Its architecture combines:

```text
FastAPI
+
LangChain
+
LangGraph
+
BGE-M3
+
Qdrant
+
BM25
+
ColBERT
+
Redis
+
LLM
+
UzPost Backend APIs
```

The resulting system can combine:

* Retrieval-augmented generation
* Real-time backend information
* Multi-agent workflows
* Conversational memory
* Multilingual support
* Intelligent intent routing
* Hybrid information retrieval
* Local or cloud LLM inference

This architecture also provides a foundation for extending the chatbot into a full **UzPost AI customer-support and call-center platform**.

