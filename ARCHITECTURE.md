System Architecture: Local Multi-User RAG
1. Overview
This system is a privacy-first, local Retrieval-Augmented Generation (RAG) pipeline designed to ingest, index, and query personal email data. 
It is architected to run entirely on consumer hardware (e.g., NVIDIA RTX 4060) without sending data to external API providers like OpenAI or Anthropic. 
The system supports multi-tenancy, ensuring strict data isolation between users.

2. Component Design
2.1 Large Language Model (LLM)
  -Model: Llama 3 (8B Instruct)
  -Runtime: Ollama (Local Inference Server)
  -Rationale: Llama 3 8B offers the best balance of reasoning capability and VRAM usage. It fits comfortably within the 8GB VRAM buffer of the RTX 4060 while utilizing 4-bit quantization for high-speed inference (approx. 50-70 tokens/sec).

2.2 Vector Database
  -Store: PostgreSQL with pgvector extension
  -Deployment: Docker Container
  -Rationale: Unlike lightweight stores (Chroma/FAISS), PostgreSQL provides production-grade reliability and persistence. 
  pgvector enables vector operations directly within SQL, allowing for robust hybrid search (combining semantic similarity with standard metadata filtering), which is crucial for the multi-user requirement.

2.3 Embeddings
  -Model: nomic-embed-text-v1.5
  -Rationale: A high-performance local embedding model that outperforms OpenAI's text-embedding-ada-002 on several benchmarks while maintaining a smaller footprint. It supports dynamic dimensionality, though we utilize the standard 768-dimensional output.

3. Data Pipeline & Ingestion
The ingestion pipeline follows a strict Extract-Transform-Load (ETL) pattern:
  -Extract: Google Gmail API fetches messages via OAuth 2.0 readonly scopes.
  -Transform:
    -Parsing: HTML bodies are cleaned to plain text; PDF attachments are parsed via pypdf.
    -Chunking: RecursiveCharacterTextSplitter segments long chains (1000 char chunks / 200 overlap) to preserve context boundaries.
    -Metadata Extraction: Sender, Recipient, Date, and Subject are extracted for filtering.
  -Load: Documents are embedded and pushed to pgvector with a strict user_id metadata tag.

4. Multi-User Isolation Strategy
To satisfy the "Bonus Challenge," the system implements Row-Level Security via Metadata Filtering rather than separate database tables.
  -Mechanism: Every read operation to the Vector DB is mandated to include a filter: {'user_id': current_session_user}.
  -Security: The user_id is derived directly from the OAuth token profile, not user input, preventing spoofing.
  -Scalability: This approach allows thousands of users to coexist in a single index without the overhead of managing thousands of separate collections.

5. RAG Retrieval Strategy
The system employs a Hybrid Retrieval approach to address common RAG limitations:
  -Semantic Search: Finds emails conceptually related to the query.
  -Precision Filtering: A UI-driven metadata filter allows users to narrow the search space by "Sender" or "Subject" before vector comparison. This solves the "Needle in a Haystack" problem, where relevant emails are missed because they share low semantic overlap with the query.
