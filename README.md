# pg_gembed

## Generate Embeddings directly in PostgreSQL

A PostgreSQL extension that brings ML-powered vector embedding generation directly into your database. Supports
local embedding generation using FastEmbed-rs, or through a local gRPC server.

## Features

- 🚀 **Self-contained**: Generate embeddings without external API calls
- ⚡ **Fast**: Rust-powered inference with thread-local model caching
- 🔒 **Private**: Your data is not sent to external inference providers
- 💰 **Cost-effective**: No per-token API fees, predictable infrastructure costs
- 🎯 **Simple**: Just SQL functions, no orchestration required
- 🔄 **Flexible**: Support for both local (FastEmbed) and remote (gRPC) inference

## Installation

### Prerequisites

- PostgreSQL 17+
- [pgvector](https://github.com/pgvector/pgvector) extension
- Rust toolchain (for building)

### Build from Source

```bash
git clone --recurse-submodules https://github.com/JoelDiaz222/pg_gembed
cd pg_gembed

make install
```

### Enable in PostgreSQL

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_gembed;
```

## Usage

### Basic Embedding Generation

```sql
SELECT embed_text(
               'embed_anything',
               'sentence-transformers/all-MiniLM-L6-v2',
               'Hello world'
       );

SELECT embed_texts(
               'embed_anything',
               'sentence-transformers/all-MiniLM-L6-v2',
               ARRAY ['Hello world', 'Embedding in PostgreSQL']
       );
```

Returns an array of `vector` types compatible with pgvector.

### Bulk Insert with IDs

```sql
-- Generate and insert embeddings with associated IDs
INSERT INTO documents (id, embedding)
SELECT id, embedding
FROM embed_texts_with_ids(
        'fastembed',
        'Qdrant/all-MiniLM-L6-v2-onnx',
        ARRAY [1, 2, 3],
        ARRAY ['First document', 'Second document', 'Third document']
     );
```

## Zero-Shot Image Classification

### Basic Usage

```sql
SELECT embed_multimodal(
               'grpc',
               'ViT-B-32',
               ARRAY [pg_read_binary_file('/path/to/image.jpg')],
               ARRAY ['A diagram', 'A photo']
       );
```

### Classification Example

```sql
WITH inputs AS (SELECT '/path/to/image.jpg'                            AS img_path,
                       ARRAY ['Dog', 'Cat', 'Bird', 'Bat', 'Elephant'] AS labels),
     embeddings AS (SELECT labels,
                           embed_multimodal('grpc', 'ViT-B-32', ARRAY [pg_read_binary_file(img_path)],
                                            labels) AS all_vecs
                    FROM inputs)
SELECT labels[ordinality]                         AS predicted_label,
       (all_vecs[1] <=> all_vecs[ordinality + 1]) AS distance
FROM embeddings, UNNEST(labels) WITH ORDINALITY
ORDER BY distance ASC;
```

### Semantic Search Example

```sql
-- Create a table with embeddings
CREATE TABLE articles
(
    id        SERIAL PRIMARY KEY,
    title     TEXT,
    content   TEXT,
    embedding vector(384)
);

-- Generate embeddings during insert
INSERT INTO articles (title, content, embedding)
SELECT title,
       content,
       (embed_texts(
               'fastembed',
               'Qdrant/all-MiniLM-L6-v2-onnx',
               ARRAY [content]
        ))[1]
FROM (VALUES ('Understanding Transformers',
              'Transformers have revolutionized NLP by using attention mechanisms.'),
             ('Graph Neural Networks',
              'GNNs operate on graph structures to capture relationships.'),
             ('Reinforcement Learning Basics',
              'An introduction to RL concepts like agents and environments.')) AS t(title, content);

-- Perform semantic search
SELECT id,
       title,
       content,
       embedding <=> (SELECT (embed_texts(
               'fastembed',
               'Qdrant/all-MiniLM-L6-v2-onnx',
               ARRAY ['machine learning']
                              ))[1]) AS distance
FROM articles
ORDER BY distance
LIMIT 10;
```

## Architecture

```
┌────────────────────────────────────────────┐
│             PostgreSQL Query               │
│   (e.g. SELECT embed_texts(...))   │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────┐
│        PostgreSQL C Extension (FFI)       │
│  - Defined via CREATE FUNCTION ...        │
│  - Calls into extern "C" Rust functions   │
└──────────────────────┬────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│             Rust Core Library                │
│   ┌──────────────────────┐  ┌────────────┐   │
│   │   fastembed-rs       │  │ gRPC Client│   │
│   │ (local embedding)    │  │ (remote)   │   │
│   └──────────────────────┘  └────────────┘   │
└──────────────────────────────────────────────┘
```

**Key design decisions**:

- **Thread-local model caching**: Models loaded once per connection
- **Zero-copy FFI**: Direct memory transfer between Rust and PostgreSQL
- **Flat memory layout**: Contiguous vector storage for optimal cache performance

## Docker

A pre-built Docker image is provided for easily setting up a PostgreSQL instance with pg_gembed and its dependencies
pre-installed.

### Build and Run

```bash
docker build -t pg_gembed .
docker run --name pg_gembed_container -d pg_gembed
```

### Create the Extension

```bash
docker exec -it pg_gembed_container psql
CREATE EXTENSION vector;
CREATE EXTENSION pg_gembed;
```

### Access the Shell

```bash
docker exec -it --user root pg_gembed_container bash
```

## Docker Compose

To run the full stack (PostgreSQL with pg_gembed + gRPC Embedding Server), use Docker Compose:

```bash
docker-compose up --build
```

This starts:

- `pg_gembed`: PostgreSQL instance with the extension installed (port 5432)
- `grpc_server`: Python-based embedding server (port 50051)

## License

Licensed under the [Apache License 2.0](./LICENSE).

## Acknowledgments

- [pgvector](https://github.com/pgvector/pgvector) for the vector datatype
- [FastEmbed-rs](https://github.com/Anush008/fastembed-rs) for the embedding library
    - [A fork](https://github.com/JoelDiaz222/fastembed-rs) has been done to support returning a contiguous buffer of
      embeddings
- [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) for gRPC protocol reference
    - [A fork](https://github.com/JoelDiaz222/text-embeddings-inference) has been done for generating batches
      ofembeddings using gRPC
