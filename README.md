# pg_gembed

## Generate Embeddings directly in PostgreSQL

A PostgreSQL extension that brings **in-database embedding generation** directly into PostgreSQL, implemented as part
of the [Gembed](https://github.com/JoelDiaz222/pg_gembed) architecture.

The extension is a thin adapter that marshals PostgreSQL types into the C ABI of the portable Gembed Rust core
(`libgembed`), which handles model loading and inference locally — no external microservices required.

## Features

- 🚀 **Self-contained**: Generate embeddings without external API calls
- ⚡ **Fast**: Rust-powered inference with backend-level model caching
- 🔒 **Private**: Your data never leaves the database host
- 💰 **Cost-effective**: No per-token API fees, predictable infrastructure costs
- 🎯 **Simple**: Just SQL functions, no orchestration required
- 🔄 **Flexible**: Pluggable backends — embed_anything, FastEmbed, ORT, gRPC, HTTP

## Architecture

```
┌──────────────────────────────────────────────┐
│             PostgreSQL Query                 │
│   (e.g. SELECT embed_texts(...))             │
└───────────────────────┬──────────────────────┘
                        │  SQL / UDF interface
                        ▼
┌──────────────────────────────────────────────┐
│       PostgreSQL C Extension (pg_gembed)     │
│  - Registered via CREATE FUNCTION            │
│  - Marshals Datum types → C ABI types        │
└───────────────────────┬──────────────────────┘
                        │  C FFI
                        ▼
┌──────────────────────────────────────────────┐
│         Rust Core Library (libgembed)        │
│  Backends: embed_anything / FastEmbed /      │
│            ORT / gRPC / HTTP                 │
└──────────────────────────────────────────────┘
```

**Key design decisions:**

- **C FFI boundary**: The PostgreSQL C extension calls into the Rust core via a stable C ABI, keeping the core
  independent of the database engine.
- **Backend/model ID caching**: Backend and model names are resolved to integer IDs on first use and cached
  per-connection, minimising FFI round-trips.
- **Flat memory layout**: Embeddings are returned as a contiguous `float32` buffer for optimal cache performance and
  zero-copy transfer.

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

For GPU-accelerated inference, install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and add the
following lines to the [`Makefile`](./Makefile), adjusting `CUDA_LIB_DIR` if your CUDA libraries are not on the default
path:

```makefile
CUDA_LIB_DIR = /usr/local/cuda/lib64
 
SHLIB_LINK += \
    -L$(CUDA_LIB_DIR) \
    -Wl,-rpath,$(CUDA_LIB_DIR) \
    -lcudart \
    -lcuda \
    -lcurand \
    -lcublas
```

### Enable in PostgreSQL

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_gembed;
```

## Usage

### Text Embedding

```sql
-- Single string
SELECT embed_text(
    'embed_anything',
    'Qdrant/all-MiniLM-L6-v2-onnx',
    'Hello world'
);

-- Batch of strings
SELECT embed_texts(
    'embed_anything',
    'Qdrant/all-MiniLM-L6-v2-onnx',
    ARRAY['Hello world', 'Embedding in PostgreSQL']
);
```

### Semantic Search

```sql
CREATE TABLE articles (
    id        SERIAL PRIMARY KEY,
    title     TEXT,
    content   TEXT,
    embedding vector(384)
);

-- Generate embeddings on insert
INSERT INTO articles (title, content, embedding)
SELECT title, content,
       (embed_texts('embed_anything', 'Qdrant/all-MiniLM-L6-v2-onnx', ARRAY[content]))[1]
FROM (VALUES
    ('Understanding Transformers', 'Transformers use attention mechanisms.'),
    ('Graph Neural Networks', 'GNNs capture relational structure.')
) AS t(title, content);

-- Semantic search
SELECT id, title,
       embedding <=> (
        embed_texts(
            'embed_anything',
            'Qdrant/all-MiniLM-L6-v2-onnx',
            ARRAY['machine learning']
        )
    )[1] AS distance
FROM articles
ORDER BY distance
LIMIT 10;
```

### Zero-Shot Image Classification

```sql
-- Embed images and labels together (multimodal)
SELECT embed_multimodal(
    'grpc',
    'ViT-B-32',
    ARRAY[pg_read_binary_file('/path/to/image.jpg')],
    ARRAY['A diagram', 'A photo']
);
```

## Docker

A pre-built Docker image is provided for easily setting up a PostgreSQL instance with `pg_gembed` and its dependencies
pre-installed.

```bash
docker build -t pg_gembed .
docker run --name pg_gembed_container -d pg_gembed

docker exec -it pg_gembed_container psql
```

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_gembed;
```

## Docker Compose

To run the full stack (PostgreSQL with `pg_gembed` + gRPC embedding server):

```bash
docker-compose up --build
```

This starts:

- `pg_gembed`: PostgreSQL instance with the extension installed (port 5432)
- `grpc_server`: Python-based embedding server (port 50051)

## License

Licensed under the [Apache License 2.0](./LICENSE).

## Acknowledgments

- [pgvector](https://github.com/pgvector/pgvector) for the vector data type.
