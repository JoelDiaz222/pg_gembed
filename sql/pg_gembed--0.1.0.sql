-- Input type enum used by the polymorphic embed() dispatcher
CREATE TYPE input_type AS ENUM (
    'text',
    'image',
    'image_directory'
);

COMMENT ON TYPE input_type IS
    'Discriminator for the polymorphic embed() function. Selects which typed embedding function is called.';

-- Core embedding generation functions
CREATE FUNCTION embed_text(
    backend text,
    model text,
    input text
)
    RETURNS vector
AS
'MODULE_PATHNAME',
'embed_text'
    LANGUAGE C
    STRICT
    PARALLEL SAFE;

CREATE FUNCTION embed_texts(
    backend text,
    model text,
    texts text[]
)
    RETURNS vector[]
AS
'MODULE_PATHNAME',
'embed_texts'
    LANGUAGE C
    STRICT
    PARALLEL SAFE;

CREATE OR REPLACE FUNCTION embed_texts_with_ids(
    backend text,
    model text,
    ids integer[],
    texts text[]
)
    RETURNS TABLE
            (
                id        integer,
                embedding vector
            )
AS
'MODULE_PATHNAME',
'embed_texts_with_ids'
    LANGUAGE C
    STRICT
    PARALLEL SAFE;

CREATE FUNCTION embed_image(
    backend text,
    model text,
    input bytea
)
    RETURNS vector
AS
'MODULE_PATHNAME',
'embed_image'
    LANGUAGE C
    STRICT
    PARALLEL SAFE;

CREATE FUNCTION embed_images(
    backend text,
    model text,
    images bytea[]
)
    RETURNS vector[]
AS
'MODULE_PATHNAME',
'embed_images'
    LANGUAGE C
    STRICT
    PARALLEL SAFE;

CREATE OR REPLACE FUNCTION embed_images_with_ids(
    backend text,
    model text,
    ids integer[],
    images bytea[]
)
    RETURNS TABLE
            (
                id        integer,
                embedding vector
            )
AS
'MODULE_PATHNAME',
'embed_images_with_ids'
    LANGUAGE C
    STRICT
    PARALLEL SAFE;

CREATE FUNCTION embed_image_directory(
    backend text,
    model text,
    path text
)
    RETURNS vector[]
AS
'MODULE_PATHNAME',
'embed_image_directory'
    LANGUAGE C
    STRICT
    PARALLEL SAFE;

CREATE FUNCTION embed_image_directories(
    backend text,
    model text,
    paths text[]
)
    RETURNS vector[]
AS
'MODULE_PATHNAME',
'embed_image_directories'
    LANGUAGE C
    STRICT
    PARALLEL SAFE;

CREATE FUNCTION embed_multimodal(
    backend text,
    model text,
    images bytea[] DEFAULT NULL,
    texts text[] DEFAULT NULL
)
    RETURNS vector[]
AS
'MODULE_PATHNAME',
'embed_multimodal'
    LANGUAGE C
    PARALLEL SAFE;

COMMENT ON FUNCTION embed_texts(text, text, text[]) IS
    'Generate embeddings for an array of text inputs using the specified backend and model';

COMMENT ON FUNCTION embed_text(text, text, text) IS
    'Generate an embedding for a single text input using the specified backend and model';

COMMENT ON FUNCTION embed_texts_with_ids(text, text, integer[], text[]) IS
    'Generate embeddings with associated IDs, returning a table of (id, embedding) pairs';

COMMENT ON FUNCTION embed_image(text, text, bytea) IS
    'Generate an embedding for a single image input using the specified backend and model';

COMMENT ON FUNCTION embed_images(text, text, bytea[]) IS
    'Generate embeddings for an array of image inputs using the specified backend and model';

COMMENT ON FUNCTION embed_images_with_ids(text, text, integer[], bytea[]) IS
    'Generate embeddings for images with associated IDs, returning a table of (id, embedding) pairs';

COMMENT ON FUNCTION embed_multimodal(text, text, bytea[], text[]) IS
    'Generate embeddings from multimodal inputs (images and/or text). At least one input must be provided.';

COMMENT ON FUNCTION embed_image_directory(text, text, text) IS
    'Generate embeddings for all images in a directory using the specified backend and model';

COMMENT ON FUNCTION embed_image_directories(text, text, text[]) IS
    'Generate embeddings for all images in multiple directories using the specified backend and model';

-- Background worker schema and tables
CREATE SCHEMA IF NOT EXISTS gembed;

CREATE TABLE gembed.embedding_jobs
(
    job_id            SERIAL PRIMARY KEY,
    source_schema     TEXT      DEFAULT 'public',
    source_table      TEXT NOT NULL,
    source_column     TEXT NOT NULL,
    source_id_column  TEXT NOT NULL,
    target_schema     TEXT      DEFAULT 'public',
    target_table      TEXT NOT NULL,
    target_column     TEXT NOT NULL,
    backend           TEXT NOT NULL,
    model             TEXT NOT NULL,
    enabled           BOOLEAN   DEFAULT true,
    last_processed_id INTEGER   DEFAULT 0,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_run_at       TIMESTAMP
);

CREATE INDEX idx_jobs_enabled
    ON gembed.embedding_jobs (enabled)
    WHERE enabled = true;

-- View for job status
CREATE VIEW gembed.job_status AS
SELECT j.job_id,
       j.source_schema || '.' || j.source_table || '.' || j.source_column AS source,
       j.target_schema || '.' || j.target_table || '.' || j.target_column AS target,
       j.backend,
       j.model,
       j.enabled,
       j.last_processed_id,
       j.last_run_at,
       j.created_at,
       CASE
           WHEN j.last_run_at IS NULL THEN 'never run'
           WHEN j.last_run_at < NOW() - INTERVAL '1 hour' THEN 'stale'
           WHEN j.enabled THEN 'active'
           ELSE 'disabled'
           END                                                            AS status
FROM gembed.embedding_jobs j;

-- Polymorphic dispatcher: routes to the appropriate typed embed function
-- based on the input_type enum value.
--
-- Supported combinations:
--   input_type 'text'            + text input   -> embed_text()
--   input_type 'image'           + bytea input  -> embed_image()
--   input_type 'image_directory' + text input   -> embed_image_directory()
--
-- For batch / multimodal use cases, call the corresponding typed functions
-- (embed_texts, embed_images, embed_multimodal, ...) directly.
CREATE FUNCTION embed(
    backend_name text,
    model_name text,
    input anynonarray,
    type input_type
)
    RETURNS vector
AS
'MODULE_PATHNAME',
'embed_dispatch'
    LANGUAGE C
    STRICT
    PARALLEL SAFE;

COMMENT ON FUNCTION embed(text, text, anynonarray, input_type) IS
    'Polymorphic embedding dispatcher (C). Pass a scalar text or bytea (or a '
    'directory path) and the matching input_type to embed a single item. '
    'For arrays use the anyarray overload.';

-- Array overload: embed a batch of texts or images in one call.
--
-- Supported combinations:
--   input_type 'text'            + text[]  input  -> embed_batch_text()
--   input_type 'image'           + bytea[] input  -> embed_batch_image()
--   input_type 'image_directory' + text[]  input  -> embed_batch_image_directory()
CREATE FUNCTION embed(
    backend_name text,
    model_name text,
    input anyarray,
    type input_type
)
    RETURNS vector[]
AS
'MODULE_PATHNAME',
'embed_dispatch_array'
    LANGUAGE C
    PARALLEL SAFE;

COMMENT ON FUNCTION embed(text, text, anyarray, input_type) IS
    'Polymorphic batch embedding dispatcher (C). Pass a text[], bytea[], or text[] '
    '(directory paths) together with the matching input_type to embed a whole '
    'array in one call. Returns NULL for an empty array.';
