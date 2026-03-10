\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION pg_gembed;

\set embedder               'embed_anything'
\set text_model             'sentence-transformers/all-MiniLM-L6-v2'
\set image_model            'openai/clip-vit-base-patch32'
\set image_path             '/home/joel/dev/pg_gembed/image/Geralt.jpg'
\set image_directory_path   '/home/joel/dev/pg_gembed/image'

-- =============================================================================
-- Text embedding functions
-- =============================================================================

-- embed_text: single text → vector
SELECT vector_dims(embed_text(:'embedder', :'text_model', 'Hello world')) > 0
    AS embed_text_returns_nonempty_vector;

-- embed_texts: text[] → vector[]
SELECT array_length(
           embed_texts(:'embedder', :'text_model', ARRAY['Hello', 'World']),
           1) = 2
    AS embed_texts_returns_two_vectors;

-- embed_texts: all dimensions must match
SELECT (SELECT COUNT(DISTINCT vector_dims(e))
        FROM  unnest(embed_texts(:'embedder', :'text_model',
                                 ARRAY['foo', 'bar', 'baz'])) AS e) = 1
    AS embed_texts_consistent_dims;

-- embed_texts_with_ids: SETOF (id, embedding)
SELECT id, vector_dims(embedding) > 0 AS has_embedding
FROM   embed_texts_with_ids(:'embedder', :'text_model',
                             ARRAY[1, 2],
                             ARRAY['Hello', 'World'])
ORDER  BY id;

-- =============================================================================
-- Image embedding functions
-- =============================================================================

-- embed_image: single bytea → vector
SELECT vector_dims(
           embed_image(:'embedder', :'image_model',
                       pg_read_binary_file(:'image_path'))) > 0
    AS embed_image_returns_nonempty_vector;

-- embed_images: bytea[] → vector[]
SELECT array_length(
           embed_images(:'embedder', :'image_model',
                        ARRAY[pg_read_binary_file(:'image_path')]),
           1) = 1
    AS embed_images_returns_one_vector;

-- embed_images_with_ids: SETOF (id, embedding)
SELECT id, vector_dims(embedding) > 0 AS has_embedding
FROM   embed_images_with_ids(:'embedder', :'image_model',
                              ARRAY[1],
                              ARRAY[pg_read_binary_file(:'image_path')])
ORDER  BY id;

-- embed_image_directory: text → vector[]
SELECT array_length(
           embed_image_directory(:'embedder', :'image_model', :'image_directory_path'),
           1) > 0
    AS embed_image_directory_returns_vectors;

-- embed_image_directories: text[] → vector[]
SELECT array_length(
           embed_image_directories(:'embedder', :'image_model',
                                   ARRAY[:'image_directory_path']),
           1) > 0
    AS embed_image_directories_returns_vectors;

-- =============================================================================
-- Multimodal embedding function
-- =============================================================================

-- embed_multimodal: images + texts → vector[]
SELECT array_length(
           embed_multimodal(:'embedder', :'image_model',
                            ARRAY[pg_read_binary_file(:'image_path')],
                            ARRAY['A cool image']),
           1) > 0
    AS embed_multimodal_returns_vectors;

-- embed_multimodal: texts only
SELECT array_length(
           embed_multimodal(:'embedder', :'image_model',
                            NULL,
                            ARRAY['Only text']),
           1) > 0
    AS embed_multimodal_text_only;

-- embed_multimodal: images only
SELECT array_length(
           embed_multimodal(:'embedder', :'image_model',
                            ARRAY[pg_read_binary_file(:'image_path')],
                            NULL),
           1) > 0
    AS embed_multimodal_image_only;

-- =============================================================================
-- Polymorphic embed() dispatcher — scalar overload (anyelement → vector)
-- =============================================================================

-- dispatch: text input
SELECT vector_dims(
           embed(:'embedder', :'text_model',
                 'Hello world'::text, 'text'::input_type)) > 0
    AS embed_dispatch_text;

-- dispatch: image input
SELECT vector_dims(
           embed(:'embedder', :'image_model',
                 pg_read_binary_file(:'image_path'), 'image'::input_type)) > 0
    AS embed_dispatch_image;

-- dispatch: image_directory returns the first vector from the directory
SELECT vector_dims(
           embed(:'embedder', :'image_model',
                 :'image_directory_path'::text, 'image_directory'::input_type)) > 0
    AS embed_dispatch_image_directory;

-- =============================================================================
-- Polymorphic embed() dispatcher — array overload (anyarray → vector[])
-- =============================================================================

-- dispatch array: text[]
SELECT array_length(
           embed(:'embedder', :'text_model',
                 ARRAY['Hello', 'World'], 'text'::input_type),
           1) = 2
    AS embed_dispatch_array_text;

-- dispatch array: consistent dimensions across the batch
SELECT (SELECT COUNT(DISTINCT vector_dims(e))
        FROM   unnest(
                   embed(:'embedder', :'text_model',
                         ARRAY['alpha', 'beta', 'gamma'],
                         'text'::input_type)) AS e) = 1
    AS embed_dispatch_array_text_consistent_dims;

-- dispatch array: bytea[]
SELECT array_length(
           embed(:'embedder', :'image_model',
                 ARRAY[pg_read_binary_file(:'image_path')],
                 'image'::input_type),
           1) = 1
    AS embed_dispatch_array_image;

-- dispatch array: text[] paths (image_directory)
SELECT array_length(
           embed(:'embedder', :'image_model',
                 ARRAY[:'image_directory_path'], 'image_directory'::input_type),
           1) > 0
    AS embed_dispatch_array_image_directory;

-- dispatch array: empty array → NULL
SELECT embed(:'embedder', :'text_model',
             ARRAY[]::text[], 'text'::input_type) IS NULL
    AS embed_dispatch_array_empty_returns_null;
