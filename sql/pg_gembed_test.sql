\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION pg_gembed;

-- Basic text embedding
SELECT vector_dims(embed_text('embed_anything', 'sentence-transformers/all-MiniLM-L6-v2', 'Hello world'));

-- Batch text embedding
SELECT array_length(embed_texts('embed_anything', 'sentence-transformers/all-MiniLM-L6-v2', ARRAY['Hello', 'World']), 1);

-- Text embedding with IDs
SELECT id, vector_dims(embedding) FROM embed_texts_with_ids('embed_anything', 'sentence-transformers/all-MiniLM-L6-v2', ARRAY[1, 2], ARRAY['Hello', 'World']);

-- Image tests
-- Note: Using absolute path as requested for this specific environment integration.
-- Ideally, images should be in a location accessible to the postgres user relative to the test runner.
\set image_path '/home/joel/dev/pg_gembed/image/Geralt.jpg'

-- Single image embedding
SELECT vector_dims(embed_image('embed_anything', 'openai/clip-vit-base-patch32', pg_read_binary_file(:'image_path')));

-- Batch image embedding
SELECT array_length(embed_images('embed_anything', 'openai/clip-vit-base-patch32', ARRAY[pg_read_binary_file(:'image_path')]), 1);

-- Image embedding with IDs
SELECT id, vector_dims(embedding) FROM embed_images_with_ids('embed_anything', 'openai/clip-vit-base-patch32', ARRAY[1], ARRAY[pg_read_binary_file(:'image_path')]);

-- Multimodal embedding
SELECT array_length(embed_multimodal('embed_anything', 'openai/clip-vit-base-patch32', ARRAY[pg_read_binary_file(:'image_path')], ARRAY['A cool image']), 1);

-- Image directory tests
\set repo_root '/home/joel/dev/pg_gembed/image'

-- Single directory embedding
SELECT array_length(embed_image_directory('embed_anything', 'openai/clip-vit-base-patch32', :'repo_root'), 1);
SELECT vector_dims((embed_image_directory('embed_anything', 'openai/clip-vit-base-patch32', :'repo_root'))[1]);

-- Batch directory embedding
SELECT array_length(embed_image_directories('embed_anything', 'openai/clip-vit-base-patch32', ARRAY[:'repo_root']), 1);
SELECT vector_dims((embed_image_directories('embed_anything', 'openai/clip-vit-base-patch32', ARRAY[:'repo_root']))[1]);
