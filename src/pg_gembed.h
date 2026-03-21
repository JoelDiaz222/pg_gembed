#ifndef PG_GEMBED_H
#define PG_GEMBED_H

#include "stddef.h"

#define INPUT_TYPE_TEXT 0
#define INPUT_TYPE_IMAGE 1
#define INPUT_TYPE_MULTIMODAL 2
#define INPUT_TYPE_IMAGE_DIRECTORY 3

/* Structure for storing generated embeddings */
typedef struct
{
    float *data;
    size_t n_vectors;
    size_t dim;
} EmbeddingBatch;

/* Structure for passing text data */
typedef struct
{
    const char *ptr;
    size_t len;
} StringSlice;

/* Structure for passing binary data */
typedef struct
{
    const unsigned char *ptr;
    size_t len;
} ByteSlice;

/* Generic input data structure */
typedef struct
{
    int input_type;                    /* INPUT_TYPE_* constant */
    const ByteSlice *binary_data;      /* For images, audio, etc. */
    size_t n_binaries;                   /* Number of binary items */
    const StringSlice *text_data;      /* For text inputs */
    size_t n_texts;                     /* Number of text items */
} InputData;

/* Validates the backend name and returns its ID (-1 if non-existent) */
extern int validate_backend(const char *name);

/* Validates the model name for a given backend and returns model ID */
extern int validate_model(int backend_id, const char *model, int input_type);

/* Generates embeddings for the given input data */
extern int generate_embeddings(
    int backend_id,
    int model_id,
    const InputData *input_data,
    EmbeddingBatch *out_batch
);

/* Frees memory allocated for an embedding batch */
extern void free_embedding_batch(EmbeddingBatch *batch);

#endif /* PG_GEMBED_H */
