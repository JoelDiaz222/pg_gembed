#ifndef GEMBED_INTERNAL_H
#define GEMBED_INTERNAL_H

#include "pg_gembed.h"
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "catalog/namespace.h"
#include "catalog/pg_enum.h"
#include "utils/syscache.h"
#include "utils/lsyscache.h"
#include "vector.h"

/* -------------------------------------------------------------------------
 * SRF context (shared by texts_with_ids and images_with_ids)
 * -------------------------------------------------------------------------
 */
typedef struct
{
    int    *ids;
    Datum  *vectors;
    void   *block;
    int     nitems;
    int     current;
} EmbeddingSRFContext;

/* -------------------------------------------------------------------------
 * Low-level helpers
 * -------------------------------------------------------------------------
 */
void        validate_embedder_and_model(text *embedder_text, text *model_text,
                                        int input_type,
                                        int *embedder_id, int *model_id);

StringSlice text_to_string_slice(text *t);
ByteSlice   bytea_to_byte_slice(bytea *b);

InputData   make_text_input(const StringSlice *texts, size_t n_texts);
InputData   make_image_input(const ByteSlice *images, size_t n_images);
InputData   make_multimodal_input(const StringSlice *texts, size_t n_texts,
                                  const ByteSlice *images, size_t n_images);
InputData   make_image_directory_input(const StringSlice *paths, size_t n_paths);

void        embed(int embedder_id, int model_id, const InputData *input,
                  EmbeddingBatch *batch);

Vector     *make_vector_from_batch(const EmbeddingBatch *batch, size_t index);
void        populate_vector_datums(const EmbeddingBatch *batch,
                                   void **out_block, Datum **out_vectors);
ArrayType  *construct_vector_array(const EmbeddingBatch *batch);

Datum       srf_return_next_embedding(FuncCallContext *funcctx,
                                      EmbeddingSRFContext *fctx,
                                      FunctionCallInfo fcinfo);

/* Resolves an input_type enum OID to its label string (palloc'd). */
char       *resolve_input_type_label(Oid enum_oid);

/* -------------------------------------------------------------------------
 * Single-item helpers
 * -------------------------------------------------------------------------
 */
Vector    *embed_one_text(text *embedder_text, text *model_text,
                          text *input_text);
Vector    *embed_one_image(text *embedder_text, text *model_text,
                           bytea *input_bytea);
ArrayType *embed_one_image_directory(text *embedder_text, text *model_text,
                                     text *path_text);

/* -------------------------------------------------------------------------
 * Batch helpers  (return NULL for an empty input array)
 * -------------------------------------------------------------------------
 */
ArrayType *embed_batch_text(text *embedder_text, text *model_text,
                            ArrayType *input_array);
ArrayType *embed_batch_image(text *embedder_text, text *model_text,
                             ArrayType *input_array);
ArrayType *embed_batch_image_directory(text *embedder_text, text *model_text,
                                       ArrayType *input_array);

/* -------------------------------------------------------------------------
 * Implementation functions (called by the PG wrappers in pg_gembed.c)
 * -------------------------------------------------------------------------
 */
Datum embed_texts_with_ids_impl(FunctionCallInfo fcinfo);
Datum embed_images_with_ids_impl(FunctionCallInfo fcinfo);
Datum embed_multimodal_impl(FunctionCallInfo fcinfo);
Datum embed_dispatch_impl(FunctionCallInfo fcinfo);
Datum embed_dispatch_array_impl(FunctionCallInfo fcinfo);

#endif /* GEMBED_INTERNAL_H */
