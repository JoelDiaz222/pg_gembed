#include "pg_gembed.h"
#include "postgres.h"
#include "fmgr.h"
#include "funcapi.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "catalog/pg_type.h"
#include "catalog/namespace.h"
#include "vector.h"

PG_MODULE_MAGIC;

/* -------------------------------------------------------------------------
 * Helper Functions
 * -------------------------------------------------------------------------
 */

/*
 * Validates embedder and model arguments
 */
static void
validate_embedder_and_model(text *embedder_text, text *model_text, int input_type,
                    int *embedder_id, int *model_id)
{
    char *embedder_str = text_to_cstring(embedder_text);
    char *model_str = text_to_cstring(model_text);

    *embedder_id = validate_embedder(embedder_str);
    if (*embedder_id < 0)
        elog(ERROR, "Invalid embedder: %s", embedder_str);

    *model_id = validate_embedding_model(*embedder_id, model_str, input_type);
    if (*model_id < 0)
        elog(ERROR, "Model not allowed: %s", model_str);
}

/*
 * Convert a text Datum to a StringSlice
 */
static StringSlice
text_to_string_slice(text *t)
{
    StringSlice s;
    s.ptr = VARDATA_ANY(t);
    s.len = VARSIZE_ANY_EXHDR(t);
    return s;
}

/*
 * Convert a bytea Datum to a ByteSlice
 */
static ByteSlice
bytea_to_byte_slice(bytea *b)
{
    ByteSlice s;
    s.ptr = (unsigned char *)VARDATA_ANY(b);
    s.len = VARSIZE_ANY_EXHDR(b);
    return s;
}

/*
 * Initialize InputData for text inputs
 */
static InputData
make_text_input(const StringSlice *texts, size_t n_texts)
{
    InputData d = {0};
    d.input_type = INPUT_TYPE_TEXT;
    d.text_data = texts;
    d.n_texts = n_texts;
    return d;
}

/*
 * Initialize InputData for image inputs
 */
static InputData
make_image_input(const ByteSlice *images, size_t n_images)
{
    InputData d = {0};
    d.input_type = INPUT_TYPE_IMAGE;
    d.binary_data = images;
    d.n_binaries = n_images;
    return d;
}

/*
 * Initialize InputData for multimodal inputs
 */
static InputData
make_multimodal_input(const StringSlice *texts, size_t n_texts,
                      const ByteSlice *images, size_t n_images)
{
    InputData d = {0};
    d.input_type = INPUT_TYPE_MULTIMODAL;
    d.text_data = texts;
    d.n_texts = n_texts;
    d.binary_data = images;
    d.n_binaries = n_images;
    return d;
}

/*
 * Generate embeddings using the specified embedder and model
 */
static void
embed(int embedder_id, int model_id, const InputData *input,
                  EmbeddingBatch *batch)
{
    int err = generate_embeddings(embedder_id, model_id, input, batch);
    if (err < 0)
    {
        free_embedding_batch(batch);
        elog(ERROR, "Embedding generation failed (code=%d)", err);
    }
}

/*
 * Create a Vector from a batch at a specific index
 */
static Vector *
make_vector_from_batch(const EmbeddingBatch *batch, size_t index)
{
    Vector *v = (Vector *)palloc(VECTOR_SIZE(batch->dim));
    SET_VARSIZE(v, VECTOR_SIZE(batch->dim));
    v->dim = batch->dim;
    v->unused = 0;
    memcpy(v->x, batch->data + index * batch->dim, sizeof(float) * batch->dim);
    return v;
}

/* -------------------------------------------------------------------------
 * Text Embedding Functions
 * -------------------------------------------------------------------------
 */

PG_FUNCTION_INFO_V1(embed_text);

Datum
embed_text(PG_FUNCTION_ARGS)
{
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    text *input_text = PG_GETARG_TEXT_P(2);
    int embedder_id, model_id;
    EmbeddingBatch batch;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_TEXT,
                        &embedder_id, &model_id);

    StringSlice c_input = text_to_string_slice(input_text);
    InputData input_data = make_text_input(&c_input, 1);

    embed(embedder_id, model_id, &input_data, &batch);

    if (batch.n_vectors != 1)
    {
        free_embedding_batch(&batch);
        elog(ERROR, "Expected 1 embedding, got %zu", batch.n_vectors);
    }

    Vector *v = make_vector_from_batch(&batch, 0);
    free_embedding_batch(&batch);

    PG_RETURN_POINTER(v);
}

PG_FUNCTION_INFO_V1(embed_texts);

Datum
embed_texts(PG_FUNCTION_ARGS)
{
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    ArrayType *input_array = PG_GETARG_ARRAYTYPE_P(2);
    Datum *text_elems;
    bool *nulls;
    int nitems;
    int embedder_id, model_id;
    EmbeddingBatch batch;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_TEXT,
                        &embedder_id, &model_id);

    deconstruct_array(input_array, TEXTOID, -1, false, 'i',
                      &text_elems, &nulls, &nitems);

    if (nitems == 0)
        PG_RETURN_NULL();

    StringSlice *c_inputs = palloc(sizeof(StringSlice) * nitems);
    for (int i = 0; i < nitems; i++)
    {
        text *t = DatumGetTextP(text_elems[i]);
        c_inputs[i] = text_to_string_slice(t);
    }

    InputData input_data = make_text_input(c_inputs, nitems);
    embed(embedder_id, model_id, &input_data, &batch);
    pfree(c_inputs);

    Datum *vectors = palloc(sizeof(Datum) * batch.n_vectors);
    for (size_t i = 0; i < batch.n_vectors; i++)
    {
        Vector *v = make_vector_from_batch(&batch, i);
        vectors[i] = PointerGetDatum(v);
    }

    ArrayType *result = construct_array(vectors, batch.n_vectors,
                             TypenameGetTypid("vector"), -1, false, 'd');

    free_embedding_batch(&batch);
    for (size_t i = 0; i < batch.n_vectors; i++)
        pfree(DatumGetPointer(vectors[i]));
    pfree(vectors);

    PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(embed_texts_with_ids);

Datum
embed_texts_with_ids(PG_FUNCTION_ARGS)
{
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    ArrayType *ids_array = PG_GETARG_ARRAYTYPE_P(2);
    ArrayType *texts_array = PG_GETARG_ARRAYTYPE_P(3);

    Datum *id_elems, *text_elems;
    bool *id_nulls, *text_nulls;
    int n_ids, n_texts;

    FuncCallContext *funcctx;
    typedef struct
    {
        int *ids;
        Vector **vectors;
        int nitems;
        int current;
    } user_fctx;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext oldcontext;
        int embedder_id, model_id;
        EmbeddingBatch batch;
        StringSlice *c_inputs;
        int *c_ids;

        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_TEXT,
                            &embedder_id, &model_id);

        deconstruct_array(ids_array, INT4OID, 4, true, 'i',
                          &id_elems, &id_nulls, &n_ids);
        deconstruct_array(texts_array, TEXTOID, -1, false, 'i',
                          &text_elems, &text_nulls, &n_texts);

        if (n_ids != n_texts)
            elog(ERROR, "Identifiers and texts arrays must have same length");

        c_inputs = palloc(sizeof(StringSlice) * n_texts);
        c_ids = palloc(sizeof(int) * n_ids);

        for (int i = 0; i < n_texts; i++)
        {
            if (id_nulls[i] || text_nulls[i])
                elog(ERROR, "NULL values not allowed");

            c_ids[i] = DatumGetInt32(id_elems[i]);
            text *t = DatumGetTextP(text_elems[i]);
            c_inputs[i] = text_to_string_slice(t);
        }

        InputData input_data = make_text_input(c_inputs, n_texts);
        embed(embedder_id, model_id, &input_data, &batch);
        pfree(c_inputs);

        Vector **vectors = palloc(sizeof(Vector *) * batch.n_vectors);
        for (size_t i = 0; i < batch.n_vectors; i++)
        {
            vectors[i] = make_vector_from_batch(&batch, i);
        }

        user_fctx *fctx = palloc(sizeof(user_fctx));
        fctx->ids = c_ids;
        fctx->vectors = vectors;
        fctx->nitems = batch.n_vectors;
        fctx->current = 0;

        funcctx->user_fctx = fctx;
        free_embedding_batch(&batch);

        TupleDesc tupdesc = CreateTemplateTupleDesc(2);
        TupleDescInitEntry(tupdesc, (AttrNumber)1, "id", INT4OID, -1, 0);
        TupleDescInitEntry(tupdesc, (AttrNumber)2, "embedding", TypenameGetTypid("vector"), -1, 0);
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);

        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    user_fctx *fctx = (user_fctx *)funcctx->user_fctx;

    if (fctx->current < fctx->nitems)
    {
        Datum values[2];
        bool nulls[2] = {false, false};
        HeapTuple tuple;

        values[0] = Int32GetDatum(fctx->ids[fctx->current]);
        values[1] = PointerGetDatum(fctx->vectors[fctx->current]);

        tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        fctx->current++;

        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    else
    {
        SRF_RETURN_DONE(funcctx);
    }
}

/* -------------------------------------------------------------------------
 * Image Embedding Functions
 * -------------------------------------------------------------------------
 */

PG_FUNCTION_INFO_V1(embed_image);

Datum
embed_image(PG_FUNCTION_ARGS)
{
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    bytea *input_bytea = PG_GETARG_BYTEA_P(2);
    int embedder_id, model_id;
    EmbeddingBatch batch;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_IMAGE,
                        &embedder_id, &model_id);

    ByteSlice c_input = bytea_to_byte_slice(input_bytea);
    InputData input_data = make_image_input(&c_input, 1);

    embed(embedder_id, model_id, &input_data, &batch);

    if (batch.n_vectors != 1)
    {
        free_embedding_batch(&batch);
        elog(ERROR, "Expected 1 embedding, got %zu", batch.n_vectors);
    }

    Vector *v = make_vector_from_batch(&batch, 0);
    free_embedding_batch(&batch);

    PG_RETURN_POINTER(v);
}

PG_FUNCTION_INFO_V1(embed_images);

Datum
embed_images(PG_FUNCTION_ARGS)
{
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    ArrayType *input_array = PG_GETARG_ARRAYTYPE_P(2);
    Datum *bytea_elems;
    bool *nulls;
    int nitems;
    int embedder_id, model_id;
    EmbeddingBatch batch;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_IMAGE,
                        &embedder_id, &model_id);

    deconstruct_array(input_array, BYTEAOID, -1, false, 'i',
                      &bytea_elems, &nulls, &nitems);

    if (nitems == 0)
        PG_RETURN_NULL();

    ByteSlice *c_inputs = palloc(sizeof(ByteSlice) * nitems);
    for (int i = 0; i < nitems; i++)
    {
        bytea *b = DatumGetByteaP(bytea_elems[i]);
        c_inputs[i] = bytea_to_byte_slice(b);
    }

    InputData input_data = make_image_input(c_inputs, nitems);
    embed(embedder_id, model_id, &input_data, &batch);
    pfree(c_inputs);

    Datum *vectors = palloc(sizeof(Datum) * batch.n_vectors);
    for (size_t i = 0; i < batch.n_vectors; i++)
    {
        Vector *v = make_vector_from_batch(&batch, i);
        vectors[i] = PointerGetDatum(v);
    }

    ArrayType *result = construct_array(vectors, batch.n_vectors,
                             TypenameGetTypid("vector"), -1, false, 'd');

    free_embedding_batch(&batch);
    for (size_t i = 0; i < batch.n_vectors; i++)
        pfree(DatumGetPointer(vectors[i]));
    pfree(vectors);

    PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(embed_images_with_ids);

Datum
embed_images_with_ids(PG_FUNCTION_ARGS)
{
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    ArrayType *ids_array = PG_GETARG_ARRAYTYPE_P(2);
    ArrayType *images_array = PG_GETARG_ARRAYTYPE_P(3);

    Datum *id_elems, *image_elems;
    bool *id_nulls, *image_nulls;
    int n_ids, n_images;

    FuncCallContext *funcctx;
    typedef struct
    {
        int *ids;
        Vector **vectors;
        int nitems;
        int current;
    } user_fctx;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext oldcontext;
        int embedder_id, model_id;
        EmbeddingBatch batch;
        ByteSlice *c_inputs;
        int *c_ids;

        funcctx = SRF_FIRSTCALL_INIT();
        oldcontext = MemoryContextSwitchTo(funcctx->multi_call_memory_ctx);

        validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_IMAGE,
                            &embedder_id, &model_id);

        deconstruct_array(ids_array, INT4OID, 4, true, 'i',
                          &id_elems, &id_nulls, &n_ids);
        deconstruct_array(images_array, BYTEAOID, -1, false, 'i',
                          &image_elems, &image_nulls, &n_images);

        if (n_ids != n_images)
            elog(ERROR, "Identifiers and images arrays must have same length");

        c_inputs = palloc(sizeof(ByteSlice) * n_images);
        c_ids = palloc(sizeof(int) * n_ids);

        for (int i = 0; i < n_images; i++)
        {
            if (id_nulls[i] || image_nulls[i])
                elog(ERROR, "NULL values not allowed");

            c_ids[i] = DatumGetInt32(id_elems[i]);
            bytea *b = DatumGetByteaP(image_elems[i]);
            c_inputs[i] = bytea_to_byte_slice(b);
        }

        InputData input_data = make_image_input(c_inputs, n_images);
        embed(embedder_id, model_id, &input_data, &batch);
        pfree(c_inputs);

        Vector **vectors = palloc(sizeof(Vector *) * batch.n_vectors);
        for (size_t i = 0; i < batch.n_vectors; i++)
        {
            vectors[i] = make_vector_from_batch(&batch, i);
        }

        user_fctx *fctx = palloc(sizeof(user_fctx));
        fctx->ids = c_ids;
        fctx->vectors = vectors;
        fctx->nitems = batch.n_vectors;
        fctx->current = 0;

        funcctx->user_fctx = fctx;
        free_embedding_batch(&batch);

        TupleDesc tupdesc = CreateTemplateTupleDesc(2);
        TupleDescInitEntry(tupdesc, (AttrNumber)1, "id", INT4OID, -1, 0);
        TupleDescInitEntry(tupdesc, (AttrNumber)2, "embedding", TypenameGetTypid("vector"), -1, 0);
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);

        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    user_fctx *fctx = (user_fctx *)funcctx->user_fctx;

    if (fctx->current < fctx->nitems)
    {
        Datum values[2];
        bool nulls[2] = {false, false};
        HeapTuple tuple;

        values[0] = Int32GetDatum(fctx->ids[fctx->current]);
        values[1] = PointerGetDatum(fctx->vectors[fctx->current]);

        tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        fctx->current++;

        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    else
    {
        SRF_RETURN_DONE(funcctx);
    }
}

/* -------------------------------------------------------------------------
 * Multimodal Embedding Functions
 * -------------------------------------------------------------------------
 */

PG_FUNCTION_INFO_V1(embed_multimodal);

Datum
embed_multimodal(PG_FUNCTION_ARGS)
{
    text *embedder_text = PG_GETARG_TEXT_P(0);
    text *model_text = PG_GETARG_TEXT_P(1);
    ArrayType *images_array = PG_ARGISNULL(2) ? NULL : PG_GETARG_ARRAYTYPE_P(2);
    ArrayType *text_array = PG_ARGISNULL(3) ? NULL : PG_GETARG_ARRAYTYPE_P(3);

    int embedder_id, model_id;
    EmbeddingBatch batch;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_MULTIMODAL,
                        &embedder_id, &model_id);

    ByteSlice *c_images = NULL;
    int n_images = 0;
    if (images_array != NULL)
    {
        Datum *bytea_elems;
        bool *nulls;
        deconstruct_array(images_array, BYTEAOID, -1, false, 'i', &bytea_elems, &nulls, &n_images);

        if (n_images > 0)
        {
            c_images = palloc(sizeof(ByteSlice) * n_images);
            for (int i = 0; i < n_images; i++)
            {
                bytea *b = DatumGetByteaP(bytea_elems[i]);
                c_images[i] = bytea_to_byte_slice(b);
            }
        }
    }

    StringSlice *c_texts = NULL;
    int n_texts = 0;
    if (text_array != NULL)
    {
        Datum *text_elems;
        bool *nulls;
        deconstruct_array(text_array, TEXTOID, -1, false, 'i', &text_elems, &nulls, &n_texts);

        if (n_texts > 0)
        {
            c_texts = palloc(sizeof(StringSlice) * n_texts);
            for (int i = 0; i < n_texts; i++)
            {
                text *t = DatumGetTextP(text_elems[i]);
                c_texts[i] = text_to_string_slice(t);
            }
        }
    }

    if (n_images == 0 && n_texts == 0)
        elog(ERROR, "At least one of images or texts must be provided");

    InputData input_data = make_multimodal_input(c_texts, n_texts, c_images, n_images);
    embed(embedder_id, model_id, &input_data, &batch);

    if (c_images)
        pfree(c_images);
    if (c_texts)
        pfree(c_texts);

    Datum *vectors = palloc(sizeof(Datum) * batch.n_vectors);
    for (size_t i = 0; i < batch.n_vectors; i++)
    {
        Vector *v = make_vector_from_batch(&batch, i);
        vectors[i] = PointerGetDatum(v);
    }

    ArrayType *result = construct_array(vectors, batch.n_vectors,
                             TypenameGetTypid("vector"), -1, false, 'd');

    free_embedding_batch(&batch);
    for (size_t i = 0; i < batch.n_vectors; i++)
        pfree(DatumGetPointer(vectors[i]));
    pfree(vectors);

    PG_RETURN_ARRAYTYPE_P(result);
}
