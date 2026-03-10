#include "internal.h"

/* =========================================================================
 * Embedder & Model validation
 * ========================================================================= */

void
validate_embedder_and_model(text *embedder_text, text *model_text,
                            int input_type,
                            int *embedder_id, int *model_id)
{
    char *embedder_str = text_to_cstring(embedder_text);
    char *model_str    = text_to_cstring(model_text);

    *embedder_id = validate_embedder(embedder_str);
    if (*embedder_id < 0)
        elog(ERROR, "Invalid embedder: %s", embedder_str);

    *model_id = validate_embedding_model(*embedder_id, model_str, input_type);
    if (*model_id < 0)
        elog(ERROR, "Model not allowed: %s", model_str);
}

/* =========================================================================
 * PostgreSQL ↔ FFI type conversions
 * ========================================================================= */

StringSlice
text_to_string_slice(text *t)
{
    StringSlice s;
    s.ptr = VARDATA_ANY(t);
    s.len = VARSIZE_ANY_EXHDR(t);
    return s;
}

ByteSlice
bytea_to_byte_slice(bytea *b)
{
    ByteSlice s;
    s.ptr = (unsigned char *)VARDATA_ANY(b);
    s.len = VARSIZE_ANY_EXHDR(b);
    return s;
}

/* =========================================================================
 * InputData constructors
 * ========================================================================= */

InputData
make_text_input(const StringSlice *texts, size_t n_texts)
{
    InputData d = {0};
    d.input_type = INPUT_TYPE_TEXT;
    d.text_data  = texts;
    d.n_texts    = n_texts;
    return d;
}

InputData
make_image_input(const ByteSlice *images, size_t n_images)
{
    InputData d = {0};
    d.input_type  = INPUT_TYPE_IMAGE;
    d.binary_data = images;
    d.n_binaries  = n_images;
    return d;
}

InputData
make_multimodal_input(const StringSlice *texts, size_t n_texts,
                      const ByteSlice *images, size_t n_images)
{
    InputData d = {0};
    d.input_type  = INPUT_TYPE_MULTIMODAL;
    d.text_data   = texts;
    d.n_texts     = n_texts;
    d.binary_data = images;
    d.n_binaries  = n_images;
    return d;
}

InputData
make_image_directory_input(const StringSlice *paths, size_t n_paths)
{
    InputData d = {0};
    d.input_type = INPUT_TYPE_IMAGE_DIRECTORY;
    d.text_data  = paths;
    d.n_texts    = n_paths;
    return d;
}

/* =========================================================================
 * Core embed() wrapper
 * ========================================================================= */

void
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

/* =========================================================================
 * Vector / ArrayType construction utilities
 * ========================================================================= */

Vector *
make_vector_from_batch(const EmbeddingBatch *batch, size_t index)
{
    Vector *v = (Vector *)palloc(VECTOR_SIZE(batch->dim));
    SET_VARSIZE(v, VECTOR_SIZE(batch->dim));
    v->dim    = batch->dim;
    v->unused = 0;
    memcpy(v->x, batch->data + index * batch->dim, sizeof(float) * batch->dim);
    return v;
}

void
populate_vector_datums(const EmbeddingBatch *batch,
                       void **out_block, Datum **out_vectors)
{
    size_t  vec_size = VECTOR_SIZE(batch->dim);
    char   *block    = palloc(batch->n_vectors * vec_size);
    Datum  *vectors  = palloc(sizeof(Datum) * batch->n_vectors);

    for (size_t i = 0; i < batch->n_vectors; i++)
    {
        Vector *v = (Vector *)(block + i * vec_size);
        SET_VARSIZE(v, vec_size);
        v->dim    = batch->dim;
        v->unused = 0;
        memcpy(v->x, batch->data + i * batch->dim, sizeof(float) * batch->dim);
        vectors[i] = PointerGetDatum(v);
    }

    *out_block   = block;
    *out_vectors = vectors;
}

ArrayType *
construct_vector_array(const EmbeddingBatch *batch)
{
    void      *block;
    Datum     *vectors;

    populate_vector_datums(batch, &block, &vectors);

    ArrayType *result = construct_array(vectors, batch->n_vectors,
                                        TypenameGetTypid("vector"),
                                        -1, false, 'd');
    pfree(block);
    pfree(vectors);
    return result;
}

/* =========================================================================
 * SRF helper
 * ========================================================================= */

Datum
srf_return_next_embedding(FuncCallContext *funcctx, EmbeddingSRFContext *fctx,
                          FunctionCallInfo fcinfo)
{
    if (fctx->current < fctx->nitems)
    {
        Datum     values[2];
        bool      nulls[2] = {false, false};
        HeapTuple tuple;

        values[0] = Int32GetDatum(fctx->ids[fctx->current]);
        values[1] = fctx->vectors[fctx->current];

        tuple = heap_form_tuple(funcctx->tuple_desc, values, nulls);
        fctx->current++;

        SRF_RETURN_NEXT(funcctx, HeapTupleGetDatum(tuple));
    }
    else
    {
        SRF_RETURN_DONE(funcctx);
    }
}

/* =========================================================================
 * Enum label resolution
 * ========================================================================= */

char *
resolve_input_type_label(Oid enum_oid)
{
    HeapTuple    enum_tup = SearchSysCache1(ENUMOID, ObjectIdGetDatum(enum_oid));
    if (!HeapTupleIsValid(enum_tup))
        elog(ERROR, "embed(): invalid input_type OID %u", enum_oid);

    Form_pg_enum en    = (Form_pg_enum) GETSTRUCT(enum_tup);
    char        *label = pstrdup(NameStr(en->enumlabel));
    ReleaseSysCache(enum_tup);
    return label;
}

/* =========================================================================
 * Single-item helpers
 * ========================================================================= */

Vector *
embed_one_text(text *embedder_text, text *model_text, text *input_text)
{
    int            embedder_id, model_id;
    EmbeddingBatch batch;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_TEXT,
                                &embedder_id, &model_id);

    StringSlice c_input    = text_to_string_slice(input_text);
    InputData   input_data = make_text_input(&c_input, 1);

    embed(embedder_id, model_id, &input_data, &batch);

    if (batch.n_vectors != 1)
    {
        free_embedding_batch(&batch);
        elog(ERROR, "Expected 1 embedding, got %zu", batch.n_vectors);
    }

    Vector *v = make_vector_from_batch(&batch, 0);
    free_embedding_batch(&batch);
    return v;
}

Vector *
embed_one_image(text *embedder_text, text *model_text, bytea *input_bytea)
{
    int            embedder_id, model_id;
    EmbeddingBatch batch;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_IMAGE,
                                &embedder_id, &model_id);

    ByteSlice c_input    = bytea_to_byte_slice(input_bytea);
    InputData input_data = make_image_input(&c_input, 1);

    embed(embedder_id, model_id, &input_data, &batch);

    if (batch.n_vectors != 1)
    {
        free_embedding_batch(&batch);
        elog(ERROR, "Expected 1 embedding, got %zu", batch.n_vectors);
    }

    Vector *v = make_vector_from_batch(&batch, 0);
    free_embedding_batch(&batch);
    return v;
}

ArrayType *
embed_one_image_directory(text *embedder_text, text *model_text, text *path_text)
{
    int            embedder_id, model_id;
    EmbeddingBatch batch;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_IMAGE_DIRECTORY,
                                &embedder_id, &model_id);

    StringSlice c_input    = text_to_string_slice(path_text);
    InputData   input_data = make_image_directory_input(&c_input, 1);

    embed(embedder_id, model_id, &input_data, &batch);

    ArrayType *result = construct_vector_array(&batch);
    free_embedding_batch(&batch);
    return result;
}

/* =========================================================================
 * Batch helpers
 * ========================================================================= */

ArrayType *
embed_batch_text(text *embedder_text, text *model_text, ArrayType *input_array)
{
    int            embedder_id, model_id;
    EmbeddingBatch batch;
    Datum         *text_elems;
    bool          *nulls;
    int            nitems;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_TEXT,
                                &embedder_id, &model_id);

    deconstruct_array(input_array, TEXTOID, -1, false, 'i',
                      &text_elems, &nulls, &nitems);

    if (nitems == 0)
        return NULL;

    StringSlice *c_inputs = palloc(sizeof(StringSlice) * nitems);
    for (int i = 0; i < nitems; i++)
        c_inputs[i] = text_to_string_slice(DatumGetTextP(text_elems[i]));

    InputData input_data = make_text_input(c_inputs, nitems);
    embed(embedder_id, model_id, &input_data, &batch);
    pfree(c_inputs);

    ArrayType *result = construct_vector_array(&batch);
    free_embedding_batch(&batch);
    return result;
}

ArrayType *
embed_batch_image(text *embedder_text, text *model_text, ArrayType *input_array)
{
    int            embedder_id, model_id;
    EmbeddingBatch batch;
    Datum         *bytea_elems;
    bool          *nulls;
    int            nitems;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_IMAGE,
                                &embedder_id, &model_id);

    deconstruct_array(input_array, BYTEAOID, -1, false, 'i',
                      &bytea_elems, &nulls, &nitems);

    if (nitems == 0)
        return NULL;

    ByteSlice *c_inputs = palloc(sizeof(ByteSlice) * nitems);
    for (int i = 0; i < nitems; i++)
        c_inputs[i] = bytea_to_byte_slice(DatumGetByteaP(bytea_elems[i]));

    InputData input_data = make_image_input(c_inputs, nitems);
    embed(embedder_id, model_id, &input_data, &batch);
    pfree(c_inputs);

    ArrayType *result = construct_vector_array(&batch);
    free_embedding_batch(&batch);
    return result;
}

ArrayType *
embed_batch_image_directory(text *embedder_text, text *model_text,
                            ArrayType *input_array)
{
    int            embedder_id, model_id;
    EmbeddingBatch batch;
    Datum         *path_elems;
    bool          *nulls;
    int            nitems;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_IMAGE_DIRECTORY,
                                &embedder_id, &model_id);

    deconstruct_array(input_array, TEXTOID, -1, false, 'i',
                      &path_elems, &nulls, &nitems);

    if (nitems == 0)
        return NULL;

    StringSlice *c_inputs = palloc(sizeof(StringSlice) * nitems);
    for (int i = 0; i < nitems; i++)
        c_inputs[i] = text_to_string_slice(DatumGetTextP(path_elems[i]));

    InputData input_data = make_image_directory_input(c_inputs, nitems);
    embed(embedder_id, model_id, &input_data, &batch);
    pfree(c_inputs);

    ArrayType *result = construct_vector_array(&batch);
    free_embedding_batch(&batch);
    return result;
}

/* =========================================================================
 * SRF implementations
 * ========================================================================= */

Datum
embed_texts_with_ids_impl(FunctionCallInfo fcinfo)
{
    text      *embedder_text = PG_GETARG_TEXT_P(0);
    text      *model_text    = PG_GETARG_TEXT_P(1);
    ArrayType *ids_array     = PG_GETARG_ARRAYTYPE_P(2);
    ArrayType *texts_array   = PG_GETARG_ARRAYTYPE_P(3);

    Datum *id_elems, *text_elems;
    bool  *id_nulls, *text_nulls;
    int    n_ids, n_texts;

    FuncCallContext *funcctx;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext   oldcontext;
        int             embedder_id, model_id;
        EmbeddingBatch  batch;
        StringSlice    *c_inputs;
        int            *c_ids;

        funcctx    = SRF_FIRSTCALL_INIT();
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
        c_ids    = palloc(sizeof(int) * n_ids);

        for (int i = 0; i < n_texts; i++)
        {
            if (id_nulls[i] || text_nulls[i])
                elog(ERROR, "NULL values not allowed");
            c_ids[i]    = DatumGetInt32(id_elems[i]);
            c_inputs[i] = text_to_string_slice(DatumGetTextP(text_elems[i]));
        }

        InputData input_data = make_text_input(c_inputs, n_texts);
        embed(embedder_id, model_id, &input_data, &batch);
        pfree(c_inputs);

        Datum *vectors;
        void  *block;
        populate_vector_datums(&batch, &block, &vectors);

        EmbeddingSRFContext *fctx = palloc(sizeof(EmbeddingSRFContext));
        fctx->ids     = c_ids;
        fctx->vectors = vectors;
        fctx->block   = block;
        fctx->nitems  = batch.n_vectors;
        fctx->current = 0;

        funcctx->user_fctx = fctx;
        free_embedding_batch(&batch);

        TupleDesc tupdesc = CreateTemplateTupleDesc(2);
        TupleDescInitEntry(tupdesc, (AttrNumber)1, "id",        INT4OID,                   -1, 0);
        TupleDescInitEntry(tupdesc, (AttrNumber)2, "embedding", TypenameGetTypid("vector"), -1, 0);
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);

        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    return srf_return_next_embedding(funcctx,
                                     (EmbeddingSRFContext *)funcctx->user_fctx,
                                     fcinfo);
}

Datum
embed_images_with_ids_impl(FunctionCallInfo fcinfo)
{
    text      *embedder_text = PG_GETARG_TEXT_P(0);
    text      *model_text    = PG_GETARG_TEXT_P(1);
    ArrayType *ids_array     = PG_GETARG_ARRAYTYPE_P(2);
    ArrayType *images_array  = PG_GETARG_ARRAYTYPE_P(3);

    Datum *id_elems, *image_elems;
    bool  *id_nulls, *image_nulls;
    int    n_ids, n_images;

    FuncCallContext *funcctx;

    if (SRF_IS_FIRSTCALL())
    {
        MemoryContext  oldcontext;
        int            embedder_id, model_id;
        EmbeddingBatch batch;
        ByteSlice     *c_inputs;
        int           *c_ids;

        funcctx    = SRF_FIRSTCALL_INIT();
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
        c_ids    = palloc(sizeof(int) * n_ids);

        for (int i = 0; i < n_images; i++)
        {
            if (id_nulls[i] || image_nulls[i])
                elog(ERROR, "NULL values not allowed");
            c_ids[i]    = DatumGetInt32(id_elems[i]);
            c_inputs[i] = bytea_to_byte_slice(DatumGetByteaP(image_elems[i]));
        }

        InputData input_data = make_image_input(c_inputs, n_images);
        embed(embedder_id, model_id, &input_data, &batch);
        pfree(c_inputs);

        Datum *vectors;
        void  *block;
        populate_vector_datums(&batch, &block, &vectors);

        EmbeddingSRFContext *fctx = palloc(sizeof(EmbeddingSRFContext));
        fctx->ids     = c_ids;
        fctx->vectors = vectors;
        fctx->block   = block;
        fctx->nitems  = batch.n_vectors;
        fctx->current = 0;

        funcctx->user_fctx = fctx;
        free_embedding_batch(&batch);

        TupleDesc tupdesc = CreateTemplateTupleDesc(2);
        TupleDescInitEntry(tupdesc, (AttrNumber)1, "id",        INT4OID,                   -1, 0);
        TupleDescInitEntry(tupdesc, (AttrNumber)2, "embedding", TypenameGetTypid("vector"), -1, 0);
        funcctx->tuple_desc = BlessTupleDesc(tupdesc);

        MemoryContextSwitchTo(oldcontext);
    }

    funcctx = SRF_PERCALL_SETUP();
    return srf_return_next_embedding(funcctx,
                                     (EmbeddingSRFContext *)funcctx->user_fctx,
                                     fcinfo);
}

/* =========================================================================
 * Multimodal implementation
 * ========================================================================= */

Datum
embed_multimodal_impl(FunctionCallInfo fcinfo)
{
    text      *embedder_text = PG_GETARG_TEXT_P(0);
    text      *model_text    = PG_GETARG_TEXT_P(1);
    ArrayType *images_array  = PG_ARGISNULL(2) ? NULL : PG_GETARG_ARRAYTYPE_P(2);
    ArrayType *text_array    = PG_ARGISNULL(3) ? NULL : PG_GETARG_ARRAYTYPE_P(3);

    int            embedder_id, model_id;
    EmbeddingBatch batch;

    validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_MULTIMODAL,
                                &embedder_id, &model_id);

    ByteSlice *c_images = NULL;
    int        n_images = 0;
    if (images_array != NULL)
    {
        Datum *bytea_elems;
        bool  *nulls;
        deconstruct_array(images_array, BYTEAOID, -1, false, 'i',
                          &bytea_elems, &nulls, &n_images);
        if (n_images > 0)
        {
            c_images = palloc(sizeof(ByteSlice) * n_images);
            for (int i = 0; i < n_images; i++)
                c_images[i] = bytea_to_byte_slice(DatumGetByteaP(bytea_elems[i]));
        }
    }

    StringSlice *c_texts = NULL;
    int          n_texts = 0;
    if (text_array != NULL)
    {
        Datum *text_elems;
        bool  *nulls;
        deconstruct_array(text_array, TEXTOID, -1, false, 'i',
                          &text_elems, &nulls, &n_texts);
        if (n_texts > 0)
        {
            c_texts = palloc(sizeof(StringSlice) * n_texts);
            for (int i = 0; i < n_texts; i++)
                c_texts[i] = text_to_string_slice(DatumGetTextP(text_elems[i]));
        }
    }

    if (n_images == 0 && n_texts == 0)
        elog(ERROR, "At least one of images or texts must be provided");

    InputData input_data = make_multimodal_input(c_texts, n_texts, c_images, n_images);
    embed(embedder_id, model_id, &input_data, &batch);

    if (c_images) pfree(c_images);
    if (c_texts)  pfree(c_texts);

    ArrayType *result = construct_vector_array(&batch);
    free_embedding_batch(&batch);

    PG_RETURN_ARRAYTYPE_P(result);
}

/* =========================================================================
 * Dispatcher implementations
 * ========================================================================= */

Datum
embed_dispatch_impl(FunctionCallInfo fcinfo)
{
    text  *embedder_text = PG_GETARG_TEXT_P(0);
    text  *model_text    = PG_GETARG_TEXT_P(1);
    Datum  input_datum   = PG_GETARG_DATUM(2);
    Oid    enum_oid      = PG_GETARG_OID(3);

    Oid   input_typeid = get_fn_expr_argtype(fcinfo->flinfo, 2);
    char *label        = resolve_input_type_label(enum_oid);

    if (strcmp(label, "text") == 0)
    {
        if (input_typeid != TEXTOID)
            ereport(ERROR,
                    (errcode(ERRCODE_DATATYPE_MISMATCH),
                     errmsg("embed(): input_type 'text' requires a text argument, "
                            "got type OID %u", input_typeid)));

        PG_RETURN_POINTER(embed_one_text(embedder_text, model_text,
                                        DatumGetTextP(input_datum)));
    }
    else if (strcmp(label, "image") == 0)
    {
        if (input_typeid != BYTEAOID)
            ereport(ERROR,
                    (errcode(ERRCODE_DATATYPE_MISMATCH),
                     errmsg("embed(): input_type 'image' requires a bytea argument, "
                            "got type OID %u", input_typeid)));

        PG_RETURN_POINTER(embed_one_image(embedder_text, model_text,
                                         DatumGetByteaP(input_datum)));
    }
    else if (strcmp(label, "image_directory") == 0)
    {
        int            embedder_id, model_id;
        EmbeddingBatch batch;

        if (input_typeid != TEXTOID)
            ereport(ERROR,
                    (errcode(ERRCODE_DATATYPE_MISMATCH),
                     errmsg("embed(): input_type 'image_directory' requires a text "
                            "(path) argument, got type OID %u", input_typeid)));

        validate_embedder_and_model(embedder_text, model_text, INPUT_TYPE_IMAGE_DIRECTORY,
                                    &embedder_id, &model_id);

        StringSlice c_input    = text_to_string_slice(DatumGetTextP(input_datum));
        InputData   input_data = make_image_directory_input(&c_input, 1);

        embed(embedder_id, model_id, &input_data, &batch);

        if (batch.n_vectors < 1)
        {
            free_embedding_batch(&batch);
            elog(ERROR, "embed(): directory '%s' produced no embeddings",
                 text_to_cstring(DatumGetTextP(input_datum)));
        }

        Vector *v = make_vector_from_batch(&batch, 0);
        free_embedding_batch(&batch);
        PG_RETURN_POINTER(v);
    }

    elog(ERROR, "embed(): unhandled input_type value: '%s'", label);
    PG_RETURN_NULL(); /* unreachable */
}

Datum
embed_dispatch_array_impl(FunctionCallInfo fcinfo)
{
    text      *embedder_text = PG_GETARG_TEXT_P(0);
    text      *model_text    = PG_GETARG_TEXT_P(1);
    ArrayType *input_array   = PG_GETARG_ARRAYTYPE_P(2);
    Oid        enum_oid      = PG_GETARG_OID(3);

    Oid array_typeid = get_fn_expr_argtype(fcinfo->flinfo, 2);
    Oid elem_typeid  = get_element_type(array_typeid);
    if (!OidIsValid(elem_typeid))
        elog(ERROR, "embed(): could not determine element type of input array");

    char *label = resolve_input_type_label(enum_oid);

    ArrayType *result = NULL;

    if (strcmp(label, "text") == 0)
    {
        if (elem_typeid != TEXTOID)
            ereport(ERROR,
                    (errcode(ERRCODE_DATATYPE_MISMATCH),
                     errmsg("embed(): input_type 'text' requires a text[] argument, "
                            "got element type OID %u", elem_typeid)));

        result = embed_batch_text(embedder_text, model_text, input_array);
    }
    else if (strcmp(label, "image") == 0)
    {
        if (elem_typeid != BYTEAOID)
            ereport(ERROR,
                    (errcode(ERRCODE_DATATYPE_MISMATCH),
                     errmsg("embed(): input_type 'image' requires a bytea[] argument, "
                            "got element type OID %u", elem_typeid)));

        result = embed_batch_image(embedder_text, model_text, input_array);
    }
    else if (strcmp(label, "image_directory") == 0)
    {
        if (elem_typeid != TEXTOID)
            ereport(ERROR,
                    (errcode(ERRCODE_DATATYPE_MISMATCH),
                     errmsg("embed(): input_type 'image_directory' requires a text[] "
                            "(paths) argument, got element type OID %u", elem_typeid)));

        result = embed_batch_image_directory(embedder_text, model_text, input_array);
    }
    else
    {
        elog(ERROR, "embed(): unhandled input_type value: '%s'", label);
    }

    if (result == NULL)
        PG_RETURN_NULL();

    PG_RETURN_ARRAYTYPE_P(result);
}
