#include "internal.h"

PG_MODULE_MAGIC;

/* =========================================================================
 * Text embedding functions
 * ========================================================================= */

PG_FUNCTION_INFO_V1(embed_text);
Datum
embed_text(PG_FUNCTION_ARGS)
{
    PG_RETURN_POINTER(embed_one_text(PG_GETARG_TEXT_P(0),
                                     PG_GETARG_TEXT_P(1),
                                     PG_GETARG_TEXT_P(2)));
}

PG_FUNCTION_INFO_V1(embed_texts);
Datum
embed_texts(PG_FUNCTION_ARGS)
{
    ArrayType *result = embed_batch_text(PG_GETARG_TEXT_P(0),
                                         PG_GETARG_TEXT_P(1),
                                         PG_GETARG_ARRAYTYPE_P(2));
    if (result == NULL) PG_RETURN_NULL();
    PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(embed_texts_with_ids);
Datum
embed_texts_with_ids(PG_FUNCTION_ARGS)
{
    return embed_texts_with_ids_impl(fcinfo);
}

/* =========================================================================
 * Image embedding functions
 * ========================================================================= */

PG_FUNCTION_INFO_V1(embed_image);
Datum
embed_image(PG_FUNCTION_ARGS)
{
    PG_RETURN_POINTER(embed_one_image(PG_GETARG_TEXT_P(0),
                                      PG_GETARG_TEXT_P(1),
                                      PG_GETARG_BYTEA_P(2)));
}

PG_FUNCTION_INFO_V1(embed_images);
Datum
embed_images(PG_FUNCTION_ARGS)
{
    ArrayType *result = embed_batch_image(PG_GETARG_TEXT_P(0),
                                          PG_GETARG_TEXT_P(1),
                                          PG_GETARG_ARRAYTYPE_P(2));
    if (result == NULL) PG_RETURN_NULL();
    PG_RETURN_ARRAYTYPE_P(result);
}

PG_FUNCTION_INFO_V1(embed_images_with_ids);
Datum
embed_images_with_ids(PG_FUNCTION_ARGS)
{
    return embed_images_with_ids_impl(fcinfo);
}

PG_FUNCTION_INFO_V1(embed_image_directory);
Datum
embed_image_directory(PG_FUNCTION_ARGS)
{
    PG_RETURN_ARRAYTYPE_P(embed_one_image_directory(PG_GETARG_TEXT_P(0),
                                                     PG_GETARG_TEXT_P(1),
                                                     PG_GETARG_TEXT_P(2)));
}

PG_FUNCTION_INFO_V1(embed_image_directories);
Datum
embed_image_directories(PG_FUNCTION_ARGS)
{
    ArrayType *result = embed_batch_image_directory(PG_GETARG_TEXT_P(0),
                                                     PG_GETARG_TEXT_P(1),
                                                     PG_GETARG_ARRAYTYPE_P(2));
    if (result == NULL) PG_RETURN_NULL();
    PG_RETURN_ARRAYTYPE_P(result);
}

/* =========================================================================
 * Multimodal embedding function
 * ========================================================================= */

PG_FUNCTION_INFO_V1(embed_multimodal);
Datum
embed_multimodal(PG_FUNCTION_ARGS)
{
    return embed_multimodal_impl(fcinfo);
}

/* =========================================================================
 * Polymorphic dispatcher functions
 * ========================================================================= */

PG_FUNCTION_INFO_V1(embed_dispatch);
Datum
embed_dispatch(PG_FUNCTION_ARGS)
{
    return embed_dispatch_impl(fcinfo);
}

PG_FUNCTION_INFO_V1(embed_dispatch_array);
Datum
embed_dispatch_array(PG_FUNCTION_ARGS)
{
    return embed_dispatch_array_impl(fcinfo);
}
