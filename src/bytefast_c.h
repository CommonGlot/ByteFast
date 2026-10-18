/* Copyright (c) 2026 CommonGlot contributors. MIT License.
 *
 * C API for prediction, the base for bindings in other languages.
 *
 *   char* error = NULL;
 *   bytefast_model* m = bytefast_load("model.safetensors", &error);
 *   bytefast_prediction top[3];
 *   int n = bytefast_predict(m, text, strlen(text), 3, 0.0f, top, &error);
 *   for (int i = 0; i < n; i++)
 *     printf("%s %f\n", bytefast_label(m, top[i].label), top[i].probability);
 *   bytefast_free(m);
 *
 * Functions that can fail take a `char** error`: on failure they set it to a
 * message that the caller frees with bytefast_free_error (it may be NULL if
 * the caller does not want the message). A loaded model is read-only, so
 * bytefast_predict can be called from several threads at once.
 */
#ifndef BYTEFAST_C_H
#define BYTEFAST_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bytefast_model bytefast_model;

typedef struct {
  int32_t label; /* index, see bytefast_label */
  float probability;
} bytefast_prediction;

/* Returns NULL on failure. */
bytefast_model* bytefast_load(const char* path, char** error);
void bytefast_free(bytefast_model* model);
void bytefast_free_error(char* error);

/* Writes the k most likely labels with probability >= threshold to out
 * (which has room for k entries), most likely first, and returns how many
 * were written, or -1 on failure. text is UTF-8 and need not end with a NUL
 * byte; a newline in it ends the line. */
int32_t bytefast_predict(
    const bytefast_model* model,
    const char* text,
    size_t length,
    int32_t k,
    float threshold,
    bytefast_prediction* out,
    char** error);

int32_t bytefast_num_labels(const bytefast_model* model);
/* The label name for an index, valid as long as the model. */
const char* bytefast_label(const bytefast_model* model, int32_t label);
int32_t bytefast_dimension(const bytefast_model* model);

#ifdef __cplusplus
}
#endif

#endif /* BYTEFAST_C_H */
