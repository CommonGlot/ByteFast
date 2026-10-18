/* Checks the C API on a model: c_api_test MODEL "text" EXPECTED_LABEL */
#include <stdio.h>
#include <string.h>

#include "bytefast_c.h"

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: c_api_test MODEL TEXT EXPECTED_LABEL\n");
    return 2;
  }
  char* error = NULL;
  if (bytefast_load("/nonexistent.safetensors", &error) || !error) {
    fprintf(stderr, "loading a missing file did not fail\n");
    return 1;
  }
  bytefast_free_error(error);
  error = NULL;

  bytefast_model* model = bytefast_load(argv[1], &error);
  if (!model) {
    fprintf(stderr, "load failed: %s\n", error);
    bytefast_free_error(error);
    return 1;
  }
  bytefast_prediction top[3];
  int32_t n = bytefast_predict(model, argv[2], strlen(argv[2]), 3, 0.0f, top, &error);
  if (n < 1) {
    fprintf(stderr, "predict failed: %s\n", error ? error : "no prediction");
    return 1;
  }
  for (int32_t i = 0; i < n; i++) {
    printf("%s %.5f\n", bytefast_label(model, top[i].label), top[i].probability);
  }
  int ok = strcmp(bytefast_label(model, top[0].label), argv[3]) == 0 &&
      bytefast_num_labels(model) >= n && bytefast_dimension(model) > 0 &&
      bytefast_label(model, -1) == NULL;
  bytefast_free(model);
  if (!ok) {
    fprintf(stderr, "unexpected result\n");
    return 1;
  }
  printf("C_API_PASS\n");
  return 0;
}
