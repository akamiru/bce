// C++ headers
#include <chrono>
#include <cstdlib>
#include <fstream>

// IACA headers
#include </home/christoph/iaca-lin64/include/iacaMarks.h>

// Architecture headers
#include "immintrin.h"

void print256_num_(__m256i var, const char* name) {
 int32_t *v64val = (int32_t*) &var;
 printf("%s: %.8x %.8x %.8x %.8x %.8x %.8x %.8x %.8x\n", name, v64val[7], v64val[6], v64val[5], v64val[4], v64val[3], v64val[2], v64val[1], v64val[0]);
}
#define print256_num(x) print256_num_((x).get_native(), #x)

// Project headers
#include "data_model.h"
#include "dictionary/dict_avx2.h"

int main(int argc, char **argv) {
  // open file
  std::ifstream archive("enwik9.bwt", std::ios::binary | std::ios::ate);

  // get size
  std::size_t size = archive.tellg();
  archive.seekg(0, std::ios::beg);
  if (size == -1) {
    printf("File not found.\n");
    return -1;
  }

  // read data
  auto data = reinterpret_cast<uint8_t*>(_mm_malloc(size + 64, 32));
  if (!archive.read(reinterpret_cast<char*>(data), size)) {
    printf("Could not read file.\n");
    return -2;
  }
  archive.close();

  auto start = std::chrono::high_resolution_clock::now();

  // allocate required working space
  auto work = reinterpret_cast<uint8_t*>(_mm_malloc(4 * size + 64, 32));

  // model
  bce::data_model model;
  auto r = model.compress<bce::int32x8_avx2>(
    data,
    size,
    work,
    4 * size + 64
  );

  // free the memory !
  _mm_free(work);
  _mm_free(data);

  if (r != bce::data_model::error::SUCCESS)
    printf("Error: %lu\n", r);

  auto end   = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  printf("Time to compress: %f s (%f MB/s, %f MB)\n", duration.count(), static_cast<double>(size) / duration.count() / 1.e6, size / 1e6);

  return 0;
}