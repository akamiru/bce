/**
 * BCE v0.4 - compressor for stationary data
 * Copyright (C) 2016  Christoph Diegelmann
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/lgpl>.
 */

#include "bce.h"

#include <chrono>

int compress(std::string archive_name, std::string file_name) {
  auto start = std::chrono::high_resolution_clock::now();

  std::ifstream file(file_name, std::ios::binary | std::ios::ate);

  std::size_t size = file.tellg();
  if (size == -1lu) return -1;
  file.seekg(0, std::ios::beg);

  std::ofstream archive(archive_name, std::ios::binary | std::ios::trunc);
  if (!archive.is_open()) return -2;

  bce::Compressor<
    bce::transform::bwt<>, 
    bce::inverse::unbwt_bytewise<>, 
    bce::coder::adaptive<31>
  > bce;

  auto remain = size;
  do {
    auto block_size = std::min(remain, static_cast<std::size_t>(1llu << 30));

    std::vector<uint8_t> map(block_size);
    if (!file.read(reinterpret_cast<char*>(map.data()), block_size)) return -2;

    bce.compress<
        bce::rank::fast<>, 
        bce::queue::packed<>
      >(map.begin(), map.end());

    remain -= block_size;
  } while(remain);

  auto data = bce.finalize();
  auto out_size = data.size() * sizeof(decltype(data)::value_type);
  archive.write(reinterpret_cast<const char*>(data.data()), out_size);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  printf("Compressed from %" SCNuMAX" B -> %zu B in %.1f s\n",
        size, out_size, duration.count());

  return 0;
}

int decompress(std::string file_name, std::string archive_name) {
  auto start = std::chrono::high_resolution_clock::now();

  std::ifstream archive(archive_name, std::ios::binary | std::ios::ate);

  std::size_t size = archive.tellg();
  if (size == -1lu) return -1;
  archive.seekg(0, std::ios::beg);

  std::vector<uint8_t> map(size);
  if (!archive.read(reinterpret_cast<char*>(map.data()), size)) return -2;

  std::ofstream file(file_name, std::ios::binary | std::ios::trunc);
  if (!file.is_open()) return -2;

  bce::Compressor<
    bce::transform::bwt<>, 
    bce::inverse::unbwt_bytewise<>,
    bce::coder::adaptive<31>
  > bce(map.begin(), map.end());

  std::size_t out_size = 0;

  while(1) {
    auto data = bce.decompress<
        bce::rank::fast<>, 
        bce::queue::packed<>
      >();

    auto size = data.size() * sizeof(decltype(data)::value_type);
    if (!size) break;

    out_size += size;
    file.write(reinterpret_cast<const char*>(data.data()), size);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  printf("Decompressed from %" SCNuMAX" B -> %zu B in %.1f s\n",
        size, out_size, duration.count());

  return 0;
}

int main(int argc, char** argv) {
  printf("BCE v0.4.1 Release\n");
  printf("Copyright (C) 2016  Christoph Diegelmann\n");
  printf("This is free software under GNU Lesser General Public License. See <http://www.gnu.org/licenses/lgpl>\n\n");

  if (argc == 4 && argv[1][0] == '-') {
    if (argv[1][1] == 'c')
      return compress(argv[2], argv[3]);
    else if (argv[1][1] == 'd')
      return decompress(argv[2], argv[3]);
  } else {
    printf("Usage:\n");
    printf("  bce -c archive.bce file\n");
    printf("   Compresses \"file\" to archive \"archive.bce\"\n");
    printf("\n");
    printf("  bce -d file archive.bce\n");
    printf("   Decompresses archive \"archive.bce\" to \"file\"\n");
  }
}
