/**
 * BCE v0.4 - compressor for stationary data
 * Copyright (C) 2016  Christoph Diegelmann
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
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

#include <cstdint>
#include <chrono>
#include <algorithm>

using Compressor = bce::Compressor<
  bce::transform::bwt<>,          // transform
  bce::inverse::unbwt_bytewise<>, // inverse transform
  bce::coder::adaptive<31>        // coder
>;

int compress(std::string archive_name, std::string file_name) {
  auto start = std::chrono::high_resolution_clock::now();

  std::ifstream file(file_name, std::ios::binary | std::ios::ate);

  std::size_t size = file.tellg();
  if (size == static_cast<decltype(size)>(-1)) return -1;
  file.seekg(0, std::ios::beg);

  std::ofstream archive(archive_name, std::ios::binary | std::ios::trunc);
  if (!archive.is_open()) return -2;

  Compressor bce;

  auto remain = size;
  decltype(size) out_size = 0;
  std::vector<uint8_t> map;
  constexpr const std::size_t block_size_max = UINT64_C(1) << 30;
  do {
    auto block_size = std::min(remain, block_size_max);
    if (map.size() < block_size)  map.resize(block_size);

    if (!file.read(reinterpret_cast<char*>(map.data()), block_size))
      return -2; // read error

    auto t_last = map.end();
    if (bce.compress<
            bce::rank::fast<>,
            bce::queue::packed<>
          >(map.begin(), map.begin() + block_size, map.begin(), &t_last) < -2) {
      return -3;
    }

    archive.write(reinterpret_cast<const char*>(map.data()), t_last - map.begin());
    out_size += t_last - map.begin();
    remain -= block_size;
  } while(remain);

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
  if (size == static_cast<decltype(size)>(-1)) return -1;
  archive.seekg(0, std::ios::beg);

  std::ofstream file(file_name, std::ios::binary | std::ios::trunc);
  if (!file.is_open()) return -2;

  Compressor bce;

  std::size_t out_size = 0;
  std::size_t remain = size;

  constexpr const std::size_t header_max = 1024;
  std::vector<uint8_t> map(header_max); // max size of the header
  while(remain > 0) {
    // read enough bytes to decode the header
    auto header_read = std::min(header_max, remain);
    if (!archive.read(reinterpret_cast<char*>(map.data()), header_read)) return -2;

    // read the info [decompressed size, compressed size]
    auto info = bce.block_info(map.begin(), map.end());

    auto max = std::max(info.first, info.second);
    if (map.size() < max)
      map.resize(max);

    if (info.second > header_read)
      if (!archive.read(reinterpret_cast<char*>(map.data() + header_read), info.second - header_read)) return -2;  // reading the remaining bytes

    auto begin = map.begin();
    auto end   = map.begin() + info.first;

    if (bce.decompress<
        bce::rank::fast<>,
        bce::queue::packed<>
      >(&begin, begin + info.second, map.begin(), &end)) {
      return -3;
    }

    remain -= info.second;
    out_size += info.first;

    if (info.second < header_max) archive.seekg(size - remain, std::ios::beg);

    file.write(reinterpret_cast<const char*>(map.data()), end - map.begin());
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
