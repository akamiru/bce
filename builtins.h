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

#ifndef BCE_BUILTINS
#define BCE_BUILTINS

#include <array>

namespace bce {
namespace builtin {

#if defined(_MSC_VER)
#include <intrin.h>
  inline constexpr const bool unlikely(const bool value) {
    return value;
  }

  inline constexpr uint64_t clz(uint64_t val) {
    unsigned long r = 0;
#if defined(_M_AMD64) || defined(_M_X64) || defined(_M_ARM)
    if (_BitScanReverse64(&r, val))
      return 63 - r;
#else
    if (_BitScanReverse(&r, static_cast<uint32_t>(val >> 32)))
      return 31 - r;
    if (_BitScanReverse(&r, static_cast<uint32_t>(val)))
      return 63 - r;
#endif
    return 64;
  }

  inline constexpr uint64_t ctz(uint64_t val) {
    unsigned long r = 0;
#if defined(_M_AMD64) || defined(_M_X64) || defined(_M_ARM)
    if (_BitScanForward64(&r, val))
      return r;
#else
    if (_BitScanForward(&r, static_cast<uint32_t>(val)))
      return r;
    if (_BitScanForward(&r, static_cast<uint32_t>(val >> 32)))
      return 32 + r;
#endif
    return 64;
  }

  inline constexpr uint64_t cnt(uint64_t val) {
    // https://en.wikipedia.org/wiki/Hamming_weight
    const uint64_t m1 = 0x5555555555555555;
    const uint64_t m2 = 0x3333333333333333;
    const uint64_t m4 = 0x0f0f0f0f0f0f0f0f;
    const uint64_t h01 = 0x0101010101010101;

    val -= (val >> 1) & m1;
    val = (val & m2) + ((val >> 2) & m2);
    val = (val + (val >> 4)) & m4;
    return (val * h01) >> 56;
  }
#else
  inline constexpr const bool unlikely(const bool value) {
    return __builtin_expect(value, 0);
  }

  inline constexpr uint64_t clz(uint64_t val) {
    return sizeof(unsigned long) == 8 ? __builtin_clzl(val) :  __builtin_clzll(val);
  }

  inline constexpr uint64_t ctz(uint64_t val) {
    return sizeof(unsigned long) == 8 ? __builtin_ctzl(val) : __builtin_ctzll(val);
  }

  inline constexpr uint64_t cnt(uint64_t val) {
    return sizeof(unsigned long) == 8 ? __builtin_popcountl(val) : __builtin_popcountll(val);
  }
#endif
  inline constexpr uint64_t clo(uint64_t val) {
    return clz(~val);
  }

  inline constexpr uint64_t cto(uint64_t val) {
    return ctz(~val);
  }

  template<class ForwardIt>
  ForwardIt to_lmsr(ForwardIt first, ForwardIt last) {
    auto size = last - first;
    auto mod = [size](auto i) {
      while (bce::builtin::unlikely(i >= size)) i -= size;
      return i;
    };

    // Credits: MaskRay [github]
    uint32_t i = 0, j = 1, k;
    while (j < size) {
      for (k = 0; first[mod(i+k)] == first[mod(j+k)] && k < size - 1; k++);

      if (first[mod(i+k)] <= first[mod(j+k)]) {
        j += k+1;
      } else {
        i += k+1;
        if (i < j)
          i = j++;
        else
          j = i+1;
      }
    }

    std::rotate(first, first + i + 1, last);

    return first + i;
  }

  template<class rank, class ForwardIt>
  std::array<rank, 8> to_dictionary(ForwardIt first, ForwardIt last) {
    std::array<rank, 8> ranks;
    for (int i = 0; i < 8; ++i)
      ranks[i] = rank(last - first);

    std::array<uint32_t, 256> C;
    C.fill(0);

    for (auto i = first; i < last; ++i)
      C[*i | 0x80]++;

    for (uint32_t i = 0x80; i < 0x100; ++i)
      for (uint32_t j = 1; j < 8; ++j)
        C[(((i << j) & 0xFF) | 0x80) >> j] += C[i];

    for (int i = 0; i < 8; ++i) {
      uint32_t sum = 0;
      for (int j = 1 << i; j < 1 << (i + 1); ++j) {
        std::swap(C[j], sum);
        sum += C[j];
      }
    }

#ifdef _OPENMP
    #pragma omp parallel for firstprivate(C)
    for (int j = 0; j < 8; ++j) {
      for (auto i = first; i < last; ++i) {
        auto chr = *i;
        auto c = (chr & ((1 << j) - 1)) | (1 << j);
        ranks[j].set_bit(C[c]++, (chr >> j) & 1);
      }
      ranks[j].build();
    }
#else
    for (auto i = first; i < last; ++i) {
      auto chr = *i;
      for (int j = 0; j < 8; ++j) {
        auto c = (chr & ((1 << j) - 1)) | (1 << j);
        ranks[j].set_bit(C[c]++, (chr >> j) & 1);
      }
    }
    for (int i = 0; i < 8; ++i) ranks[i].build();
#endif
    return ranks;
  }
}}  // bce::builtins

#endif  // BCE_BUILTINS
