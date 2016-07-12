/**
 * BCE v0.4 - compressor for stationary data
 * The MIT License (MIT)
 * Copyright (c) 2016  Christoph Diegelmann
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files (the "Software"), to deal 
 * in the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the Software is furnished 
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all 
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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

  inline uint32_t clz(uint64_t val) {
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

  inline uint32_t ctz(uint64_t val) {
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

  inline uint32_t cnt(uint64_t val) {
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

  inline constexpr uint32_t clz(uint64_t val) {
    return sizeof(unsigned long) == 8 ? __builtin_clzl(val) :  __builtin_clzll(val);
  }

  inline constexpr uint32_t ctz(uint64_t val) {
    return sizeof(unsigned long) == 8 ? __builtin_ctzl(val) : __builtin_ctzll(val);
  }

  inline constexpr uint32_t cnt(uint64_t val) {
    return sizeof(unsigned long) == 8 ? __builtin_popcountl(val) : __builtin_popcountll(val);
  }
#endif
  inline uint32_t clo(uint64_t val) {
    return clz(~val);
  }

  inline uint32_t cto(uint64_t val) {
    return ctz(~val);
  }

  template<class ForwardIt>
  ForwardIt to_lmsr(ForwardIt first, ForwardIt last) {
    auto size = static_cast<uint32_t>(last - first);
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
      ranks[i] = rank(static_cast<uint32_t>(last - first));

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
