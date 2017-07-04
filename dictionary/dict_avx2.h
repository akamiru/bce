#ifndef DICTIONARY_AVX2_H
#define DICTIONARY_AVX2_H

// C++ headers
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

// Architecture headers
#include <immintrin.h>  // AVX2
#include <nmmintrin.h>  // POPCNT

// Project headers
#include "../vector/int32x8_avx2.h"

namespace bce {

namespace detail {

template<>
inline void binary_partition<int32x8_avx2>(std::uint64_t* dictionary, std::uint64_t& rank, std::uint8_t* data, std::uint8_t*& bytes_0, std::uint8_t*& bytes_1) {
  // load data into register
  auto dlocal = int32x8_avx2::load(reinterpret_cast<int32_t*>(data));

  // extract and count mask
  std::uint32_t mask  = _mm256_movemask_epi8((dlocal << 7).get_native());
  std::uint64_t cnt_1 = _mm_popcnt_u64(mask);
  std::uint64_t cnt_0 = 32 - cnt_1;

  // store into the dictionary
  *dictionary = (rank << 32) | mask;
  rank += cnt_1;

  // copy and update for next loop
  auto* lbytes_0 = bytes_0;
  auto* lbytes_1 = bytes_1;
  bytes_0 += cnt_0;
  bytes_1 += cnt_1;

  // rotate the data by one bit
  dlocal = dlocal >> 1;

  // spread the mask
  auto mask_lo = _pdep_u64(mask >>  0, UINT64_C(0x1111111111111111)) * 0xF;
  auto mask_hi = _pdep_u64(mask >> 16, UINT64_C(0x1111111111111111)) * 0xF;

  // move one  masks into the vector
  auto mask_lo1 = _pext_u64(UINT64_C(0xfedcba9876543210), mask_lo);
  auto mask_hi1 = _pext_u64(UINT64_C(0xfedcba9876543210), mask_hi);
  auto xmask_1 = _mm_set_epi64x(mask_hi1, mask_lo1);

  // move zero masks into the vector
  auto mask_lo0 = _pext_u64(UINT64_C(0xfedcba9876543210),~mask_lo);
  auto mask_hi0 = _pext_u64(UINT64_C(0xfedcba9876543210),~mask_hi);
  auto xmask_0 = _mm_set_epi64x(mask_hi0, mask_lo0);

  // count lower part
  cnt_1 = _mm_popcnt_u64(mask & 0xFFFF);
  cnt_0 = 16 - cnt_1;

  // extend masks  0xAB -> 0x00AB
  int32x8_avx2 ymask_1{_mm256_cvtepu8_epi16(xmask_1)};
  int32x8_avx2 ymask_0{_mm256_cvtepu8_epi16(xmask_0)};

  // shift masks   0x00AB -> 0x0AXB
  ymask_1 = ymask_1 ^ (ymask_1 << 4);
  ymask_0 = ymask_0 ^ (ymask_0 << 4);

  // clean masks   0x0AXB -> 0x0A0B
  ymask_1 = ymask_1 & int32x8_avx2{0x0F0F0F0F};
  ymask_0 = ymask_0 & int32x8_avx2{0x0F0F0F0F};

  // shuffle
  auto ydata_1 = dlocal.shuffle(ymask_1.get_native());
  auto ydata_0 = dlocal.shuffle(ymask_0.get_native());

  // store zero part
  ydata_0.extract<0>().storeu(lbytes_0);
  lbytes_0 += cnt_0;
  ydata_0.extract<1>().storeu(lbytes_0);

  // store one part
  ydata_1.extract<0>().storeu(lbytes_1);
  lbytes_1 += cnt_1;
  ydata_1.extract<1>().storeu(lbytes_1);
}

}  // namespace detail

/**
 * wavelet matrix + rank dictionary
 *
 */
template<>
class dictionary<bce::int32x8_avx2> {
 public:
    typedef bce::int32x8_avx2 vector_type;

    /**
     * Represents a queryable view on the dictionary
     * e.g for a bit offset
     *
     */
    class view {
     public:
        explicit view(std::uint64_t* storage) : storage_(storage) {}

        /**
         * Get ranks for given indices
         */
        vector_type query(vector_type offsets) const {
          // constants
          const int32x8_avx2 lut{_mm256_setr_epi8(
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
          )};

          // calculate the offsets
          auto index = offsets >> 5;
#if 1
          // this results in [01][45][23][67] when gathering
          index = int32x8_avx2{_mm256_permute4x64_epi64(index.get_native(), _MM_SHUFFLE(3,1,2,0))};

          // gather the values
          auto storage = reinterpret_cast<const long long int*>(this->storage_);
          __m256i dictl = _mm256_i32gather_epi64(storage, index.extract<0>().get_native(), 8);
          __m256i dicth = _mm256_i32gather_epi64(storage, index.extract<1>().get_native(), 8);

          // seperates ranks and bits
          dictl = _mm256_shuffle_epi32(dictl, _MM_SHUFFLE(3,1,2,0));
          dicth = _mm256_shuffle_epi32(dicth, _MM_SHUFFLE(3,1,2,0));

          // unpack merges lows and highs: [01][23][45][56]
          int32x8_avx2 bits{_mm256_unpacklo_epi64(dictl, dicth)};
          int32x8_avx2 rank{_mm256_unpackhi_epi64(dictl, dicth)};
#else
          // IACA says faster on Broadwell
          auto storage = reinterpret_cast<const int32_t*>(this->storage_);
          int32x8_avx2 bits{_mm256_i32gather_epi32(storage + 0, index.get_native(), 8)};
          int32x8_avx2 rank{_mm256_i32gather_epi32(storage + 1, index.get_native(), 8)};
#endif
          //print256_num(bits);
          //print256_num(rank);

          // extract the lowest 5 bit and mask out
#if 1
          offsets = offsets & int32x8_avx2{0x1F};
          offsets = int32x8_avx2{-1} << offsets;
          bits    = bits.and_not(offsets);
#else
          // Strangely this doesn't seem to be faster
          // to store (bits+bits) to save a cycle (we never count the highest bit)
          offsets = int32x8_avx2{0x1F}.and_not(offsets);
          bits    = (bits + bits) << offsets;
#endif

          // byte popcount
          auto bitsh = bits >> 4;
          auto bitsl = bits  & int32x8_avx2{0x0F0F0F0F};
               bitsh = bitsh & int32x8_avx2{0x0F0F0F0F};
               bitsl = lut.shuffle(bitsl.get_native());
               bitsh = lut.shuffle(bitsh.get_native());

          // uint32_t popcount
          bits = bitsl + bitsh;
          bits = bits  + (bits << 16);
          bits = bits  + (bits <<  8);
          bits = bits >> 24;

          return rank + bits;
        }

     private:
        std::uint64_t* storage_;
    };

    /**
     * Initialise the underlying memory
     *
     * Storage should map to 2 * N + 32 bytes
     * where N is the size of the data
     *
     * @param T*       storage  memory to store the dictionary
     * @param int32_t  size     size of the text N
     */
    template<class T>
    dictionary(T* storage, int32_t size) :
      storage_(reinterpret_cast<std::uint64_t*>(storage)),
      size_(size * sizeof(T)) {}

    /**
     * Construct a dictionary from data
     *
     * @param uint8_t* data     data to construct the dictionary for
     * @param uint8_t* work     temporary working space
     */
    void construct(std::uint8_t* data, std::uint8_t* work) {
      // count zeros
      this->count_zeros(data);

      // iterator over the storage
      auto storage = this->storage_;
      for (std::size_t bit = 0; bit < 7; ++bit) {
        // current rank
        std::uint64_t rank = 0;

        // sorting targets
        auto bytes_0 = work;
        auto bytes_1 = work + this->get_zeros(bit);
        auto base    = bytes_1 - 32;

        // current data
        auto* cdata = data;

        // Simply store all non overriding
        for (; bytes_0 < base; cdata += 32) {
          detail::binary_partition<vector_type>(
            storage++,
            rank,
            cdata,
            bytes_0,
            bytes_1
          );
        }

        base += 32;
        // at this point either we are finished with the
        // possibly overriden area or after at most another
        // 48 Byte we will be
        std::size_t remaining_bytes = (data + this->size_) - cdata;
        if (48 < remaining_bytes)
          remaining_bytes = 48;

        if (bytes_1 - base < 16) {
          for (auto j = cdata + remaining_bytes; cdata < j; cdata += 32) {
            // load the unfinished section that might be overriden
            auto critical = int32x8_avx2::loadu(base);
            // save the current position
            auto lbytes_1 = bytes_1;

            // partition
            detail::binary_partition<vector_type>(
              storage++,
              rank,
              cdata,
              bytes_0,
              bytes_1
            );

            // load the newly added bytes
            auto added = int32x8_avx2::loadu(lbytes_1);
            // restore the critical possibly overriding the new
            critical.storeu(base    );
            // restore the added bytes
            added   .storeu(lbytes_1);
          }
        }

        // at this point at least the critical section is finished
        // so we save it
        auto critical = int32x4_sse4::loadu(base);

        // and finish our job
        for (; cdata < data + this->size_; cdata += 32) {
          detail::binary_partition<vector_type>(
            storage++,
            rank,
            cdata,
            bytes_0,
            bytes_1
          );
        }

        // finally restore the critical
        critical.storeu(base);

        // store the last rank
        *storage++ = rank << 32;

        // swap working space with data
        std::swap(data, work);
      }

      // current rank
      std::uint64_t rank = 0;

      #pragma unroll(1)
      for (std::size_t i = 0; i < this->size_; i += 32) {
        // load data into register
        auto dlocal = int32x8_avx2::load(reinterpret_cast<int32_t*>(data + i));

        // extract and count mask
        std::uint32_t mask  = _mm256_movemask_epi8((dlocal << 7).get_native());

        // store into the dictionary
        *storage++ = (rank << 32) | mask;

        // update rank
        rank += _mm_popcnt_u64(mask);
      }
      // store the last rank
      *storage = rank << 32;
    }

    /**
     * Count the zeros at each position
     *
     * @param uint8_t* data     data to count the bits for
     */
    void count_zeros(std::uint8_t* data) {
      // accumulators
      std::int32_t a = 0, b = 0;       // scalar
      __m256i vc, vd, ve, vf, vg, vh;  // vector
      vc = vd = ve = vf = vg = vh = _mm256_setzero_si256();

      // vector masks
      __m256i n, o, p, q, r, s;
      n = _mm256_set1_epi8(0x20);
      o = _mm256_set1_epi8(0x10);
      p = _mm256_set1_epi8(0x08);
      q = _mm256_set1_epi8(0x04);
      r = _mm256_set1_epi8(0x02);
      s = _mm256_set1_epi8(0x01);

      // runs at ~8 cycles per 32 byte
      for (std::uint64_t i = 0; i < this->size_; i += 32) {
        auto dataA = _mm256_loadu_si256(reinterpret_cast<__m256i*>(data + i));
        a += _mm_popcnt_u32(_mm256_movemask_epi8(dataA));
        b += _mm_popcnt_u32(_mm256_movemask_epi8(_mm256_add_epi16(dataA, dataA)));
        vc = _mm256_add_epi64(vc, _mm256_sad_epu8(_mm256_and_si256(dataA, n), _mm256_setzero_si256()));
        vd = _mm256_add_epi64(vd, _mm256_sad_epu8(_mm256_and_si256(dataA, o), _mm256_setzero_si256()));
        ve = _mm256_add_epi64(ve, _mm256_sad_epu8(_mm256_and_si256(dataA, p), _mm256_setzero_si256()));
        vf = _mm256_add_epi64(vf, _mm256_sad_epu8(_mm256_and_si256(dataA, q), _mm256_setzero_si256()));
        vg = _mm256_add_epi64(vg, _mm256_sad_epu8(_mm256_and_si256(dataA, r), _mm256_setzero_si256()));
        vh = _mm256_add_epi64(vh, _mm256_sad_epu8(_mm256_and_si256(dataA, s), _mm256_setzero_si256()));
      }

      // add low and high halves
      auto mmC = _mm_add_epi64(_mm256_extracti128_si256(vc, 0), _mm256_extracti128_si256(vc, 1));
      auto mmD = _mm_add_epi64(_mm256_extracti128_si256(vd, 0), _mm256_extracti128_si256(vd, 1));
      auto mmE = _mm_add_epi64(_mm256_extracti128_si256(ve, 0), _mm256_extracti128_si256(ve, 1));
      auto mmF = _mm_add_epi64(_mm256_extracti128_si256(vf, 0), _mm256_extracti128_si256(vf, 1));
      auto mmG = _mm_add_epi64(_mm256_extracti128_si256(vg, 0), _mm256_extracti128_si256(vg, 1));
      auto mmH = _mm_add_epi64(_mm256_extracti128_si256(vh, 0), _mm256_extracti128_si256(vh, 1));

      auto h = static_cast<std::uint32_t>((_mm_cvtsi128_si64(mmH) + _mm_extract_epi64(mmH, 1)) >> 0);
      auto g = static_cast<std::uint32_t>((_mm_cvtsi128_si64(mmG) + _mm_extract_epi64(mmG, 1)) >> 1);
      auto f = static_cast<std::uint32_t>((_mm_cvtsi128_si64(mmF) + _mm_extract_epi64(mmF, 1)) >> 2);
      auto e = static_cast<std::uint32_t>((_mm_cvtsi128_si64(mmE) + _mm_extract_epi64(mmE, 1)) >> 3);
      auto d = static_cast<std::uint32_t>((_mm_cvtsi128_si64(mmD) + _mm_extract_epi64(mmD, 1)) >> 4);
      auto c = static_cast<std::uint32_t>((_mm_cvtsi128_si64(mmC) + _mm_extract_epi64(mmC, 1)) >> 5);

      // calculate the number of zeros
      this->zeros_[0] = this->size_ - h;
      this->zeros_[1] = this->size_ - g;
      this->zeros_[2] = this->size_ - f;
      this->zeros_[3] = this->size_ - e;
      this->zeros_[4] = this->size_ - d;
      this->zeros_[5] = this->size_ - c;
      this->zeros_[6] = this->size_ - b;
      this->zeros_[7] = this->size_ - a;

#if 0
      printf("zeros_[0] = %9i\n", this->zeros_[0]);
      printf("zeros_[1] = %9i\n", this->zeros_[1]);
      printf("zeros_[2] = %9i\n", this->zeros_[2]);
      printf("zeros_[3] = %9i\n", this->zeros_[3]);
      printf("zeros_[4] = %9i\n", this->zeros_[4]);
      printf("zeros_[5] = %9i\n", this->zeros_[5]);
      printf("zeros_[6] = %9i\n", this->zeros_[6]);
      printf("zeros_[7] = %9i\n", this->zeros_[7]);
#endif
    }

    /**
     * Get the number of zeros at given position
     *
     * @param uint64_t bit      position of the bit to count
     */
    inline std::int32_t get_zeros(std::uint64_t bit) const {
      return this->zeros_[bit];
    }

    /**
     * Get a queryable view on the dictionary
     *
     * @param uint64_t bit      position of the bit to count
     */
    inline view get_view(std::uint64_t bit) const {
      return dictionary::view(this->storage_ + ((size_ + 31) / 32 + 1) * bit);
    }

 private:
    std::uint64_t*              storage_;
    std::size_t                 size_   ;
    std::array<std::int32_t, 8> zeros_  ;
};

}  // namespace bce

#endif  // DICTIONARY_AVX2_H