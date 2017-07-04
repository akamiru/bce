#ifndef INT32x8_AVX2_H
#define INT32x8_AVX2_H

// C++ headers
#include <cstdint>

// Architecture headers
#include <immintrin.h>  // AVX
#include <nmmintrin.h>  // POPCNT
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t), "AVX2 requires x64");

// Project headers
#include "int32x4_sse4.h"

namespace bce {

class int32x8_avx2 {
 public:
    static constexpr std::size_t size = 8;  // fits 8 ints
    static constexpr std::size_t has_vector_addressing = 1;
    typedef __m256i permute_mask_type;
    typedef __m256i pack_mask_type;

    struct mask_type {
        friend class int32x8_avx2;

     public:
        // Constructor
        explicit mask_type(__m256i value) : value_(value) {}

        // Integer functions
        inline std::uint64_t as_int() const {
          // cast to unsigned first to avoid propergating the sign bit
          return static_cast<unsigned>(_mm256_movemask_epi8(this->value_));
        }

        inline std::uint64_t popcnt() const {
          return _mm_popcnt_u64(this->as_int());
        }

        // Left packing functions
        inline std::int32_t* pack_advance(std::int32_t* ptr) const {
          return reinterpret_cast<std::int32_t*>(
            reinterpret_cast<std::uint8_t*>(ptr) + this->popcnt()
          );
        }

        pack_mask_type pack_mask() const {
          // extract indices we want to forward
          std::uint64_t indices = _pdep_u64(
            _pext_u64(0x76543210, this->as_int()),
            UINT64_C(0x0F0F0F0F0F0F0F0F)
          );

          // extend the indices to produce the shuffle mask
          return _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(indices));
        }

        // Bit level operators
        inline mask_type operator&(const mask_type& rhs) const {
          return mask_type{_mm256_and_si256(this->value_, rhs.value_)};
        }

        inline mask_type operator|(const mask_type& rhs) const {
          return mask_type{_mm256_or_si256(this->value_, rhs.value_)};
        }

        inline mask_type operator^(const mask_type& rhs) const {
          return mask_type{_mm256_xor_si256(this->value_, rhs.value_)};
        }

     private:
        __m256i value_;
    };

    // Constructors
    explicit int32x8_avx2(std::int32_t value) : value_(_mm256_set1_epi32(value)) {}
    explicit int32x8_avx2(__m256i  value) : value_(value) {}

    // Memory access
    static inline int32x8_avx2 load(const std::int32_t* ptr) {
      return int32x8_avx2{_mm256_load_si256(reinterpret_cast<const __m256i*>(ptr))};
    }

    template<class T>
    static inline int32x8_avx2 loadu(const T* ptr) {
      return int32x8_avx2{_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr))};
    }

    inline void store(std::int32_t* ptr) const {
      _mm256_store_si256(reinterpret_cast<__m256i*>(ptr), this->value_);
    }

    template<class T>
    inline void storeu(T* ptr) const {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(ptr), this->value_);
    }

    // Conditional selection
    static inline int32x8_avx2 blend(
      const int32x8_avx2& lhs,
      const int32x8_avx2& rhs,
      const int32x8_avx2::mask_type& mask
    ) {
      return int32x8_avx2{_mm256_blendv_epi8(lhs.value_, rhs.value_, mask.value_)};
    }

    // Extrema
    static inline int32x8_avx2 min(const int32x8_avx2& lhs, const int32x8_avx2& rhs) {
      return int32x8_avx2{_mm256_min_epi32(lhs.value_, rhs.value_)};
    }

    static inline int32x8_avx2 max(const int32x8_avx2& lhs, const int32x8_avx2& rhs) {
      return int32x8_avx2{_mm256_max_epi32(lhs.value_, rhs.value_)};
    }

    // Permutation
    inline int32x8_avx2 permute(const permute_mask_type& mask) const {
      return int32x8_avx2{_mm256_permutevar8x32_epi32(this->value_, mask)};
    }

    inline int32x8_avx2 shuffle(const permute_mask_type& mask) const {
      return int32x8_avx2{_mm256_shuffle_epi8(this->value_, mask)};
    }

    inline int32x8_avx2 pack(const pack_mask_type& mask) const {
      return int32x8_avx2{_mm256_permutevar8x32_epi32(this->value_, mask)};
    }

    inline void pack_and_store(std::int32_t* ptr, const pack_mask_type& mask) const {
      this->pack(mask).storeu(ptr);  // single instruction in avx512
    }

    // Arithmetic operators
    inline int32x8_avx2 operator* (const int32x8_avx2& rhs) const {
      return int32x8_avx2{_mm256_mullo_epi32(this->value_, rhs.value_)};
    }

    inline int32x8_avx2 operator+ (const int32x8_avx2& rhs) const {
      return int32x8_avx2{_mm256_add_epi32(this->value_, rhs.value_)};
    }

    inline int32x8_avx2 operator- (const int32x8_avx2& rhs) const {
      return int32x8_avx2{_mm256_sub_epi32(this->value_, rhs.value_)};
    }

    // Bit level operators
    inline int32x8_avx2 operator& (const int32x8_avx2& rhs) const {
      return int32x8_avx2{_mm256_and_si256(this->value_, rhs.value_)};
    }

    inline int32x8_avx2 and_not (const int32x8_avx2& rhs) const {
      return int32x8_avx2{_mm256_andnot_si256(rhs.value_, this->value_)};
    }

    inline int32x8_avx2 operator| (const int32x8_avx2& rhs) const {
      return int32x8_avx2{_mm256_or_si256(this->value_, rhs.value_)};
    }

    inline int32x8_avx2 operator^ (const int32x8_avx2& rhs) const {
      return int32x8_avx2{_mm256_xor_si256(this->value_, rhs.value_)};
    }

    // Shift operators
    inline int32x8_avx2 operator<<(int rhs) const {
      return int32x8_avx2{_mm256_slli_epi32(this->value_, rhs)};
    }

    inline int32x8_avx2 operator<<(const int32x8_avx2& rhs) const {
      return int32x8_avx2{_mm256_sllv_epi32(this->value_, rhs.value_)};
    }

    // performs a logic right shift !
    inline int32x8_avx2 operator>>(int rhs) const {
      return int32x8_avx2{_mm256_srli_epi32(this->value_, rhs)};
    }

    inline int32x8_avx2 operator>>(const int32x8_avx2& rhs) const {
      return int32x8_avx2{_mm256_srlv_epi32(this->value_, rhs.value_)};
    }

    // Comparsion operators
    inline int32x8_avx2::mask_type operator< (const int32x8_avx2& rhs) const {
      return int32x8_avx2::mask_type{_mm256_cmpgt_epi32(rhs.value_, this->value_)};
    }

    inline int32x8_avx2::mask_type operator> (const int32x8_avx2& rhs) const {
      return int32x8_avx2::mask_type{_mm256_cmpgt_epi32(this->value_, rhs.value_)};
    }

    inline int32x8_avx2::mask_type operator==(const int32x8_avx2& rhs) const {
      return int32x8_avx2::mask_type{_mm256_cmpeq_epi32(this->value_, rhs.value_)};
    }

    // Access to the underlying type
    inline __m256i get_native() {
      return this->value_;
    }

    template<int I>
    inline int32x4_sse4 extract() {
      return int32x4_sse4{_mm256_extracti128_si256(this->value_, I)};
    }
 private:
    __m256i value_;
};

// Partial specialization for extract
template<>
inline int32x4_sse4 int32x8_avx2::extract<0>() {
  return int32x4_sse4{_mm256_castsi256_si128(this->value_)};
}

}  // namespace bce

#endif  // INT32x8_AVX2_H