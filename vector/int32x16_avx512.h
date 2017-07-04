#ifndef INT32x16_AVX512_H
#define INT32x16_AVX512_H

// C++ headers
#include <cstdint>

// Architecture headers
#include "immintrin.h"  // AVX512
#include <nmmintrin.h>  // POPCNT
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t), "AVX512 requires x64");

// Project headers
#include "int32x8_avx2.h"

namespace bce {

class int32x16_avx512 {
 public:
    static constexpr std::size_t size = 16;  // fits 16 ints
    static constexpr std::size_t has_vector_addressing = 1;
    typedef __m512i   permute_mask_type;
    typedef __mmask16 pack_mask_type;

    struct mask_type {
        friend class int32x16_avx512;

     public:
        // Constructor
        explicit mask_type(__mmask16 value) : value_(value) {}

        // Integer functions
        inline std::uint64_t as_int() const {
          return _mm512_kmov(this->value_);
        }

        inline std::uint64_t popcnt() const {
          return _mm_popcnt_u64(this->as_int());
        }

        // Left packing functions
        inline std::int32_t* pack_advance(std::int32_t* ptr) const {
          return ptr + this->popcnt();
        }

        pack_mask_type pack_mask() const {
          return pack_mask_type{this->value_};
        }

        // Bit level operators
        inline mask_type operator&(const mask_type& rhs) const {
          return mask_type{_mm512_kand(this->value_, rhs.value_)};
        }

        inline mask_type operator|(const mask_type& rhs) const {
          return mask_type{_mm512_kor (this->value_, rhs.value_)};
        }

        inline mask_type operator^(const mask_type& rhs) const {
          return mask_type{_mm512_kxor(this->value_, rhs.value_)};
        }

     private:
        __mmask16 value_;
    };

    // Constructors
    explicit int32x16_avx512(std::int32_t value) : value_(_mm512_set1_epi32(value)) {}
    explicit int32x16_avx512(__m512i  value) : value_(value) {}

    // Memory access
    static inline int32x16_avx512 load(const std::int32_t* ptr) {
      return int32x16_avx512{_mm512_load_epi32(reinterpret_cast<__m512i*>(ptr))};
    }

    template<class T>
    static inline int32x16_avx512 loadu(const T* ptr) {
      return int32x16_avx512{_mm512_loadu_si512(reinterpret_cast<__m512i*>(ptr))};
    }

    inline void store(std::int32_t* ptr) const {
      _mm512_store_si512(reinterpret_cast<__m512i*>(ptr), this->value_);
    }

    template<class T>
    inline void storeu(T* ptr) const {
      _mm512_storeu_si512(reinterpret_cast<__m512i*>(ptr), this->value_);
    }

    // Conditional selection
    static inline int32x16_avx512 blend(
      const int32x16_avx512& lhs,
      const int32x16_avx512& rhs,
      const int32x16_avx512::mask_type& mask
    ) {
      return int32x16_avx512{_mm512_mask_blend_epi32(mask.value_, lhs.value_, rhs.value_)};
    }

    // Extrema
    static inline int32x16_avx512 min(const int32x16_avx512& lhs, const int32x16_avx512& rhs) {
      return int32x16_avx512{_mm512_min_epi32(lhs.value_, rhs.value_)};
    }

    static inline int32x16_avx512 max(const int32x16_avx512& lhs, const int32x16_avx512& rhs) {
      return int32x16_avx512{_mm512_max_epi32(lhs.value_, rhs.value_)};
    }

    // Permutation
    inline int32x16_avx512 permute(const permute_mask_type& mask) const {
      return int32x16_avx512{_mm512_permutevar_epi32(mask, this->value_)};
    }

    inline int32x16_avx512 shuffle(const permute_mask_type& mask) const {
      return int32x16_avx512{_mm512_shuffle_epi8(mask, this->value_)};
    }

    inline int32x16_avx512 pack(const pack_mask_type& mask) const {
      return int32x16_avx512{_mm512_mask_compress_epi32(this->value_, mask, this->value_)};
    }

    inline void pack_and_store(std::int32_t* ptr, const pack_mask_type& mask) const {
      _mm512_mask_compressstoreu_epi32(reinterpret_cast<__m256i*>(ptr), mask, this->value_);
    }

    // Arithmetic operators
    inline int32x16_avx512 operator* (const int32x16_avx512& rhs) const {
      return int32x16_avx512{_mm512_mullo_epi32(this->value_, rhs.value_)};
    }

    inline int32x16_avx512 operator+ (const int32x16_avx512& rhs) const {
      return int32x16_avx512{_mm512_add_epi32(this->value_, rhs.value_)};
    }

    inline int32x16_avx512 operator- (const int32x16_avx512& rhs) const {
      return int32x16_avx512{_mm512_sub_epi32(this->value_, rhs.value_)};
    }

    // Bit level operators
    inline int32x16_avx512 operator& (const int32x16_avx512& rhs) const {
      return int32x16_avx512{_mm512_and_si512(this->value_, rhs.value_)};
    }

    inline int32x16_avx512 and_not   (const int32x16_avx512& rhs) const {
      return int32x16_avx512{_mm512_andnot_si512(rhs.value_, this->value_)};
    }

    inline int32x16_avx512 operator| (const int32x16_avx512& rhs) const {
      return int32x16_avx512{_mm512_or_si512(this->value_, rhs.value_)};
    }

    inline int32x16_avx512 operator^ (const int32x16_avx512& rhs) const {
      return int32x16_avx512{_mm512_xor_si512(this->value_, rhs.value_)};
    }

    // Shift operators
    inline int32x16_avx512 operator<<(int rhs) const {
      return int32x16_avx512{_mm512_slli_epi32(this->value_, rhs)};
    }

    inline int32x16_avx512 operator<<(const int32x16_avx512& rhs) const {
      return int32x16_avx512{_mm512_sllv_epi32(this->value_, rhs.value_)};
    }

    // performs a logic right shift !
    inline int32x16_avx512 operator>>(int rhs) const {
      return int32x16_avx512{_mm512_srli_epi32(this->value_, rhs)};
    }

    inline int32x16_avx512 operator>>(const int32x16_avx512& rhs) const {
      return int32x16_avx512{_mm512_srlv_epi32(this->value_, rhs.value_)};
    }

    // Comparsion operators
    inline int32x16_avx512::mask_type operator< (const int32x16_avx512& rhs) const {
      return int32x16_avx512::mask_type{_mm512_cmpgt_epi32_mask(rhs.value_, this->value_)};
    }

    inline int32x16_avx512::mask_type operator> (const int32x16_avx512& rhs) const {
      return int32x16_avx512::mask_type{_mm512_cmpgt_epi32_mask(this->value_, rhs.value_)};
    }

    inline int32x16_avx512::mask_type operator==(const int32x16_avx512& rhs) const {
      return int32x16_avx512::mask_type{_mm512_cmpeq_epi32_mask(this->value_, rhs.value_)};
    }

    // Access to the underlying type
    inline __m512i get_native() {
      return this->value_;
    }

    template<int I>
    inline int32x8_avx2 extract_avx() {
      return int32x8_avx2{_mm512_extracti32x8_epi32(this->value_, I)};
    }

    template<int I>
    inline int32x4_sse extract_sse() {
      return int32x4_sse{_mm512_extracti32x4_epi32(this->value_, I)};
    }
 private:
    __m512i value_;
};

// Partial specialization for extract
template<>
inline int32x8_avx2 int32x16_avx512::extract_avx<0>() {
  return int32x8_avx2{_mm512_castsi512_si256(this->value_)};
}

template<>
inline int32x4_sse int32x16_avx512::extract_sse<0>() {
  return int32x4_sse{_mm512_castsi512_si128(this->value_)};
}

}  // namespace bce

#endif  // INT32x16_AVX512_H