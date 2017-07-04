#ifndef INT32x4_SSE4_H
#define INT32x4_SSE4_H

// C++ headers
#include <cstdint>

// Architecture headers
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <tmmintrin.h>  // SSSE3
#include <smmintrin.h>  // SSE4.1
#include <immintrin.h>  // SSE4.2
#include <nmmintrin.h>  // POPCNT

// Project headers
#include "int32x1_cpp.h"

namespace bce {

class int32x4_sse4 {
 public:
    static constexpr std::size_t size = 4;  // fits 4 ints
    static constexpr std::size_t has_vector_addressing = 0;
    typedef __m128i permute_mask_type;
    typedef __m128i pack_mask_type;

    struct mask_type {
        friend class int32x4_sse4;

     public:
        // Constructor
        explicit mask_type(__m128i value) : value_(value) {}

        // Integer functions
        inline std::size_t as_int() const {
          // cast to unsigned first to avoid propergating the sign bit
          return static_cast<unsigned>(_mm_movemask_ps(_mm_cvtepi32_ps(this->value_)));
        }

        inline std::size_t popcnt() const {
          if (sizeof(std::size_t) == sizeof(std::uint64_t))
            return _mm_popcnt_u64(this->as_int());
          return _mm_popcnt_u32(this->as_int());
        }

        // Left packing functions
        inline std::int32_t* pack_advance(std::int32_t* ptr) const {
          return ptr + this->popcnt();
        }

        pack_mask_type pack_mask() const {
          return _mm_load_si128(&mask_type::pack_table[this->as_int()]);
        }

        // Bit level operators
        inline mask_type operator&(const mask_type& rhs) const {
          return mask_type{_mm_and_si128(this->value_, rhs.value_)};
        }

        inline mask_type operator|(const mask_type& rhs) const {
          return mask_type{_mm_or_si128(this->value_, rhs.value_)};
        }

        inline mask_type operator^(const mask_type& rhs) const {
          return mask_type{_mm_xor_si128(this->value_, rhs.value_)};
        }

     private:
        // lookup table for left packing
        static const __m128i pack_table[16];

        __m128i value_;
    };

    // Constructors
    explicit int32x4_sse4(std::int32_t value) : value_(_mm_set1_epi32(value)) {}
    explicit int32x4_sse4(__m128i  value) : value_(value) {}

    // Memory access
    static inline int32x4_sse4 load(const std::int32_t* ptr) {
      return int32x4_sse4{_mm_load_si128(reinterpret_cast<const __m128i*>(ptr))};
    }

    template<class T>
    static inline int32x4_sse4 loadu(const T* ptr) {
      return int32x4_sse4{_mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr))};
    }

    inline void store(std::int32_t* ptr) const {
      _mm_store_si128(reinterpret_cast<__m128i*>(ptr), this->value_);
    }

    template<class T>
    inline void storeu(T* ptr) const {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr), this->value_);
    }

    // Conditional selection
    static inline int32x4_sse4 blend(
      const int32x4_sse4& lhs,
      const int32x4_sse4& rhs,
      const int32x4_sse4::mask_type& mask
    ) {
      return int32x4_sse4{_mm_blendv_epi8(lhs.value_, rhs.value_, mask.value_)};
    }

    // Extrema
    static inline int32x4_sse4 min(const int32x4_sse4& lhs, const int32x4_sse4& rhs) {
      return int32x4_sse4{_mm_min_epi32(lhs.value_, rhs.value_)};
    }

    static inline int32x4_sse4 max(const int32x4_sse4& lhs, const int32x4_sse4& rhs) {
      return int32x4_sse4{_mm_max_epi32(lhs.value_, rhs.value_)};
    }

    // Permutation
    inline int32x4_sse4 permute(const permute_mask_type& mask) const {
      return int32x4_sse4{_mm_shuffle_epi8(this->value_, mask)};
    }

    inline int32x4_sse4 shuffle(const permute_mask_type& mask) const {
      return int32x4_sse4{_mm_shuffle_epi8(this->value_, mask)};
    }

    inline int32x4_sse4 pack(const pack_mask_type& mask) const {
      return int32x4_sse4{_mm_shuffle_epi8(this->value_, mask)};
    }

    inline void pack_and_store(std::int32_t* ptr, const pack_mask_type& mask) const {
      this->pack(mask).storeu(ptr);  // single instruction in avx512
    }

    // Arithmetic operators
    inline int32x4_sse4 operator* (const int32x4_sse4& rhs) const {
      return int32x4_sse4{_mm_mullo_epi32(this->value_, rhs.value_)};
    }

    inline int32x4_sse4 operator+ (const int32x4_sse4& rhs) const {
      return int32x4_sse4{_mm_add_epi32(this->value_, rhs.value_)};
    }

    inline int32x4_sse4 operator- (const int32x4_sse4& rhs) const {
      return int32x4_sse4{_mm_sub_epi32(this->value_, rhs.value_)};
    }

    // Bit level operators
    inline int32x4_sse4 operator& (const int32x4_sse4& rhs) const {
      return int32x4_sse4{_mm_and_si128(this->value_, rhs.value_)};
    }

    inline int32x4_sse4 and_not   (const int32x4_sse4& rhs) const {
      return int32x4_sse4{_mm_andnot_si128(rhs.value_, this->value_)};
    }

    inline int32x4_sse4 operator| (const int32x4_sse4& rhs) const {
      return int32x4_sse4{_mm_or_si128(this->value_, rhs.value_)};
    }

    inline int32x4_sse4 operator^ (const int32x4_sse4& rhs) const {
      return int32x4_sse4{_mm_xor_si128(this->value_, rhs.value_)};
    }

    // Shift operators
    inline int32x4_sse4 operator<<(int rhs) const {
      return int32x4_sse4{_mm_slli_epi32(this->value_, rhs)};
    }

    inline int32x4_sse4 operator<<(const int32x4_sse4& rhs) const {
      return int32x4_sse4{_mm_sllv_epi32(this->value_, rhs.value_)};
    }

    // performs a logic right shift !
    inline int32x4_sse4 operator>>(int rhs) const {
      return int32x4_sse4{_mm_srli_epi32(this->value_, rhs)};
    }

    inline int32x4_sse4 operator>>(const int32x4_sse4& rhs) const {
      return int32x4_sse4{_mm_srlv_epi32(this->value_, rhs.value_)};
    }

    // Comparsion operators
    inline int32x4_sse4::mask_type operator< (const int32x4_sse4& rhs) const {
      return int32x4_sse4::mask_type{_mm_cmpgt_epi32(rhs.value_, this->value_)};
    }

    inline int32x4_sse4::mask_type operator> (const int32x4_sse4& rhs) const {
      return int32x4_sse4::mask_type{_mm_cmpgt_epi32(this->value_, rhs.value_)};
    }

    inline int32x4_sse4::mask_type operator==(const int32x4_sse4& rhs) const {
      return int32x4_sse4::mask_type{_mm_cmpeq_epi32(this->value_, rhs.value_)};
    }

    //inline int32x4_sse4::mask_type conflict_mask() const {
    //  return int32x4_sse4::mask_type{};
    //}

    // access to the underlying type
    inline __m128i get_native() {
      return this->value_;
    }

    template<int I>
    inline int32x1_cpp extract() {
      return int32x1_cpp{_mm_extract_epi32(this->value_, I)};
    }

 private:
    __m128i value_;
};

// Partial specialization for extract
template<>
int32x1_cpp int32x4_sse4::extract<0>() {
  return int32x1_cpp{_mm_cvtsi128_si32(this->value_)};
}

const __m128i int32x4_sse4::mask_type::pack_table[16]{
  _mm_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x80808080),  // 0000
  _mm_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x03020100),  // 0001
  _mm_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x07060504),  // 0010
  _mm_set_epi32(0x80808080, 0x80808080, 0x07060504, 0x03020100),  // 0011
  _mm_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x0b0a0908),  // 0100
  _mm_set_epi32(0x80808080, 0x80808080, 0x0b0a0908, 0x03020100),  // 0101
  _mm_set_epi32(0x80808080, 0x80808080, 0x0b0a0908, 0x07060504),  // 0110
  _mm_set_epi32(0x80808080, 0x0b0a0908, 0x07060504, 0x03020100),  // 0111
  _mm_set_epi32(0x80808080, 0x80808080, 0x80808080, 0x0f0e0d0c),  // 1000
  _mm_set_epi32(0x80808080, 0x80808080, 0x0f0e0d0c, 0x03020100),  // 1001
  _mm_set_epi32(0x80808080, 0x80808080, 0x0f0e0d0c, 0x07060504),  // 1010
  _mm_set_epi32(0x80808080, 0x0f0e0d0c, 0x07060504, 0x03020100),  // 1011
  _mm_set_epi32(0x80808080, 0x80808080, 0x0f0e0d0c, 0x0b0a0908),  // 1100
  _mm_set_epi32(0x80808080, 0x0f0e0d0c, 0x0b0a0908, 0x03020100),  // 1101
  _mm_set_epi32(0x80808080, 0x0f0e0d0c, 0x0b0a0908, 0x07060504),  // 1110
  _mm_set_epi32(0x0f0e0d0c, 0x0b0a0908, 0x07060504, 0x03020100),  // 1111
};

}  // namespace bce

#endif  // INT32x4_SSE4_H