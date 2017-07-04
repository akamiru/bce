#ifndef INT32x1_CPP_H
#define INT32x1_CPP_H

// C++ headers
#include <algorithm>
#include <cstdint>

namespace bce {

class int32x1_cpp {
 public:
    static const std::size_t size = 1;  // fits 1 int
    static constexpr std::size_t has_vector_addressing = 1;
    typedef std::size_t permute_mask_type;
    typedef bool pack_mask_type;

    struct mask_type {
        friend class int32x1_cpp;

     public:
        // Constructor
        explicit mask_type(bool value) : value_(value) {}

        // Integer functions
        inline std::size_t as_int() const {
          return static_cast<std::size_t>(this->value_);
        }

        inline std::size_t popcnt() const {
          return this->as_int();
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
          return mask_type{this->value_ && rhs.value_};
        }

        inline mask_type operator|(const mask_type& rhs) const {
          return mask_type{this->value_ || rhs.value_};
        }

        inline mask_type operator^(const mask_type& rhs) const {
          return mask_type{this->value_ != rhs.value_};
        }

     private:
        bool value_;
    };

    // Constructors
    explicit int32x1_cpp(std::int32_t value) : value_(value) {}

    // Memory access
    static inline int32x1_cpp load(const std::int32_t* ptr) {
      return int32x1_cpp{*ptr};
    }

    template<class T>
    static inline int32x1_cpp loadu(const T* ptr) {
      return int32x1_cpp{*reinterpret_cast<std::int32_t*>(ptr)};
    }

    inline void store(std::int32_t* ptr) const {
      *ptr = this->value_;
    }

    template<class T>
    inline void storeu(T* ptr) const {
      *reinterpret_cast<std::int32_t*>(ptr) = this->value_;
    }

    // Conditional selection
    static inline int32x1_cpp blend(
      const int32x1_cpp& lhs,
      const int32x1_cpp& rhs,
      const int32x1_cpp::mask_type& mask
    ) {
      return int32x1_cpp{mask.value_ ? rhs.value_ : lhs.value_};
    }

    // Extrema
    static inline int32x1_cpp min(const int32x1_cpp& lhs, const int32x1_cpp& rhs) {
      return int32x1_cpp{std::min(lhs.value_, rhs.value_)};
    }

    static inline int32x1_cpp max(const int32x1_cpp& lhs, const int32x1_cpp& rhs) {
      return int32x1_cpp{std::max(lhs.value_, rhs.value_)};
    }

    // Permutation
    inline int32x1_cpp permute(const permute_mask_type& mask) const {
      return *this; // permute a single element?
    }

    inline int32x1_cpp shuffle(const permute_mask_type& mask) const {
      return *this; // there's no byte shuffle in c++
    }

    inline int32x1_cpp pack(const pack_mask_type& mask) const {
      return *this;
    }

    inline void pack_and_store(std::int32_t* ptr, const pack_mask_type& mask) const {
      this->pack(mask).storeu(ptr);
    }

    // Arithmetic operators
    inline int32x1_cpp operator* (const int32x1_cpp& rhs) const {
      return int32x1_cpp{this->value_ * rhs.value_};
    }

    inline int32x1_cpp operator+ (const int32x1_cpp& rhs) const {
      return int32x1_cpp{this->value_ + rhs.value_};
    }

    inline int32x1_cpp operator- (const int32x1_cpp& rhs) const {
      return int32x1_cpp{this->value_ - rhs.value_};
    }

    // Bit level operators
    inline int32x1_cpp operator& (const int32x1_cpp& rhs) const {
      return int32x1_cpp{this->value_ & rhs.value_};
    }

    inline int32x1_cpp and_not   (const int32x1_cpp& rhs) const {
      return int32x1_cpp{this->value_ &~rhs.value_};
    }

    inline int32x1_cpp operator| (const int32x1_cpp& rhs) const {
      return int32x1_cpp{this->value_ | rhs.value_};
    }

    inline int32x1_cpp operator^ (const int32x1_cpp& rhs) const {
      return int32x1_cpp{this->value_ ^ rhs.value_};
    }

    // Shift operators
    inline int32x1_cpp operator<<(int rhs) const {
      return int32x1_cpp{this->value_ << rhs};
    }

    inline int32x1_cpp operator<<(const int32x1_cpp& rhs) const {
      return int32x1_cpp{this->value_ << rhs.value_};
    }

    // performs a logic right shift !
    inline int32x1_cpp operator>>(int rhs) const {
      return int32x1_cpp{static_cast<std::int32_t>(static_cast<std::uint32_t>(this->value_) >> rhs)};
    }

    inline int32x1_cpp operator>>(const int32x1_cpp& rhs) const {
      return int32x1_cpp{static_cast<std::int32_t>(static_cast<std::uint32_t>(this->value_) >> rhs.value_)};
    }

    // Comparsion operators
    inline int32x1_cpp::mask_type operator< (const int32x1_cpp& rhs) const {
      return int32x1_cpp::mask_type{this->value_ < rhs.value_};
    }

    inline int32x1_cpp::mask_type operator> (const int32x1_cpp& rhs) const {
      return int32x1_cpp::mask_type{this->value_ > rhs.value_};
    }

    inline int32x1_cpp::mask_type operator==(const int32x1_cpp& rhs) const {
      return int32x1_cpp::mask_type{this->value_ == rhs.value_};
    }

    inline int32x1_cpp::mask_type conflict_mask() const {
      return int32x1_cpp::mask_type{true};
    }

    // access to the underlying type
    inline std::int32_t get_native() {
      return this->value_;
    }

 private:
    std::int32_t value_;
};

}  // namespace bce

#endif  // INT32x1_CPP_H