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

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cinttypes>
#include <cmath>

#include <chrono>
#include <fstream>

#include <algorithm>
#include <tuple>
#include <atomic>

#include <array>
#include <vector>
#include <unordered_map>

#include "divsufsort.h"

#if defined(_MSC_VER)
#include <intrin.h>

inline constexpr const bool bce_unlikely(const bool value) {
  return value;
}

inline uint64_t bce_clz(uint64_t val) {
  unsigned long r = 0;
#if defined(_M_AMD64) || defined(_M_X64) || defined(_M_ARM)
  if (_BitScanReverse64(&r, val)) {
    return 63 - r;
  }
#else
  if (_BitScanReverse(&r, static_cast<uint32_t>(val >> 32))) {
    return 31 - r;
  }
  if (_BitScanReverse(&r, static_cast<uint32_t>(val))) {
    return 63 - r;
  }
#endif
  return 64;
}

inline uint64_t bce_ctz(uint64_t val) {
  unsigned long r = 0;
#if defined(_M_AMD64) || defined(_M_X64) || defined(_M_ARM)
  if (_BitScanForward64(&r, val)) {
    return r;
  }
#else
  if (_BitScanForward(&r, static_cast<uint32_t>(val))) {
    return r;
  }
  if (_BitScanForward(&r, static_cast<uint32_t>(val >> 32))) {
    return 32 + r;
  }
#endif
  return 64;
}

inline uint64_t bce_clo(uint64_t val) {
  return bce_clz(~val);
}

inline uint64_t bce_cto(uint64_t val) {
  return bce_ctz(~val);
}

inline uint64_t bce_cnt(uint64_t val) {
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

inline constexpr const bool bce_unlikely(const bool value) {
  return __builtin_expect(value, 0);
}

inline constexpr uint64_t bce_clz(uint64_t val) {
  return sizeof(unsigned long) == 8 ? __builtin_clzl(val) :  __builtin_clzll(val);
}

inline constexpr uint64_t bce_ctz(uint64_t val) {
  return sizeof(unsigned long) == 8 ? __builtin_ctzl(val) : __builtin_ctzll(val);
}

inline constexpr uint64_t bce_clo(uint64_t val) {
  return sizeof(unsigned long) == 8 ? __builtin_clzl(~val) : __builtin_clzll(~val);
}

inline constexpr uint64_t bce_cto(uint64_t val) {
  return sizeof(unsigned long) == 8 ? __builtin_ctzl(~val) : __builtin_ctzll(~val);
}

inline constexpr uint64_t bce_cnt(uint64_t val) {
  return sizeof(unsigned long) == 8 ? __builtin_popcountl(val) : __builtin_popcountll(val);
}
#endif

/************************
 *  The Rank Dictionary *
 ************************/

class Rank {
 public:
    Rank() : rank_{} {}

    Rank(uint32_t n) : rank_{} {
      rank_.insert(rank_.begin(), n / 32 + 1, 0);
    }

    void build() {
      uint32_t rank = 0;
      for (std::size_t i = 0; i < rank_.size(); ++i) {
        auto b = rank_[i];
        rank_[i] = (b << 32) | rank;
        rank += bce_cnt(b);
      }
    }

    template<int S>
    inline uint32_t get(uint32_t index) const  {
      auto rank = rank_[index / 32] & (-1llu >> (32 - index % 32));
      return rank + bce_cnt(rank >> 32);
    }

    void set(uint32_t _x, uint32_t value) {
      uint64_t n = value - get<1>(_x);
      if (n == 0) return;
      assert (n < (1llu << 32));

      uint64_t i = _x / 32llu;
      uint64_t o = _x % 32llu;

      // bits      rank
      // [76543210][00000000]
      uint64_t b = rank_[i];
      uint32_t r = b;

      if (r + o + 32 < n) {
        b += n - o - r;
        n = o;
      }

      uint64_t m0 = -1llu << (32 + o);
      uint64_t m1 =  0 + bce_ctz(((b & m0) >> 32) | +(1llu << 31)  );  // get bits
      uint64_t m2 = 64 - bce_clo( (b | m0)     /* & ~(1llu << 31)*/);  // put bits - probably decomment for block sizes >= (1 << 31)

      m1 = ((1llu << (m1 + n)) - (1llu <<      m1 )) << 32llu;
      m2 = ((1llu <<      m2 ) - (1llu << (m2 - n)));

      b += bce_cnt(static_cast<uint32_t>(m2));
      b &= ~m1;
      b |= (m2 >> 32llu) << 32llu;

      rank_[i] = b;

      assert(value == get<1>(_x));
    }

    void finalize() {
      for (uint32_t i = 0; i < rank_.size() - 1; ++i) {
        auto cur = static_cast<uint32_t>(rank_[i]) + bce_cnt(rank_[i] >> 32);
        auto nxt = static_cast<uint32_t>(rank_[i + 1]);

        rank_[i] |= static_cast<uint64_t>(nxt - cur) << 63;
      }
    }

    inline uint32_t bit(uint32_t offset) const {
      return (rank_[offset / 32] >> (offset % 32 + 32)) & 1;
    }

    inline uint32_t set_bit(uint32_t offset, uint64_t bit) {
      return rank_[offset / 32] |= bit << (offset % 32);
    }

    inline uint32_t size() {
      return rank_.size();
    }

    void clear() {
      rank_.clear();
      rank_.shrink_to_fit();
    }
 private:
    std::vector<uint64_t> rank_;
};

template<>
uint32_t Rank::get<0>(uint32_t index) const  {
  return index - get<1>(index);
}

/*********************************
 *  The pArray                   *
 * implements Elias-Gamma Coding *
 *********************************/

class pArray {
 public:
    pArray() : data_{}, cur_(0), pos_(64) {}

    pArray(uint32_t n) : data_{}, cur_(0), pos_(64) {
      data_.reserve(n);
    }

    inline void push_back(uint32_t a) {
      assert(a > 0);
      auto n = 127 - 2 * bce_clz(a);

      if (pos_ <= n) {
        data_.push_back(cur_);
        cur_ = 0;
        pos_ = 64;
      }

      cur_ |= static_cast<uint64_t>(a) << (pos_ - n);
      pos_ -= n;
    }

    inline void push_back(uint32_t a, uint32_t b) {
      assert(a > 0 && b > 0);
      auto m = 127 - 2 * bce_clz(a);
      auto n = 127 - 2 * bce_clz(b);

      if (pos_ <= m + n) {
        push_back(a);
        push_back(b);
        return;
      }

      cur_ |= (static_cast<uint64_t>(a) << (pos_ - m))
            | (static_cast<uint64_t>(b) << (pos_ - m - n));
      pos_ -= m + n;
    }

    inline void push_back(uint32_t a, uint32_t b, uint32_t c) {
      assert(a > 0 && b > 0 && c > 0);
      auto m = 127 - 2 * bce_clz(a);
      auto n = 127 - 2 * bce_clz(b);
      auto o = 127 - 2 * bce_clz(c);

      if (pos_ <= m + n + o) {
        push_back(a, b);
        push_back(c);
        return;
      }

      cur_ |= (static_cast<uint64_t>(a) << (pos_ - m))
            | (static_cast<uint64_t>(b) << (pos_ - m - n))
            | (static_cast<uint64_t>(c) << (pos_ - m - n - o));
      pos_ -= m + n + o;
    }

    uint32_t size() const  {
      return data_.size() * sizeof(decltype(data_)::value_type);
    }

    struct iterator {
      using vit = std::vector<uint64_t>::const_iterator;

      explicit iterator(vit it) : it_(std::move(it)), cur_(0) {}

      uint64_t operator*() const  {
        auto val = *it_ << cur_;
        return val >> (63 - 2 * bce_clz(val));
      }

      iterator& operator++() {
        auto val = *it_;
        cur_ += 2 * bce_clz(val << cur_) + 1;

        if (!(val << cur_)) {
          it_++;
          cur_ = 0;
        }

        return *this;
      }

      iterator operator++(int) {
        auto it(*this);
        ++(*this);
        return it;
      }

      bool operator==(const iterator& it) const  {
        return std::tie(it_, cur_) == std::tie(it.it_, it.cur_);
      }

      bool operator!=(const iterator& it) const  {
        return !(*this == it);
      }

    public:
      vit it_;
      uint64_t cur_;
    };

    iterator begin() {
      if (cur_) {
        data_.push_back(cur_);
        cur_ = 0;
        pos_ = 64;
      }
      return iterator(data_.begin());
    };

    iterator end() const  {
      return iterator(data_.end());
    };

    void clear() {
      cur_ = 0;
      pos_ = 64;

      data_.clear();
      data_.shrink_to_fit();
    }

    bool empty() {
      return pos_ == 64 && data_.empty();
    }

 private:
    std::vector<uint64_t> data_;
    uint64_t cur_;
    uint64_t pos_;
};

/*********
 * Coder *
 *********/

template<class C>
struct VCoder {
  void setv(uint32_t s) {
    while (s) {
      reinterpret_cast<C*>(this)->C::set(s & 1, 3);
      s >>= 1;
    }
    reinterpret_cast<C*>(this)->C::set(2, 3);
  }

  uint32_t getv() {
    uint32_t s = 0;
    for (int i = 0, j = reinterpret_cast<C*>(this)->C::get(3); i < 31 && j != 2; ++i, j = reinterpret_cast<C*>(this)->C::get(3))
      s |= j << i;
    return s;
  }
};

class UniformCoder : public VCoder<UniformCoder> {
 public:
    using value_type = std::vector<uint16_t>;
    // maximum value for range that will be adaptivly encoded
    static constexpr const int max = 0;

    UniformCoder(int i) : l_(0), h_(-1llu) {}

    explicit UniformCoder(int i, UniformCoder::value_type&& data):
      l_(0), h_(-1llu), m_(0), o_(sizeof(m_) / sizeof(data_[0])), data_(std::move(data)) {
      for (uint32_t i = 0; i < data_.size() && i < o_; i++)
        m_ = (m_ << 16) + data_[i];

      if (data_.size() < o_)
        m_ <<= 16 * (o_ - data_.size());
    }

    void set(uint32_t s, uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) {
      set(s, k);  // encode all numbers using uniform distribution
    }

    void set(uint32_t s, uint32_t k) {
      assert(s < k);

      if (h_ - l_ < k) {
        for (int i = 0; i < 4; ++i)
          data_.push_back(l_ >> (48 - 16 * i));
        l_ = 0;
        h_ = -1llu;
      }

      uint64_t step = (h_ - l_) / k;
      l_ += step * s;
      h_  = step + l_ - 1;

      shift_out();
    }

    uint32_t get(uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) {
      return get(k);
    }

    uint32_t get(uint32_t k) {
      if (h_ - l_ < k) {
        for (int i = 0; i < 4; ++i)
          m_ = (m_ << 16) + ((o_ < data_.size()) ? data_[o_++] : 0);
        l_ = 0;
        h_ = -1llu;
      }

      uint64_t step = (h_ - l_) / k;
      uint32_t s    = (m_ - l_) / step;

      l_ += step * s;
      h_  = step + l_ - 1;

      shift_in();
      return s;
    }

    void flush() {
      shift_out();

      uint32_t bits = bce_clz(l_ ^ h_) + 1;
      data_.push_back((h_ >> (64 - bits)) << (16 - bits));
    }

    inline const UniformCoder::value_type& data() const  {
      return data_;
    }

    void clear() {
      UniformCoder::value_type().swap(data_);
      std::vector<uint8_t>().swap(stat_);
    }

    static void load_config(std::string file) {}

 private:
    uint64_t l_;
    uint64_t h_;
    uint64_t m_;
    uint32_t o_;

    UniformCoder::value_type data_;
    std::vector<uint8_t> stat_;

    inline void shift_out() {
      while (!((h_ ^ l_ ) >> 48)) {
        data_.push_back(h_ >> 48);
        l_ = (l_ << 16) + 0x0000;
        h_ = (h_ << 16) + 0xFFFF;
      }
    }

    inline void shift_in() {
      while (!((h_ ^ l_ ) >> 48)) {
        m_ = (m_ << 16) + ((o_ < data_.size()) ? data_[o_++] : 0);
        l_ = (l_ << 16) + 0x0000;
        h_ = (h_ << 16) + 0xFFFF;
      }
    }
};

template<int L>
class AdaptiveCoder : public VCoder<AdaptiveCoder<L>> {
 public:
    using value_type = std::vector<uint16_t>;
    // maximum value for range that will be adaptivly encoded
    static constexpr const int max = L;

    AdaptiveCoder(int i) : l_(0), h_(-1llu) {
      init(1, i);
    }

    explicit AdaptiveCoder(int i, typename AdaptiveCoder::value_type&& data):
      l_(0), h_(-1llu), m_(0), o_(sizeof(m_) / sizeof(data_[0])), data_(std::move(data)) {
      for (uint32_t i = 0; i < data_.size() && i < o_; i++)
        m_ = (m_ << 16) + data_[i];

      if (data_.size() < o_)
        m_ <<= 16 * (o_ - data_.size());

      init(0, i);
    }

    void set(uint32_t s, uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) {
      if (k > AdaptiveCoder::max) {
        set(s & 1, 2);
        return set(s >> 1, (k + (~s & 1)) >> 1, c1, c2, cs);
      }

      auto* ctx = get_context(k, c1, c2, cs);

      uint32_t l = 0;
      for (uint32_t i = 0; i < s; ++i) l += ctx[i];
      uint32_t n = l + s;
      for (uint32_t i = s; i < k; ++i) l += ctx[i];
      l += k;

      if (bce_unlikely(h_ - l_ < l)) {
        for (int i = 0; i < 4; ++i)
          data_.push_back(l_ >> (48 - 16 * i));
        l_ = 0;
        h_ = -1llu;
      }

      uint64_t step = (h_ - l_) / l;
      l_ += step * n;
      h_ = l_ + step * (ctx[s] + 1) - 1;

      if (++ctx[s] == 0xFF)
        for (uint32_t i = 0; i < k; ++i)
          ctx[i] >>= 1;

      shift_out();
    }

    void set(uint32_t s, uint32_t k) {
      assert(s < k);

      if (bce_unlikely(h_ - l_ < k)) {
        for (int i = 0; i < 4; ++i)
          data_.push_back(l_ >> (48 - 16 * i));
        l_ = 0;
        h_ = -1llu;
      }

      uint64_t step = (h_ - l_) / k;
      l_ += step * s;
      h_  = step + l_ - 1;

      shift_out();
    }

    uint32_t get(uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) {
      if (k > AdaptiveCoder::max) {
        auto s = get(2);
        return (get((k + (~s & 1)) >> 1, c1, c2, cs) << 1) | s;
      }

      auto* ctx = get_context(k, c1, c2, cs);

      uint32_t l = k;
      for (uint32_t i = 0; i < k; ++i) l += ctx[i];

      if (bce_unlikely(h_ - l_ < l)) {
        for (int i = 0; i < 4; ++i)
          m_ = (m_ << 16) + ((o_ < data_.size()) ? data_[o_++] : 0);
        l_ = 0;
        h_ = -1llu;
      }

      uint64_t step = (h_ - l_) / l;

      h_ = l_ - 1;
      uint32_t s = -1u;
      do {
        s++;
        l_ = h_ + 1;
        h_ += step * (ctx[s] + 1);
      } while (h_ < m_);

      if (++ctx[s] == 0xFF)
        for (uint32_t i = 0; i < k; ++i)
          ctx[i] >>= 1;

      shift_in();

      return s;
    }

    uint32_t get(uint32_t k) {
      if (bce_unlikely(h_ - l_ < k)) {
        for (int i = 0; i < 4; ++i)
          m_ = (m_ << 16) + ((o_ < data_.size()) ? data_[o_++] : 0);
        l_ = 0;
        h_ = -1llu;
      }

      uint64_t step = (h_ - l_) / k;
      uint32_t s    = (m_ - l_) / step;

      l_ += step * s;
      h_  = step + l_ - 1;

      shift_in();
      return s;
    }

    void flush() {
      shift_out();

      uint32_t bits = bce_clz(l_ ^ h_) + 1;
      data_.push_back((h_ >> (64 - bits)) << (16 - bits));
    }

    inline const value_type& data() const  {
      return data_;
    }

    void clear() {
      value_type().swap(data_);
      std::vector<uint8_t>().swap(stat_);
    }

    static void load_config(std::string file) {
      std::ifstream archive(file, std::ios::binary | std::ios::ate);
      std::size_t size = archive.tellg();
      if (size != (AdaptiveCoder::max + 1) * 9) {
        printf("Config not found or wrong size.\n");
        return;
      }

      archive.seekg(0, std::ios::beg);

      if (!archive.read(reinterpret_cast<char*>(init_.data()), size)) {
        printf("Could not read Config.\n");
        return;
      }
      archive.close();
    }

 private:
    uint64_t l_;
    uint64_t h_;
    uint64_t m_;
    uint32_t o_;

    typename AdaptiveCoder::value_type data_;
    std::array<uint32_t, L + 1> off_;
    std::vector<uint8_t> stat_;

    static std::array<std::array<uint8_t, L + 1>, 9> init_;

    inline void shift_out() {
      while (!((h_ ^ l_ ) >> 48)) {
        data_.push_back(h_ >> 48);
        l_ = (l_ << 16) + 0x0000;
        h_ = (h_ << 16) + 0xFFFF;
      }
    }

    inline void shift_in() {
      while (!((h_ ^ l_ ) >> 48)) {
        m_ = (m_ << 16) + ((o_ < data_.size()) ? data_[o_++] : 0);
        l_ = (l_ << 16) + 0x0000;
        h_ = (h_ << 16) + 0xFFFF;
      }
    }

    inline auto get_context(uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) -> decltype(this->stat_.data()) {
      auto off = off_[k];
      auto bits = off >> 24;
      auto ctx = (((c1 << bits) / cs) << bits) | ((c2 << bits) / cs);

      return this->stat_.data() + (off & 0x00FFFFFF) + ctx * k;
    }

    void init(int mode, int i) {
      std::array<uint8_t, AdaptiveCoder::max + 1> bits;

      if (mode) {
        if (0 > i || i > 7) i = 8;
        bits = init_[i];
        auto last = 0;
        for (auto& bit : bits) {
          set(bit != last, 2);
          if (bit != last)
            set(bit, 6);
          last = bit;
        }
      } else {
        auto last = 0;
        for (auto& bit : bits) {
          bit = get(2) ? get(6) : last;
          last = bit;
        }
      }

      uint32_t start = 0;
      for (int i = 2; i < AdaptiveCoder::max + 1; ++i) {
        off_[i] = start | (bits[i] << 24);
        start += i << bits[i] * 2;
      }
      stat_.insert(stat_.begin(), start, 0);

#ifdef M_TIME
      printf("Coder[%i]: %u B\n", i, start);
#endif
    }
};

template<int L>
std::array<std::array<uint8_t, L + 1>, 9> AdaptiveCoder<L>::init_ = {
  0,0,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,
  0,0,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,
  0,0,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,0,
  0,0,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,0,
  0,0,5,5,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,0,
  0,0,5,5,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,0,
  0,0,5,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,0,
  0,0,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,0,
  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
};

template<int L>
class ScanCoder : public VCoder<ScanCoder<L>> {
 public:
    using value_type = std::vector<uint16_t>;
    // maximum value for range that will be adaptivly encoded
    static constexpr const int max = L;

    ScanCoder(int i): z_(0), i_{i < 0 || i > 7 ? 8 : i} {}

    explicit ScanCoder(int i, typename ScanCoder::value_type&& data) = delete; // not a decoder

    void set(uint32_t s, uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) {
      if (k > ScanCoder::max) {
        z_ += log(2);
        return set(s >> 1, (k >> 1) + ((~s) & 1), c1, c2, cs);
      }

      stat_[k][(((c2 << 8) / cs) << 16) | ((c1 << 8) / cs)].push_back(s);
    }

    void set(uint32_t s, uint32_t k) {}

    uint32_t get(uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) { return 0; }
    uint32_t get(uint32_t k) { return 0; }

    void flush() {
      std::vector<uint16_t> s;

      for (uint32_t k = 2; k < ScanCoder::max; ++k) {
        double z_min = 0;
        for (auto& pair : stat_[k])
          z_min += log(k) * pair.second.size();

        // clustering hash
        for (uint32_t j = 0; j <= 5; ++j) {
          s.clear();
          s.insert(s.begin(), k << (2 * j), 0);

          double z = 0;
          for (auto& pair : stat_[k]) {
            auto c = pair.first;
            uint16_t c1 = c >>  0;
            uint16_t c2 = c >> 16;

            c1 >>= 8 - j;
            c2 >>= 8 - j;

            c = (c1 << j) | c2;

            auto* ctx = &s[c * k];

            for (auto& s : pair.second) {
              uint32_t l = k;
              for (uint32_t i = 0; i < k; ++i) {
                l += ctx[i];
              }

              z += log(static_cast<double>(l) / (1 + ctx[s]));

              // update
              if (++ctx[s] == 0xFF)
                for (uint32_t i = 0; i < k; ++i)
                  ctx[i] >>= 1;
            }
          }

          if (z < z_min) {
            z_min = z;
            init_[i_][k] = j;
          }
        }
        z_ += z_min;
      }
      printf("Result size: %.1f B\n", z_ / log(256));
    }

    inline const typename ScanCoder::value_type& data() const  {
      return data_;
    }

    void clear() {}

    static void load_config(std::string file) {}

    static void save_config(std::string file) {
      std::ofstream f(file, std::ios::binary | std::ios::trunc);
      f.write(reinterpret_cast<const char*>(init_.data()), (L + 1) * 9);

#ifdef DUMP_CONFIG
      for (int i = 0; i < 9; ++i) {
        printf("  0");
        for (int j = 1; j < L + 1; ++j) {
          printf(",%u", init_[i][j]);
        }
        printf("\n");
      }
#endif
    }
 private:
    std::array<std::unordered_map<uint32_t, std::vector<uint8_t>>, ScanCoder::max + 1> stat_;
    typename ScanCoder::value_type data_;
    double z_;
    int i_;

    static std::array<std::array<uint8_t, L + 1>, 9> init_;
};

template<int L>
std::array<std::array<uint8_t, L + 1>, 9> ScanCoder<L>::init_;

/**************
 * File class *
 **************/

class File {
 public:
    explicit File(const std::string& file_name) : offset_(0) {
      std::ifstream file(file_name, std::ios::binary | std::ios::ate);

      size_ = file.tellg();
      if (size_ == -1lu) {
        status_ = 1;
        return;
      }

      file.seekg(0, std::ios::beg);

      map_.resize(size_);

      status_ = !file.read(reinterpret_cast<char*>(map_.data()), size_);
    }

    uintmax_t rotate() {
      auto mod = [this](auto i) {
        while (bce_unlikely(i >= size_)) i -= size_;
        return i;
      };

#ifdef M_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif

      // Credits: MaskRay [github]
      uint32_t i = 0, j = 1, k;
      while (j < size_) {
        for (k = 0; map_[mod(i+k)] == map_[mod(j+k)] && k < size_ - 1; k++);

        if (map_[mod(i+k)] <= map_[mod(j+k)]) {
          j += k+1;
        } else {
          i += k+1;
          if (i < j)
            i = j++;
          else
            j = i+1;
        }
      }

      std::rotate(map_.begin(), map_.begin() + i + 1, map_.end());

#ifdef M_TIME
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;

      printf("Rotate: %f s (index: %u)\n", duration.count(), i);
#endif

      return offset_ = i;
    }

    void bwt() {
#ifdef M_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif

      uint32_t i = divbwt(map_.data(), map_.data(), 0, size_ - 1);
      std::rotate(map_.begin() + i, map_.end() - 1, map_.end());

#ifdef M_TIME
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;

      printf("BWT: %f s\n", duration.count());
#endif
    }

    unsigned char operator[](uintmax_t index) const  {
      return map_[index];
    };

    inline uintmax_t size() const  { return size_; }
    inline uintmax_t offset() const  { return offset_; }
    inline uintmax_t status() const  { return status_; }

 protected:
    std::vector<unsigned char> map_;
    std::size_t size_;
    uint32_t offset_;
    uint32_t status_;
};

/**********************************
 *           Rank File            *
 * BCE's tool for querying a file *
 **********************************/

struct RankFile : public File {
  RankFile(const std::string& file) : File(file), ranks {} {
    if (status_ == 0) {
      rotate();
      bwt();

      for (int i = 0; i < 8; ++i)
        ranks[i] = Rank(size());

#ifdef M_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif
      std::array<uint32_t, 256> C;
      C.fill(0);

      for (std::size_t i = 0; i < size_; ++i)
        C[map_[i] | 0x80]++;

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

      for (std::size_t i = 0; i < size_; ++i) {
        auto chr = map_[i];
        for (int j = 0; j < 8; ++j) {
          auto c = (chr & ((1 << j) - 1)) | (1 << j);
          ranks[j].set_bit(C[c]++, (chr >> j) & 1);
        }
      }

      for (int i = 0; i < 8; ++i) ranks[i].build();
      map_.clear();
      map_.shrink_to_fit();

#ifdef M_TIME
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;

      printf("Rank: %f s\n", duration.count());
#endif
    }
  }

  std::array<Rank, 8> ranks;
};

/***********************
 *   Unbwt Policies    *
 ***********************/

namespace unbwt {

class noop {
 public:
    std::vector<uint8_t> unbwt(std::array<Rank, 8>& ranks, uint32_t offset, uint32_t n) { return std::vector<uint8_t>(); }
};

class bitwise {
 public:
    std::vector<uint8_t> unbwt(std::array<Rank, 8>& ranks, uint32_t offset, uint32_t n) {
#ifdef M_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif

      std::vector<uint8_t> out(n);

      std::array<uint32_t, 8> C = {
        ranks[0].get<0>(n),
        ranks[1].get<0>(n),
        ranks[2].get<0>(n),
        ranks[3].get<0>(n),
        ranks[4].get<0>(n),
        ranks[5].get<0>(n),
        ranks[6].get<0>(n),
        ranks[7].get<0>(n)
      };

      // unbwt
      uintmax_t s = 0;
      for (uintmax_t i = n - 1; i != -1lu; --i) {
        uint8_t chr = 0;
        for (int j = 0; j < 8; j++) {
          auto bit = ranks[j].bit(s);

          chr |= bit << j;

          s = (bit ? C[j] + ranks[j].get<1>(s) : ranks[j].get<0>(s));
        }
        out[(i + offset) % n] = chr;
      }

#ifdef M_TIME
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;
      printf("Bitwise UnBWT: %f s (index: %u)\n",  duration.count(), offset);
#endif

      return out;
    }
};

class bytewise {
 public:
    std::vector<uint8_t> unbwt(std::array<Rank, 8>& ranks, uint32_t offset, uint32_t n) {
#ifdef M_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif

      std::vector<uint8_t> out(n);
      {
        uint32_t s = (((n / 8) >> 12) + 1) << 12;

        std::array<uint32_t, 8> C = {
          ranks[0].get<0>(n),
          ranks[1].get<0>(n),
          ranks[2].get<0>(n),
          ranks[3].get<0>(n),
          ranks[4].get<0>(n),
          ranks[5].get<0>(n),
          ranks[6].get<0>(n),
          ranks[7].get<0>(n)
        };

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int64_t a = 0; a < n; a += s) {
          std::array<uint32_t, 256> D;
          D.fill(0);
          D[1] = static_cast<uint32_t>(a);

          for (int i = 0; i < 7; ++i) {
            for (int j = 0; j < (1 << i); ++j) {
              auto e = D[(1 << i) | j];
              D[(2 << i) | j] =        ranks[i].get<0>(e);
              D[(3 << i) | j] = C[i] + ranks[i].get<1>(e);
            }
          }

          for (uint32_t i = static_cast<uint32_t>(a); i < std::min(n, static_cast<uint32_t>(a) + s); ++i) {
            auto chr = 0;
            for (int j = 0; j < 8; ++j)
              chr |= ranks[j].bit(D[(1 << j) | chr]++) << j;
            out[i] = chr;
          }
        }

        for (int i = 0; i < 8; ++i)
          ranks[i].clear();
      }

      inverse_bw_transform(out.data(), out.data(), nullptr, n, 1);

      std::rotate(out.begin(), out.end() - offset, out.end());

#ifdef M_TIME
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;
      printf("Bytewise UnBWT: %f s (index: %u)\n",  duration.count(), offset);
#endif

      return out;
    }
};

}

/**********************
 * The BCE compressor *
 **********************/

template<class policy_coder, class policy_unbwt>
class BCE : private policy_unbwt {
 public:
    using coder_type = policy_coder;
    BCE() {}

    typename coder_type::value_type encode(RankFile& file) {
#ifdef M_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif

      auto n = file.size();

      std::array<coder_type, 8> coder_ = {0, 1, 2, 3, 4, 5, 6, 7};

      std::array<uint32_t, 8> C;
      for (int i = 0; i < 8; ++i) {
        C[i] = file.ranks[(i + 7) % 8].get<0>(n);
        coder_[i].set(C[i], n + 1);
      }

      code(coder_, C, file.ranks, n, 1);

      auto size = 0u;
      for (int i = 0; i < 8; ++i) {
        coder_[i].flush();
        size += coder_[i].data().size();
      }

      // Build the header data
      coder_type main(-1);
      main.setv(n);
      main.set(file.offset(), n + 1);
      main.setv(size);
      for (int i = 0, s = size; i < 7; ++i) {
        main.set(coder_[i].data().size(), s + 1);
        s -= coder_[i].data().size();
      }
      main.flush();

      // tie the data together
      typename coder_type::value_type data;
      data.push_back(main.data().size());
      data.insert(data.end(), main.data().begin(), main.data().end());

      for (int i = 0; i < 8; ++i)
        data.insert(data.end(), coder_[i].data().begin(), coder_[i].data().end());

#ifdef M_TIME
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;

      printf("Encode: %f s\n", duration.count());
#endif

      return data;
    }

    std::vector<uint8_t> decode(typename coder_type::value_type& data) {
#ifdef M_TIME
      auto start = std::chrono::high_resolution_clock::now();
#endif
      auto data_chunk = [&data](auto a, auto b) {
        return typename coder_type::value_type(data.begin() + a, data.begin() + a + b);
      };

      // Decoding the header
      auto header_size = static_cast<uint32_t>(data[0]);
      coder_type main(-1, data_chunk(1, header_size));

      auto n      = main.getv();  // size of the decompressed file
      auto offset = main.get(n + 1);
      auto size   = main.getv();  // size of the compressed file

      std::array<uint32_t, 8> coder_offsets = {header_size + 1u};

      for (int i = 0; i < 7; ++i) {
        coder_offsets[i + 1] = coder_offsets[i] + main.get(size + 1u);
        size -= coder_offsets[i + 1] - coder_offsets[i];
      }
      main.clear();

      std::array<coder_type, 8> coder_ = {
        coder_type(0, data_chunk(coder_offsets[0], coder_offsets[1] - coder_offsets[0])),
        coder_type(1, data_chunk(coder_offsets[1], coder_offsets[2] - coder_offsets[1])),
        coder_type(2, data_chunk(coder_offsets[2], coder_offsets[3] - coder_offsets[2])),
        coder_type(3, data_chunk(coder_offsets[3], coder_offsets[4] - coder_offsets[3])),
        coder_type(4, data_chunk(coder_offsets[4], coder_offsets[5] - coder_offsets[4])),
        coder_type(5, data_chunk(coder_offsets[5], coder_offsets[6] - coder_offsets[5])),
        coder_type(6, data_chunk(coder_offsets[6], coder_offsets[7] - coder_offsets[6])),
        coder_type(7, data_chunk(coder_offsets[7], data.size()      - coder_offsets[7])),
      };
      std::vector<uint16_t>().swap(data);

      std::array<Rank, 8> ranks = {n, n, n, n, n, n, n, n};

      std::array<uint32_t, 8> C;
      for (int i = 0; i < 8; ++i) {
        C[i] =  coder_[i].get(n + 1);
        ranks[(i + 7) % 8].set(n, n-C[i]);
      }

      std::array<std::array<pArray, 4>, 8> Q;
      for (int i = 0; i < 8; ++i)
        if (C[i] && n - C[i])
          Q[i][0].push_back(1, C[i], n - C[i]);

      code(coder_, C, ranks, n, 0);

      for (int i = 0; i < 8; ++i) {
        ranks[i].finalize();
        coder_[i].clear();
      }

#ifdef M_TIME
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;

      printf("Decode: %f s\n", duration.count());
#endif

      return this->unbwt(ranks, offset, n);
    }

 public:
    void code(std::array<coder_type, 8>& coder_, const std::array<uint32_t, 8>& C, std::array<Rank, 8>& ranks, const uint32_t n, const uint32_t mode) {
      std::array<std::array<pArray, 4>, 8> Q;
      for (int i = 0; i < 8; ++i)
        if (C[i] && n - C[i])
          Q[i][0].push_back(1, C[i], n - C[i]);

      uint64_t prev_state = 0;
      std::atomic<uint64_t> state(0);
      bool again = false;

      do {
        again = false;

#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int i = 0; i < 8; ++i) {
          uint32_t local_state = 0;
          std::array<uint32_t, 2> offset = {0, 0};

          for (int j = 0; j < 2; ++j) {
            auto cur = Q[i][j].begin();
            auto end = Q[i][j].end();
            auto s = C[i] * j;

            while (cur != end) {
              local_state++;

              s += *cur++ - 1;
              auto s1 = ranks[i].get<1>(s);

              auto _x0 = *cur++;
              auto _x1 = *cur++;
              auto _x = _x0 + _x1;

              auto _1x = ranks[i].get<1>(s + _x) - s1;
              auto s0 = s - s1;

              if (!_1x) {
                Q[i][2].push_back(s0 - offset[0] + 1, _x0, _x1);
                offset[0] = s0;
                if (!mode) ranks[i].set(s + _x0, s1 + 0);
                continue;
              }

              auto _0x = _x - _1x;
              if (!_0x) {
                Q[i][3].push_back(s1 - offset[1] + 1, _x0, _x1);
                offset[1] = s1;
                if (!mode) ranks[i].set(s + _x0, s1 + _x0);
                continue;
              }

              // Min Max
              uint32_t min = _x0 - _1x;
              uint32_t max = _1x - _x1;
              min &= ~(static_cast<int32_t>(min) >> 31);
              max &= ~(static_cast<int32_t>(max) >> 31);
              max = _x0 - max;

              // Encode/Decode
              uint32_t _0x0 = min;

              if (max != min) {
                if (mode) {
                  _0x0 = ranks[i].get<0>(s + _x0) - s0;
                  coder_[i].set(_0x0 - min, max - min + 1, _0x, _x1, _x);
                } else {
                  _0x0 = min + coder_[i].get(max - min + 1, _0x, _x1, _x);
                }
                assert(min <= _0x0 && _0x0 <= max);
              }

#if 0 // different checks for debugging
              assert(_0x <= _x);
              assert(_1x <= _x);

              assert(_x0 >= 0);
              assert(_x1 >= 0);
              assert(_0x >= 0);
              assert(_1x >= 0);
              assert(_x0 + _x1 == _0x + _1x && _0x + _1x == _x);

              assert(_0x0 <= _x0);
              assert(_0x0 <= _0x);

              assert(_1x0 <= _x0);
              assert(_1x0 <= _1x);

              assert(_0x1 <= _x1);
              assert(_0x1 <= _0x);

              assert(_1x1 <= _x1);
              assert(_1x1 <= _1x);

              assert(_1x0 + _0x0 == _x0);
              assert(_1x1 + _0x1 == _x1);
              assert(_0x0 + _0x1 == _0x);
              assert(_1x0 + _1x1 == _1x);
#endif

              auto _0x1 = _0x - _0x0;
              if (_0x0 && _0x1) {
                Q[i][2].push_back(s0 - offset[0] + 1, _0x0, _0x1);
                offset[0] = s0;
              }

              auto _1x1 = _x1 - _0x1;
              auto _1x0 = _1x - _1x1;
              if (_1x0 && _1x1) {
                Q[i][3].push_back(s1 - offset[1] + 1, _1x0, _1x1);
                offset[1] = s1;
              }

              if (!mode) ranks[i].set(s + _x0, s1 + _1x0);
            }
          }
          state += local_state;
          auto cur_state = state.load(std::memory_order_relaxed) * 10000 / 8 / n;
          if (cur_state > prev_state) {
            printf("Coded: %" PRIu64 ".%02" PRIu64 " %%\r", cur_state / 100, cur_state % 100);
            prev_state = cur_state;
          }
        }

        for (int i = 0; i < 8; ++i) {
          std::swap(Q[(i + 1) % 8][0], Q[i][2]);
          std::swap(Q[(i + 1) % 8][1], Q[i][3]);

          Q[i][2].clear();
          Q[i][3].clear();

          if (!Q[(i + 1) % 8][0].empty() || !Q[(i + 1) % 8][1].empty())
            again = true;
        }
      } while (again);
      printf("                    \r");
    }
};

int main(int argc, char** argv) {
  printf("BCE v0.4 Release\n");
  printf("Copyright (C) 2016  Christoph Diegelmann\n");
  printf("This is free software under GNU Lesser General Public License. See <http://www.gnu.org/licenses/lgpl>\n\n");

  constexpr const int max = 31;
  using coder_type = AdaptiveCoder<max>;

  if (argc == 4 && argv[1][0] == '-' && argv[1][1] == 's') {
    auto start = std::chrono::high_resolution_clock::now();
    // Compress
    BCE<ScanCoder<max>, unbwt::noop> bce;

    RankFile file {std::string(argv[3])};
    if (file.status()) {
      printf("Error loading file\n");
      return -1;
    }

    bce.encode(file);

    decltype(bce)::coder_type::save_config(argv[2]);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("Scanned %" SCNuMAX " B in %.1f s\n",
           file.size(), duration.count());
  } else if ((argc == 4 || argc == 5) && argv[1][0] == '-' && argv[1][1] == 'c') {
    auto start = std::chrono::high_resolution_clock::now();
    // Compress
    BCE<coder_type, unbwt::bytewise> bce;

    if (argc == 5)
      decltype(bce)::coder_type::load_config(argv[4]);

    RankFile file {std::string(argv[3])};
    if (file.status()) {
      printf("Error loading file\n");
      return -1;
    }

    auto data = bce.encode(file);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    printf("Compressed from %" SCNuMAX" B -> %zu B in %.1f s\n",
           file.size(), data.size() * sizeof(decltype(data)::value_type), duration.count());

    std::ofstream archive(std::string(argv[2]), std::ios::binary | std::ios::trunc);

    archive.write(reinterpret_cast<const char*>(data.data()),
                  data.size() * sizeof(decltype(data)::value_type));
  } else if (argc == 4 && argv[1][0] == '-' && argv[1][1] == 'd') {
    // Decompress
    auto decompress = [&argv](auto bce) {
      auto start = std::chrono::high_resolution_clock::now();

      std::ifstream archive(std::string(argv[3]), std::ios::binary | std::ios::ate);
      size_t size = archive.tellg();
      if (size == -1lu) {
        printf("Archive not found.\n");
        return -1;
      }

      archive.seekg(0, std::ios::beg);

      typename decltype(bce)::coder_type::value_type archive_data;
      archive_data.resize(size / sizeof(typename decltype(bce)::coder_type::value_type::value_type));

      if (!archive.read(reinterpret_cast<char*>(archive_data.data()), size)) {
        printf("Could not read Archive.\n");
        return -2;
      }
      archive.close();

      auto data = bce.decode(archive_data);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> duration = end - start;
      printf("Decompressed from %zu B -> %zu B in %.1f s\n",
            size, data.size(), duration.count());

      std::ofstream file(std::string(argv[2]), std::ios::binary | std::ios::trunc);

      file.write(reinterpret_cast<const char*>(data.data()),
                data.size() * sizeof(typename decltype(data)::value_type));

      return 0;
    };

    if (argv[1][2] == 's') {
      BCE<coder_type, unbwt::bitwise> bce;
      return decompress(bce);
    } else {
      BCE<coder_type, unbwt::bytewise> bce;
      return decompress(bce);
    }
  } else {
    printf("Usage:\n");
    printf("  bce -c archive.bce file [config.bcc]\n");
    printf("   Compresses \"file\" to archive \"archive.bce\" [using config \"config.bcc\"]\n");
    printf("\n");
    printf("  bce -d file archive.bce\n");
    printf("   Decompresses archive \"archive.bce\" to \"file\"\n");
    printf("\n");
    printf("  bce -s config.bcc file\n");
    printf("   Scan \"file\" and generate a config file \"config.bcc\" to improve the AdaptiveCoder (uses a lot of memory)\n");
  }
}
