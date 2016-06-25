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

#ifndef BCE
#define BCE

#include <cassert>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <fstream>
#include <numeric>

#include <array>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <divsufsort.h>

#include "builtins.h"

namespace bce {
  // policies for rank dictionaries
  namespace rank {
    // standard rank dictionary takes 2 * N bits of memory to store text + dictionary
    template<class Allocator = std::allocator<uint64_t>>
    class fast {
     public:
        fast() : rank_{} {}

        fast(uint32_t n) : rank_{} {
          rank_.insert(rank_.begin(), n / 32 + 1, 0);
        }

        void build() {
          uint32_t rank = 0;
          for (auto& b : rank_) {
            b = (b << 32) | rank;
            rank += builtin::cnt(b >> 32);
          }
        }

        template<int S>
        inline uint32_t get(uint32_t index) const  {
          if (S == 0) return index - get<1>(index);
          auto rank = rank_[index / 32] & (UINT64_C(-1) >> (32 - index % 32));
          return rank + builtin::cnt(rank >> 32);
        }

        void set(uint32_t _x, uint32_t value) {
          auto n = value - get<1>(_x);
          if (n == 0) return;
          assert (n < (UINT64_C(1) << 32));

          auto i = _x / 32;
          auto o = _x % 32;

          // bits      rank
          // [76543210][00000000]
          auto b = rank_[i];
          auto r = static_cast<uint32_t>(b);

          if (o + r + 32 < n) {
            b += n - (o + r);
            n = o;
          }

          auto m0 = UINT64_C(-1) << (32 + o);
          auto m1 = UINT64_C( 0) + builtin::ctz(((b & m0) >> 32) | +(UINT64_C(1) << 31)  );  // get bits
          auto m2 = UINT64_C(64) - builtin::clo( (b | m0)     /* & ~(UINT64_C(1) << 31)*/);  // put bits - probably decomment for block sizes >= (1 << 31)

          m1 = ((UINT64_C(1) << (m1 + n)) - (UINT64_C(1) <<      m1 )) << 32;
          m2 = ((UINT64_C(1) <<      m2 ) - (UINT64_C(1) << (m2 - n)));

          b += builtin::cnt(static_cast<uint32_t>(m2));
          b &= ~m1;
          b |= m2 & (UINT64_C(-1) << 32);

          rank_[i] = b;

          assert(value == get<1>(_x));
        }

        void finalize() {
          for (uint32_t i = 0; i < rank_.size() - 1; ++i) {
            auto cur = static_cast<uint32_t>(rank_[i]) + builtin::cnt(rank_[i] >> 32);
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
        std::vector<uint64_t, Allocator> rank_;
    };
  }

  // policies for transformations
  namespace transform {
    // Orginal Burrows Wheeler Transform (differs in the index returned)
    template<class Allocator = std::allocator<int32_t>>
    class bwt {
     public:
        template<class ForwardIt>
        ForwardIt transform(ForwardIt first, ForwardIt last) {
          auto index = builtin::to_lmsr(first, last);

          std::vector<int32_t, Allocator> SA(last - first);
          auto i = divbwt(&*first, &*first, SA.data(), last - first - 1);

          std::rotate(first + i, last - 1, last);
          return index;
        }
    };

    // no transform at all
    template<class Allocator = std::allocator<int32_t>>
    class identity {
     public:
        template<class ForwardIt>
        ForwardIt transform(ForwardIt first, ForwardIt last) {
          (void) last;
          return first;
        }
    };
  }

  // policies for inverting the transform
  namespace inverse {
    // do nothing at all
    template<class Allocator = std::allocator<uint8_t>>
    class noop {
     public:
        using value_type = std::vector<uint8_t, Allocator>;

        template<class R>
        void inverse(std::array<R, 8>& ranks, uint32_t offset, uint32_t n) {
          (void) ranks; (void) offset; (void) n;
        }
    };

    // unbuild the dictionary
    template<class Allocator = std::allocator<uint8_t>>
    class identity {
     public:
        using value_type = std::vector<uint8_t, Allocator>;

        template<class R, class It>
        void inverse(std::array<R, 8>& ranks, uint32_t offset, It out, uint32_t n) {
          (void) offset;
          auto s = (((n / 8) >> 12) + 1) << 12;

          std::array<uint32_t, 8> C;
          for (int i = 0; i < 8; ++i)
            C[i] = ranks[i].template get<0>(n);

#ifdef _OPENMP
          #pragma omp parallel for
#endif
          for (decltype(n) a = 0; a < n; a += s) {
            std::array<uint32_t, 256> D;
            D.fill(0);
            D[1] = a;

            for (int i = 0; i < 7; ++i) {
              for (int j = 0; j < (1 << i); ++j) {
                auto e = D[(1 << i) | j];
                D[(2 << i) | j] =        ranks[i].template get<0>(e);
                D[(3 << i) | j] = C[i] + ranks[i].template get<1>(e);
              }
            }

            std::generate(out + a, out + std::min(n, a + s), [&D, &ranks](){
              auto chr = 0;
              for (int j = 0; j < 8; ++j)
                chr |= ranks[j].bit(D[(1 << j) | chr]++) << j;
              return chr;
            });
          }

          for (int i = 0; i < 8; ++i)
            ranks[i].clear();
        }
    };

    // slow but memory efficient inverse burrows wheeler
    template<class Allocator = std::allocator<uint8_t>>
    class unbwt_bitwise {
     public:
        using value_type = std::vector<uint8_t, Allocator>;

        template<class R, class It>
        void inverse(std::array<R, 8>& ranks, uint32_t offset, It out, uint32_t n) {
          std::array<uint32_t, 8> C;
          for (int i = 0; i < 8; ++i)
            C[i] = ranks[i].template get<0>(n);

          // unbwt
          uint32_t s = 0;

          auto step = [&ranks, &C, &s](decltype(*out)& chr) {
            for (int j = 0; j < 8; j++) {
              auto bit = ranks[j].bit(s);
              chr |= bit << j;
              s = (bit ? C[j] + ranks[j].template get<1>(s) : ranks[j].template get<0>(s));
            }
          };

          for (auto i = offset - 1; i != UINT32_C(-1); --i) step(out[i]);
          for (auto i = n - 1; i != offset - 1; --i) step(out[i]);
        }
    };

    // fast but 5N memory inverse burrows wheeler
    template<class Allocator = std::allocator<uint8_t>>
    class unbwt_bytewise : private identity<Allocator> {
     public:
        using value_type = typename identity<Allocator>::value_type;

        template<class R, class It>
        void inverse(std::array<R, 8>& ranks, uint32_t offset, It out, uint32_t n) {
          identity<Allocator>::inverse(ranks, offset, out, n);

          std::vector<int32_t, Allocator> SA(n);
          inverse_bw_transform(&*out, &*out, nullptr, n, 1);
          std::rotate(out, out + n - offset, out + n);
        }
    };
  }

  // policies for the queue
  namespace queue {
    // a queue implemented using elias gamma coding (linear memory usage)
    template<class Allocator = std::allocator<uint64_t>>
    class packed {
     public:
        packed() : data_{}, cur_(0), pos_(64) {}

        inline void push_back(uint32_t a) {
          assert(a > 0);
          auto n = 127 - 2 * builtin::clz(a);

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
          auto m = 127 - 2 * builtin::clz(a);
          auto n = 127 - 2 * builtin::clz(b);

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
          auto m = 127 - 2 * builtin::clz(a);
          auto n = 127 - 2 * builtin::clz(b);
          auto o = 127 - 2 * builtin::clz(c);

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

        struct iterator {
            using vit = std::vector<uint64_t>::const_iterator;

            explicit iterator(vit it) : it_(std::move(it)), cur_(0) {}

            uint64_t operator*() const {
              auto val = *it_ << cur_;
              return val >> (63 - 2 * builtin::clz(val));
            }

            iterator& operator++() {
              auto val = *it_;
              cur_ += 2 * builtin::clz(val << cur_) + 1;

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

         private:
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
        std::vector<uint64_t, Allocator> data_;
        uint64_t cur_;
        uint64_t pos_;
    };

    // vector based queue eats a huge amount of ram but may be faster (O(n * log(n)) memory)
    template<class Allocator = std::allocator<uint32_t>>
    class vector : public std::vector<uint32_t, Allocator> {
     public:
        using std::vector<uint32_t>::push_back;
        inline void push_back(uint32_t a, uint32_t b) {
          std::vector<uint32_t>::push_back(a);
          std::vector<uint32_t>::push_back(b);
        }

        inline void push_back(uint32_t a, uint32_t b, uint32_t c) {
          std::vector<uint32_t>::push_back(a);
          std::vector<uint32_t>::push_back(b);
          std::vector<uint32_t>::push_back(c);
        }

        void clear() {
          std::vector<uint32_t>::clear();
          std::vector<uint32_t>::shrink_to_fit();
        }
    };
  }

  namespace coder {
    // base class of (currently) all coders not a coder itself
    template<class Allocator = std::allocator<uint8_t>>
    class arithmetic {
     public:
        using state_type = uint64_t;
        using element_type = uint8_t;
        using value_type = std::vector<element_type, Allocator>;
        // maximum value for range that will be adaptivly encoded
        static constexpr const int max = 0;
        static constexpr const int bitsS = sizeof(state_type) * CHAR_BIT; // state bits
        static constexpr const int bitsE = sizeof(element_type) * CHAR_BIT; // element bits

        arithmetic(int i) : l_{0}, h_{UINT64_C(-1)} { (void) i; }

        arithmetic(int i, element_type* first, element_type* last):
          l_{0}, h_{UINT64_C(-1)}, cur_{first}, last_{last} {
          (void) i;
          for (std::size_t i = 0; i < bitsS / bitsE; i++)
            m_ = (m_ << bitsE) + ((cur_ < last_) ? *cur_++ : 0);
        }

        void setv(uint32_t s) {
          while (s) {
            set(s & 1, 3);
            s >>= 1;
          }
          set(2, 3);
        }

        void set(uint32_t s, uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) = delete;

        void set(uint32_t s, uint32_t k) {
          assert(s < k);

          if (builtin::unlikely(h_ - l_ < k)) {
            for (std::size_t i = 0; i < bitsS / bitsE; ++i)
              data_.push_back(l_ >> (bitsS - bitsE * (i + 1)));
            l_ = 0;
            h_ = UINT64_C(-1);
          }

          auto step = (h_ - l_) / k;
          l_ += step * s;
          h_  = step + l_ - 1;

          shift_out();
        }

        uint32_t getv() {
          uint32_t s = 0;
          for (int i = 0, j = get(3); i < 31 && j != 2; ++i, j = get(3))
            s |= j << i;
          return s;
        }

        uint32_t get(uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) = delete;

        uint32_t get(uint32_t k) {
          if (builtin::unlikely(h_ - l_ < k)) {
            for (std::size_t i = 0; i < bitsS / bitsE; ++i)
              m_ = (m_ << bitsE) + ((cur_ < last_) ? *cur_++ : 0);
            l_ = 0;
            h_ = UINT64_C(-1);
          }

          auto step = (h_ - l_) / k;
          auto s    = (m_ - l_) / step;

          l_ += step * s;
          h_  = step + l_ - 1;

          shift_in();
          return s;
        }

        const value_type& flush() {
          shift_out();

          // @todo get this working
          auto bits = builtin::clz(l_ ^ h_) + 1;
          (void) bits;

          for (std::size_t i = 0; i < bitsS / bitsE; ++i)
            data_.push_back(h_ >> (bitsS - bitsE * (i + 1)));

          //data_.push_back((h_ >> (bitsS - bits)) << (bitsE - bits));
          if (cur_ != nullptr) cur_++;
          return data_;
        }

        const value_type& data() const  { return data_; }
        element_type* cur() const { return cur_; };

        /*void clear() {
          data_.clear();
          data_.shrink_to_fit();

          stat_.clear();
          stat_.shrink_to_fit();
        }*/

        // reset this coder for reuse (doesn't reset the stats)
        void reset() {
          data_.clear();
          data_.shrink_to_fit();

          l_ = 0;
          h_ = UINT64_C(-1);
        }

        // reset this coder for reuse (doesn't reset the stats)
        void reset(element_type* first, element_type* last) {
          data_.clear();
          data_.shrink_to_fit();

          l_ = m_ = 0;
          h_ = UINT64_C(-1);
          cur_ = first;
          last_ = last;

          for (std::size_t i = 0; i < bitsS / bitsE; i++)
            m_ = (m_ << bitsE) + ((cur_ < last_) ? *cur_++ : 0);
        }

        static void load_config(std::string file) { (void) file; }

     protected:
        state_type l_;
        state_type h_;
        state_type m_;

        value_type data_;
        element_type* cur_;
        element_type* last_;
        std::vector<uint8_t, Allocator> stat_;

        inline void shift_out() {
          while (!((h_ ^ l_ ) >> (bitsS - bitsE))) {
            data_.push_back(h_ >> (bitsS - bitsE));
            l_ = (l_ << bitsE) +  static_cast<element_type>(0);
            h_ = (h_ << bitsE) + ~static_cast<element_type>(0);
          }
        }

        inline void shift_in() {
          while (!((h_ ^ l_ ) >> (bitsS - bitsE))) {
            m_ = (m_ << bitsE) + ((cur_ < last_) ? *cur_++ : 0);
            l_ = (l_ << bitsE) +  static_cast<element_type>(0);
            h_ = (h_ << bitsE) + ~static_cast<element_type>(0);
          }
        }
    };

    // Coder with uniform distribution (worse compression but faster)
    template<class Allocator = std::allocator<uint8_t>>
    class uniform : public arithmetic<Allocator> {
     public:
        using arithmetic<Allocator>::arithmetic;

        using arithmetic<Allocator>::set;
        void set(uint32_t s, uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) {
          (void) c1; (void) c2; (void) cs;
          set(s, k);  // encode all numbers using uniform distribution
        }

        using arithmetic<Allocator>::get;
        uint32_t get(uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) {
          (void) c1; (void) c2; (void) cs;
          return get(k);
        }
    };

    // Coder with adaptive distribution (better compression but slower)
    template<int L, class Allocator = std::allocator<uint8_t>>
    class adaptive : public arithmetic<Allocator> {
     public:
       using typename arithmetic<Allocator>::state_type;
        // maximum value for range that will be adaptivly encoded without further splitting
        static constexpr const int max = L;
        using arithmetic<Allocator>::bitsS;
        using arithmetic<Allocator>::bitsE;

        adaptive(int i) : arithmetic<Allocator>(i) {
          init(1, i);
        }

        adaptive(int i, typename arithmetic<Allocator>::element_type* first, typename arithmetic<Allocator>::element_type* last):
          arithmetic<Allocator>(i, first, last) {
          init(0, i);
        }

        using arithmetic<Allocator>::set;
        inline void set(uint32_t s, uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) {
          if (k > adaptive::max) {
            set(s & 1, 2);
            return set(s >> 1, (k + (~s & 1)) >> 1, c1, c2, cs);
          }

          auto* ctx = get_context(k, c1, c2, cs);

          state_type n = std::accumulate(ctx, ctx + s, s);
          state_type l = std::accumulate(ctx + s, ctx + k, n - s + k);

          if (builtin::unlikely(arithmetic<Allocator>::h_ - arithmetic<Allocator>::l_ < l)) {
            for (std::size_t i = 0; i < bitsS / bitsE; ++i)
              data_.push_back(l_ >> (bitsS - bitsE * (i + 1)));
            l_ = 0;
            h_ = UINT64_C(-1);
          }

          auto step = (h_ - l_) / l;
          l_ += step * n;
          h_ = l_ + step * (ctx[s] + 1) - 1;

          if (++ctx[s] == 0xFF)
            for (uint32_t i = 0; i < k; ++i)
              ctx[i] >>= 1;

          arithmetic<Allocator>::shift_out();
        }

        using arithmetic<Allocator>::get;
        inline uint32_t get(uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) {
          if (k > adaptive::max) {
            auto s = get(2);
            return (get((k + (~s & 1)) >> 1, c1, c2, cs) << 1) | s;
          }

          auto* ctx = get_context(k, c1, c2, cs);

          auto l = std::accumulate(ctx, ctx + k, k);

          if (builtin::unlikely(h_ - l_ < l)) {
            for (std::size_t i = 0; i < bitsS / bitsE; ++i)
              m_ = (m_ << bitsE) + ((cur_ < last_) ? *cur_++ : 0);
            l_ = 0;
            h_ = UINT64_C(-1);
          }

          auto step = (h_ - l_) / l;

          h_ = l_ - 1;
          uint32_t s = UINT32_C(-1);
          do {
            ++s;
            l_ = h_ + 1;
            h_ += step * (ctx[s] + 1);
          } while (h_ < m_);

          if (++ctx[s] == 0xFF)
            for (uint32_t i = 0; i < k; ++i)
              ctx[i] >>= 1;

          arithmetic<Allocator>::shift_in();

          return s;
        }

        static void load_config(std::string file) {
          std::ifstream archive(file, std::ios::binary | std::ios::ate);
          std::size_t size = archive.tellg();
          if (size != (adaptive::max + 1) * 9) {
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
        using arithmetic<Allocator>::l_;
        using arithmetic<Allocator>::h_;
        using arithmetic<Allocator>::m_;
        using arithmetic<Allocator>::data_;
        using arithmetic<Allocator>::cur_;
        using arithmetic<Allocator>::last_;
        using arithmetic<Allocator>::stat_;
        std::array<uint32_t, L + 1> off_;

        static std::array<std::array<uint8_t, L + 1>, 8> init_;

        inline auto get_context(uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) -> decltype(this->stat_.data()) {
          auto off = off_[k];
          auto bits = off >> 24;
          auto ctx = (((c1 << bits) / cs) << bits) | ((c2 << bits) / cs);

          return this->stat_.data() + (off & 0x00FFFFFF) + ctx * k;
        }

        void init(int mode, int i) {
          std::array<uint8_t, adaptive::max + 1> bits;

          if (0 <= i && i <= 7) {
            if (mode) {
              bits = init_[i];
              /*auto last = 0;
              for (auto& bit : bits) {
                set(bit != last, 2);
                if (bit != last) set(bit, 6);
                last = bit;
              }*/
            } else {
              auto last = 0;
              for (auto& bit : bits) {
                bit = get(2) ? get(6) : last;
                last = bit;
              }
            }
          } else bits.fill(0);

          uint32_t start = 0;
          for (int i = 2; i < adaptive::max + 1; ++i) {
            off_[i] = start | (bits[i] << 24);
            start += i << bits[i] * 2;
          }
          stat_.insert(stat_.begin(), start, 0);
        }
    };

    template<int L, class Allocator>
    std::array<std::array<uint8_t, L + 1>, 8> adaptive<L, Allocator>::init_ = {{
      std::array<uint8_t, L + 1>{{0,0,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0}},
      std::array<uint8_t, L + 1>{{0,0,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0}},
      std::array<uint8_t, L + 1>{{0,0,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,0}},
      std::array<uint8_t, L + 1>{{0,0,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,0}},
      std::array<uint8_t, L + 1>{{0,0,5,5,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,0}},
      std::array<uint8_t, L + 1>{{0,0,5,5,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,0}},
      std::array<uint8_t, L + 1>{{0,0,5,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,0}},
      std::array<uint8_t, L + 1>{{0,0,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,0}}
    }};

    // Used to calculate init_ parameter for the adaptive coder
    template<int L, class Allocator = std::allocator<uint8_t>>
    class scan : public arithmetic<Allocator> {
     public:
        // maximum value for range that will be adaptivly encoded without further splitting
        static constexpr const int max = L;
        using element_type = typename arithmetic<Allocator>::element_type;

        scan(int i): arithmetic<Allocator>{i}, z_(0), i_{i < 0 || i > 7 ? 8 : i}  {}

        scan(int i, element_type* first, element_type* last) = delete; // not a decoder

        using arithmetic<Allocator>::set;
        void set(uint32_t s, uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) {
          if (k > scan::max) {
            z_ += log(2);
            return set(s >> 1, (k >> 1) + ((~s) & 1), c1, c2, cs);
          }

          stat_[k][(((c2 << 8) / cs) << 16) | ((c1 << 8) / cs)].push_back(s);
        }

        uint32_t get(uint32_t k, uint32_t c1, uint32_t c2, uint32_t cs) = delete;
        uint32_t get(uint32_t k) = delete;

        void flush() {
          std::vector<uint16_t> s;

          for (uint32_t k = 2; k < scan::max; ++k) {
            double z_min = std::accumulate(stat_[k].begin(), stat_[k].end(), 0., [k](double z, decltype(*stat_[k].begin())& pair) {
              return z + log(k) * pair.second.size();
            });

            // clustering hash
            for (uint32_t j = 0; j <= 5; ++j) {
              s.clear();
              s.insert(s.begin(), k << (2 * j), 0);

              double z = 0;
              for (auto& pair : stat_[k]) {
                auto c = pair.first;
                c = (((c & 0xFFFF) >> (8 - j)) << j) | (c >> (24 - j));

                auto* ctx = &s[c * k];

                auto l = std::accumulate(ctx, ctx + k, k);
                for (auto& s : pair.second) {
                  z += log(static_cast<double>(l) / (1 + ctx[s]));

                  // update
                  ++l;
                  if (++ctx[s] == 0xFF) {
                    for (uint32_t i = 0; i < k; ++i)
                      ctx[i] >>= 1;
                    l = std::accumulate(ctx, ctx + k, k);
                  }
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

        void clear() {
          arithmetic<Allocator>::clear();
          stat_.clear();
          stat_.shrink_to_fit();
        }

        static void load_config(std::string file) { (void) file;}

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
        std::array<std::unordered_map<uint32_t, std::vector<uint8_t, Allocator>, Allocator>, scan::max + 1> stat_;
        double z_;
        int i_;

        static std::array<std::array<uint8_t, L + 1>, 9> init_;
    };

    template<int L, class Allocator>
    std::array<std::array<uint8_t, L + 1>, 9> scan<L, Allocator>::init_;
  }
          template<class x> struct y;

  // ties all options together and implements the real coding stage
  template<class transform = transform::bwt<>, class inverse = inverse::unbwt_bytewise<>, class coder = coder::adaptive<31>> // coder::uniform<>
  class Compressor : private transform, private inverse {
   public:
      // Init a new compressor
      Compressor() : coder_{{0,1,2,3,4,5,6,7}} {}

      // compress a range [first, last) to [t_first, *t_last)
      // ranges may interleave
      template<class rank = rank::fast<>, class queue = queue::packed<>, class It, class tIt>
      int compress(It first, It last, tIt t_first, tIt* t_last) {
        if (last == first || *t_last == t_first) return -1;

        auto offset = this->transform(first, last);

        auto ranks = builtin::to_dictionary<rank>(first, last);
        auto n = last - first;

        std::array<uint32_t, 8> C;
        for (int i = 0; i < 8; ++i) {
          C[i] = ranks[(i + 7) % 8].template get<0>(n);
          coder_[i].set(C[i], n + 1);
        }

        code<1, queue>(C, ranks, n);

        uint32_t size = 0;
        for (auto& c : coder_) size += c.flush().size();

        // Build the header data
        coder main(-1);
        main.setv(n);
        main.setv(size);
        main.set(offset - first, n); // offset must not be n

        for (uint32_t s = size, i = 0; i < 7; ++i) {
          main.set(coder_[i].data().size(), s + 1);
          s -= coder_[i].data().size();
        }
        main.flush();

        size += main.data().size();
        if (*t_last < t_first + size) {
          *t_last = t_first + size;
          return -2;
        }
        *t_last = t_first + size;

        t_first = std::copy(main.data().begin(), main.data().end(), t_first);
        for (auto& c : coder_) {
          t_first = std::copy(c.data().begin(), c.data().end(), t_first);
          c.reset();
        }

        return 0;
      }

      // get the decompressed and compressed size of the block
      // @return std::pair<uncompressed, compressed> size
      template<class It>
      std::pair<std::size_t, std::size_t> block_info(It first, It last) {
        coder main(-1, &*first, &*last);
        auto n      = main.getv();
        auto size   = main.getv();
        auto offset = main.get(n); // offset must not be n
        (void) offset;

        auto s = size;
        for (int i = 0; i < 7; ++i)
          s -= main.get(s + 1u);
        main.flush();

        auto off = main.cur() - &*first - 1;
        return std::make_pair(n, size + off);
      }

      // decompress a compressed range [*first, last) to [t_first, *t_last)
      // ranges may not interleave
      template<class rank = rank::fast<>, class queue = queue::packed<>, class It, class tIt>
      int decompress(It* first, It last, It t_first, tIt* t_last) {
        coder main(-1, &**first, &*last);
        auto n      = main.getv();
        auto size   = main.getv();
        auto offset = main.get(n); // offset must not be n

        if (last < *first + size) {
          *first = last - size;
          return -2; // not enough loaded
        }

        if (*t_last < t_first + n) {
          *t_last = t_first + n;
          return -3; // not enough memory
        }

        std::array<uint32_t, 8> coder_offsets = {{0}};

        for (int i = 0; i < 7; ++i) {
          coder_offsets[i + 1] = coder_offsets[i] + main.get(size + 1u);
          size -= coder_offsets[i + 1] - coder_offsets[i];
        }
        main.flush();

        auto off = main.cur() - &**first - 1;

        for (int i = 0; i < 8; ++i)
          coder_[i].reset(&*(*first + coder_offsets[i] + off), &*last);

        *first += size;
        *t_last = t_first + n;

        std::array<rank, 8> ranks = {{n, n, n, n, n, n, n, n}};

        std::array<uint32_t, 8> C;
        for (int i = 0; i < 8; ++i) {
          C[i] = coder_[i].get(n + 1);
          ranks[(i + 7) % 8].set(n, n - C[i]);
        }

        code<0, queue>(C, ranks, n);

        for (int i = 0; i < 8; ++i) {
          ranks[i].finalize();
          coder_[i].reset();
        }

        this->inverse(ranks, offset, t_first, n);

        return 0;
      }
   private:
      std::array<coder, 8> coder_;

      template<int mode, class queue, class rank>
      void code(const std::array<uint32_t, 8>& C, std::array<rank, 8>& ranks, const uint32_t n) {
        std::array<std::array<queue, 4>, 8> Q;

        for (int i = 0; i < 8; ++i)
          if (C[i] && n - C[i])
            Q[i][0].push_back(1, C[i], n - C[i]);

        bool again = false;

        do {
          again = false;

#ifdef _OPENMP
          #pragma omp parallel for
#endif
          for (int i = 0; i < 8; ++i) {
            std::array<uint32_t, 2> offset = {{0, 0}};

            for (int j = 0; j < 2; ++j) {
              auto cur = Q[i][j].begin();
              auto end = Q[i][j].end();
              auto s = C[i] * j;

              while (cur != end) {
                s += *cur++ - 1;
                auto s1 = ranks[i].template get<1>(s);

                auto _x0 = *cur++;
                auto _x1 = *cur++;
                auto _x = _x0 + _x1;

                auto _1x = ranks[i].template get<1>(s + _x) - s1;
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
                    _0x0 = ranks[i].template get<0>(s + _x0) - s0;
                    coder_[i].set(_0x0 - min, max - min + 1, _0x, _x1, _x);
                  } else {
                    _0x0 = min + coder_[i].get(max - min + 1, _0x, _x1, _x);
                  }
                  assert(min <= _0x0 && _0x0 <= max);
                }

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
      }
  };
}

#endif  // BCE
