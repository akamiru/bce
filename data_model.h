#ifndef CSE_H
#define CSE_H

// C++ headers
#include <algorithm>
#include <array>
#include <cstdint>

// Project headers
#include "context.h"

// Library headers
//#include <divsufsort.h>

namespace bce {

namespace detail {

/**
 * Rotate a string to its lexicographically minimal string rotation
 */
std::size_t to_lmsr(std::uint8_t* data, std::size_t size) {
  // Inspired by: MaskRay [github]
  std::uint8_t *i = data,
               *j = data + 1,
               *e = data + size,
               *k_i, *k_j;
  std::size_t  k = 0;

  while (j < e) {
    for (k = 1, k_i = i, k_j = j; (*k_i == *k_j) & (k_i < i + size - 1);) {
      ++k; ++k_i; ++k_j;
      if (k_i >= e) k_i = data;
      if (k_j >= e) k_j = data;
    }

    if (*k_i <= *k_j) {
      j += k;
      continue;
    }

    i += k;
    if (i < j) i = j++;
    else       j = i + 1;
  }

  return i - data;
}

template<class I, class O>
O copy(I first, I last, O d_first) {
#ifdef __INTEL_COMPILER
  if (first != d_first) {
    while (first != last)
      *d_first++ = *first++;
    return d_first;
  }
  return d_first + (last - first);
#else
  return std::copy(first, last, d_first);
#endif
}

template<class I, class O>
O copy_backward(I first, I last, O d_last) {
#ifdef __INTEL_COMPILER
  if (last != d_last) {
    while (first != last)
      *(--d_last) = *(--last);
    return d_last;
  }
  return d_last - (last - first);
#else
  return std::copy_backward(first, last, d_last);
#endif
}

}  // namespace detail

class data_model {
 public:
    enum class error : std::size_t {
      SUCCESS               = 0,
      NEGATIVE_SIZE         = 1,
      LIMITED_WORKING_SPACE = 2,
      UNALIGNED_SIZE        = 4,
      OUT_OF_SPACE          = 8
    };

    /**
     *
     */
    data_model() {}

    /**
     *
     */
    template<class V>
    error compress(std::uint8_t* data, std::int32_t isize,
                   std::uint8_t* work, std::size_t  wsize) {
      // Size of the used vector
      using vector_type = V;
      const auto vector_size  = V::size;
      const auto vector_bytes = V::size * sizeof(std::int32_t);
      auto align_next = [vector_size](auto i) {
        return (i + vector_size - 1) & -vector_size;
      };

      // size as size_t
      auto size = static_cast<std::size_t>(isize);

      // check if the size is non negative
      if (isize < 0)
        return error::NEGATIVE_SIZE;

      // check if working space is large enough to fit:
      // the size, at least 3*8*2 queue vectors and over-the-end space
      if (wsize < std::max(4 * size, 48 * vector_bytes) + 2 * vector_bytes)
        return error::LIMITED_WORKING_SPACE;

      // check if size in aligned to the vector size
      if (size & (vector_bytes - 1))
        return error::UNALIGNED_SIZE;

      // check if data in aligned to the vector size
      if (reinterpret_cast<std::uintptr_t>(data) & (vector_bytes - 1))
        return error::UNALIGNED_SIZE;

      // check if work in aligned to the vector size
      if (reinterpret_cast<std::uintptr_t>(work) & (vector_bytes - 1))
        return error::UNALIGNED_SIZE;

#if 0
      // we need the true cyclic bwt without sentinel
      auto idx = detail::to_lmsr(data, size);
      auto i = divbwt(data, data, reinterpret_cast<saidx_t*>(work), size - 1);
      std::rotate(data + i, data + size - 1, data + size);

      // store the bwt during development to save time on tests
      std::ofstream f("enwik8.bwt", std::ios::binary | std::ios::trunc);
      f.write(reinterpret_cast<const char*>(data), size);
#endif

      // memory layout
      auto dictionary_size = 2 * size + 2 * vector_bytes;
      auto queue_size      =    wsize - dictionary_size ;
      auto queue_step      = (queue_size / 3) & -vector_bytes;
           queue_size      = 3 * queue_step;
           queue_step      = queue_step / sizeof(std::int32_t);
      auto queue_base      = reinterpret_cast<std::int32_t*>(work);
      auto queue_last      = queue_base + queue_step;

      // construct the dictionary
      bce::dictionary<V> dict(work + queue_size, size);
      dict.construct(data, work);

#if 1
      // initialze the queues
      std::array<std::array<std::int32_t*, 4>, 9> queue;

      auto queue_local = queue_base;
      for (std::size_t i = 0; i < 8; ++i) {
        // empty the current vector (should compile to a single vector mov)
        for (std::size_t j = 0; j < vector_size; ++j) queue_local[0 * queue_step + j] = 0;
        for (std::size_t j = 0; j < vector_size; ++j) queue_local[1 * queue_step + j] = 0;
        for (std::size_t j = 0; j < vector_size; ++j) queue_local[2 * queue_step + j] = 0;

        // init the starting context
         // interval start
        queue_local[0 * queue_step] =                           0;
         // number of zeros in the prev context
        queue_local[1 * queue_step] = dict.get_zeros((i + 7) % 8);
         // interval end
        queue_local[2 * queue_step] =                        size;

        // advance
        queue[i][0]  = queue_local;
        queue[i][1]  = queue_local + vector_size;

#if 0
        // minimal spacing required
        queue_local += 2 * vector_size;
#else
        // equally spaced
        queue_local += (queue_step / sizeof(std::int32_t) / 8) & -vector_size;
#endif
      }
      // queue end
      queue[8][0] = queue_local;

      // run the main compression
      bce::context<vector_type> ctx(dict);
      for(std::size_t prefix_length = 0; true; ++prefix_length) {
        // main loop
        //#pragma omp parallel for
        for (std::size_t i = 0; i < 8; ++i) {
          // is this queue empty?
          if (queue[i][0] != queue[i][1]) {
            std::int32_t* coder_last;

            // evaluate the contexts
            std::tie(queue[i][2], queue[i][3], coder_last) = ctx.evaluate(
              // queue
              queue[i][0],
              queue_step,
              queue[i][1],
              // coder
              reinterpret_cast<std::int32_t*>(data),
              size / 3 / sizeof(std::int32_t),
              // bit position
              (i + prefix_length) % 8
            );

            // @todo actually compress
          } else // of course we don't produce any new elements
            queue[i][2] = queue[i][3] = queue[i][0];
        }

        // at this point queue has 4 pointers each:
        //  queue[i][0] = starting point of the queue we just processed
        //  queue[i][1] = one past the end of the queue we just processed (rouned up to the next full vector)
        //  queue[i][1] = one past the end of n0 output queue
        //  queue[i][1] = one past the end of n1 output queue
        // this means the orginal range was split into 2 new:
        //  the first from queue[i][0] to queue[i][1] (not rounded up on full vector)
        //  the first from queue[i][1] to queue[i][3] (not rounded up on full vector)
        // so we got to copy [queue[i][1], queue[i][3]) behind [queue[i][0], queue[i][1])
        // and additionally we might need to move [queue[i][0], queue[i][1]) in order
        // to gain enough space for the next iteration

        // check if there's still enough space in each queue for the next iteration
        // we do this here so the previous loop doesn't have to synchronize
        //  we usally should have to rescale only a few times
        bool rescale = false;
        std::size_t contexts_total = 0;
        for (std::size_t i = 0; i < 8; ++i) {
          auto contexts = (queue[i][2] - queue[i][0]) + (queue[i][3] - queue[i][1]);
          contexts_total += align_next(contexts);
          if (2 * align_next(contexts) > queue[i + 1][0] - queue[i][0])
            rescale = true;
        }

        if (contexts_total == 0)
          break;

        if (rescale) {
          // we rescale the queues in a two step procedure:
          //  - first we compact the queues left to right using std::copy
          //  - then we expand the queue right to left using std::copy_backward
          // This way we can do it in place with the guarantee
          // to not override anything

          auto queue_scale = queue_step / contexts_total;
          if (queue_scale < 2)
            return error::OUT_OF_SPACE;  // not enough memory

          // compact the queues
          auto queue_cur   = queue_base;
          auto queue_nxt   = queue_base;
          for (std::size_t i = 0; i < 8; ++i) {
            auto n0 = queue[i][2] - queue[i][0];
            auto n1 = queue[i][3] - queue[i][1];

            auto queue_cur_last  = queue_cur;
            for (std::size_t j = 3; j-- > 0;) {
              // copy n0
              queue_cur_last = detail::copy(
                queue[i][0] + j * queue_step,
                queue[i][2] + j * queue_step,
                queue_cur   + j * queue_step
              );
              // copy n1
              queue_cur_last = detail::copy(
                queue[i][1] + j * queue_step,
                queue[i][3] + j * queue_step,
                queue_cur_last
              );
            }

            // update queue
            queue[i][0] = queue_cur     ;
            queue[i][1] = queue_cur_last;

            // number of contexts
            auto n_contexts = queue_cur_last - queue_cur;
            // used range end
            queue[i][2] = queue_nxt + n_contexts;
            // used range end aligned to the next vector
            queue[i][3] = queue_nxt + align_next(n_contexts);
            // update queue_nxt to allow space for queue_scale * n
            queue_nxt  += queue_scale * align_next(n_contexts);

            queue_cur = queue_cur_last;
          }

          // expand the queues
          for (std::size_t i = 8; i-- > 0;) {
            std::int32_t* queue_start;
            for (std::size_t j = 3; j-- > 0;) {
              // copy queue
              queue_start = detail::copy_backward(
                queue[i][0] + j * queue_step,
                queue[i][1] + j * queue_step,
                queue[i][2] + j * queue_step
              );
              // clear partial vectors
              std::fill(
                queue[i][2] + j * queue_step,
                queue[i][3] + j * queue_step,
                0
              );
            }

            // update queue
            queue[i][0] = queue_start;
            queue[i][1] = queue[i][3];
          }
        } else {
          // the queue has enough space to support the next iteration:
          //  only copy n0 and n1 together
          for (std::size_t i = 0; i < 8; ++i) {
            auto n0 = queue[i][2] - queue[i][0];
            auto n1 = queue[i][3] - queue[i][1];

            if (n0 + n1 > 0) {
              for (std::size_t j = 3; j-- > 0;) {
                // copy ones
                detail::copy(
                  queue[i][1] + j * queue_step,
                  queue[i][3] + j * queue_step,
                  queue[i][2] + j * queue_step
                );
                // clear partial vectors
                std::fill(
                  queue[i][2] + j * queue_step + n1,
                  queue[i][0] + j * queue_step + align_next(n0 + n1),
                  0
                );
              }
              // the start doesn't change
              queue[i][1] = queue[i][0] + align_next(n0 + n1);
            }
          }
        }
      }
#endif

      return error::SUCCESS;
    }
};

}  // namespace bce

#endif  // CSE_H