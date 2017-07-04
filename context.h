#ifndef CONTEXT_H
#define CONTEXT_H

// Optional parameters
#ifndef BCE_CONTEX_DISPLAY_SUM
  #define BCE_CONTEX_DISPLAY_SUM 0
#endif

// C++ headers
#if BCE_CONTEX_DISPLAY_SUM
  #include <cmath>
#endif
#include <cstdint>
#include <tuple>

// Project headers
#include "dictionary.h"

namespace bce {

template<class V>
class context {
 public:
    typedef V vector_type;

    context(const bce::dictionary<V>& dictionary) : dictionary_(dictionary) {}

    decltype(auto)
    evaluate(std::int32_t* queue, std::size_t queue_step, std::int32_t* queue_last,
             std::int32_t* coder, std::size_t coder_step,
             std::int64_t bit) {
      // current queue
      auto queue_iter = queue;

      // next queue
      auto queue_out0 = queue_iter;
      auto queue_out1 = queue_last;

      // dictionary
      vector_type offset {this->dictionary_.get_zeros(bit)};
      auto        view  = this->dictionary_.get_view (bit) ;

      // coder
      auto coder_iter = coder;

      while (queue_iter < queue_last) {
        // load queue intervals [v1, v2) [v2, v3)
        auto v_1 = vector_type::load(queue_iter + 0 * queue_step);
        auto v_2 = vector_type::load(queue_iter + 1 * queue_step);
        auto v_3 = vector_type::load(queue_iter + 2 * queue_step);
        queue_iter += vector_type::size;

        // gather ranks
        auto r_1 = view.query(v_1);
        auto r_2 = view.query(v_2);
        auto r_3 = view.query(v_3);

        // reduce context: conditional swap _x0, _x1
        auto n__x_ = v_3   - v_1  ;
        auto n__x1 = r_3   - r_1  ;
        auto n__x0 = n__x_ - n__x1;

        auto n_1x_ = v_3   - v_2  ;
        auto n_1x1 = r_3   - r_2  ;
        auto n_1x0 = n_1x_ - n_1x1;

        n_1x1  = vector_type::blend(n_1x0, n_1x1, n__x1 < n__x0);
        n__x1  = vector_type::min  (n__x0, n__x1);

        // reduce context: conditional swap 0x_, 1x_
        auto n_0x_ = n__x_ - n_1x_;
        auto n_0x1 = n__x1 - n_1x1;
        n_1x1  = vector_type::blend(n_0x1, n_1x1, n_1x_ < n_0x_);
        n_1x_  = vector_type::min  (n_0x_, n_1x_);

        // reduce context: conditional swap 1x_, _x1
        auto temp = n__x1;
        n__x1 = vector_type::min(temp, n_1x_);
        n_1x_ = vector_type::max(temp, n_1x_);

        // calculate context offset
        const std::int32_t L {    64};
        const vector_type  L4{-L - 4};
        auto context_offset =
          ( (n__x1 + vector_type{1}) * (n__x1 + L4)
          - (n_1x_ + vector_type{1}) * (n_1x_ + L4) ) * n__x_
          + (n__x_ - (n_1x_ + n_1x_)) + vector_type{1}
        ;

        // store coder information
        auto      mask_c = n__x1 > vector_type{0};
        auto pack_mask_c = mask_c.pack_mask();
        n__x1         .pack_and_store(coder_iter + 0 * coder_step, pack_mask_c);
        n_1x1         .pack_and_store(coder_iter + 1 * coder_step, pack_mask_c);
        context_offset.pack_and_store(coder_iter + 2 * coder_step, pack_mask_c);
        coder_iter = mask_c.pack_advance(coder_iter);

        // store next ones queue
        auto      mask_1 = (r_2 > r_1) & (r_3 > r_2);
        auto pack_mask_1 = mask_1.pack_mask();
        (r_1 + offset).pack_and_store(queue_out1 + 0 * queue_step, pack_mask_1);
        (r_2 + offset).pack_and_store(queue_out1 + 1 * queue_step, pack_mask_1);
        (r_3 + offset).pack_and_store(queue_out1 + 2 * queue_step, pack_mask_1);
        queue_out1 = mask_1.pack_advance(queue_out1);

        // store next zeros queue
        auto      mask_0 = (v_2 - v_1 > r_2 - r_1) & (v_3 - v_2 > r_3 - r_2);
        auto pack_mask_0 = mask_0.pack_mask();
        (v_1 - r_1).pack_and_store(queue_out0 + 0 * queue_step, pack_mask_0);
        (v_2 - r_2).pack_and_store(queue_out0 + 1 * queue_step, pack_mask_0);
        (v_3 - r_3).pack_and_store(queue_out0 + 2 * queue_step, pack_mask_0);
        queue_out0 = mask_0.pack_advance(queue_out0);
      }

#if BCE_CONTEX_DISPLAY_SUM
      static double sum = 0;
      for (auto it = coder; it != coder_iter; ++it)
        sum += log(*it + 1);
      printf("%f\n", sum / log(256));
#endif

      // return the number of codings and contexts
      return std::make_tuple(
        queue_out0,  // n0 contexts
        queue_out1,  // n1 contexts
        coder_iter   // coding points
      );
    }

 private:
    bce::dictionary<V> dictionary_;
};

}  // namespace bce

#endif  // CONTEXT_H