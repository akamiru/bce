#ifndef DICTIONARY_H
#define DICTIONARY_H

// C++ headers
#include <cstdint>

namespace bce {

namespace detail {

template<class V>
void binary_partition(std::uint64_t* dictionary, std::uint64_t& rank, std::uint8_t* data,
                      std::uint8_t*& bytes_0, std::uint8_t*& bytes_1);

}  // namespace detail

/**
 * wavelet matrix + rank dictionary
 *
 */
template<class V>
class dictionary {
 public:
    typedef V vector_type;

    /**
     * Initialise the underlying memory
     *
     * Storage should map to 2 * N + 32 bytes
     * where N is the size of the data in bytes
     *
     * @param T*       storage  memory to store the dictionary
     * @param int32_t  size     size of the text N in bytes
     */
    template<class T>
    dictionary(T* storage, std::int32_t size) :
      storage_(reinterpret_cast<void*>(storage)),
      size_(size) {}

    /**
     * Construct a dictionary from data
     *
     * @param uint8_t* data     data to construct the dictionary for
     * @param uint8_t* work     temporary working space
     */
    void construct(std::uint8_t* data, std::uint8_t* work);

    /**
     * Count the zeros at each position
     *
     * @param uint8_t* data     data to count the bits for
     */
    void count_zeros(std::uint8_t* data);

    /**
     * Get the number of zeros at given position
     *
     * @param size_t bit      position of the bit to count
     */
    int32_t get_zeros(std::size_t bit) const;

    /**
     * Get a queryable view on the dictionary
     *
     * @param size_t bit      position of the bit to count
     */
    //view_type get_view(std::size_t bit) const = 0;

 protected:
    void*       storage_;
    std::size_t size_   ;
};

}  // namespace bce

#endif  // DICTIONARY_H