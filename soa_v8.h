/*
 * Structure-of-Arrays template with "columns" and "scalars", defined through preprocessor macros,
 * with compile-time size and alignment, and accessors to the "rows" and "columns".
 */

#pragma once

#include <iostream>
#include <cassert>
#include "boost/preprocessor.hpp"

// CUDA attributes
#ifdef __CUDACC__
#define SOA_HOST_ONLY __host__
#define SOA_HOST_DEVICE __host__ __device__
#define SOA_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define SOA_HOST_ONLY
#define SOA_HOST_DEVICE
#define SOA_HOST_DEVICE_INLINE inline
#endif

// compile-time sized SoA

// Helper template managing the value within it column
template<typename T, class P>
class SoAValue {
public:
  SOA_HOST_DEVICE_INLINE SoAValue(size_t baseOffset, size_t index, P & parent):
    baseOffset_(baseOffset), index_(index), parent_(parent) {}
  SOA_HOST_DEVICE_INLINE operator T&() { return getRef(); }
  SOA_HOST_DEVICE_INLINE operator const __restrict__ T&() const { 
    const SoAValue<T, P>& cThis = *this;
    return cThis.getRef(); 
  }
  SOA_HOST_DEVICE_INLINE T* operator& () { return &getRef(); }
  SOA_HOST_DEVICE_INLINE const __restrict__ T* operator& () const { return &getRef(); }
  template <typename T2>
  SOA_HOST_DEVICE_INLINE T& operator= (const T2& v) { return getRef() = v; }
  typedef T valueType;
  static constexpr auto valueSize = sizeof(T);
private:
  SOA_HOST_DEVICE_INLINE
  T& getRef() {
    return reinterpret_cast<T *>(&parent_.mem_[baseOffset_])[index_];
  }
  SOA_HOST_DEVICE_INLINE
  const __restrict__ T& getRef() const {
    if constexpr(!std::is_pointer<T>::value) {
      const __restrict__ T * base = 
        reinterpret_cast<const __restrict__ T *>(&parent_.mem_[baseOffset_]);
      return base[index_];
    }
    return reinterpret_cast<const T *>(&parent_.mem_[baseOffset_])[index_];
  }
  size_t baseOffset_, index_;
  P & parent_;
};

/* declare "scalars" (one value shared across the whole SoA) and "columns" (one value per element) */
#define _VALUE_TYPE_SCALAR 0
#define _VALUE_TYPE_COLUMN 1

#define SoA_scalar(TYPE, NAME) (_VALUE_TYPE_SCALAR, TYPE, NAME)
#define SoA_column(TYPE, NAME) (_VALUE_TYPE_COLUMN, TYPE, NAME)

/* General helper macros for iterating on columns or scalars reparately */
/* Predicate for Boost PP filters. ELEMENT is expected to be SoA_scalar or Soa_column */
#define _IS_VALUE_TYPE_PREDICATE(S, DATA, ELEM)                                                                                     \
  BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(0, ELEM), DATA)

/* Iterate on the macro MACRO chosen type of elements */
#define _ITERATE_ON_VALUE_TYPE(MACRO, DATA, VALUE_TYPE, ...) \
  BOOST_PP_SEQ_FOR_EACH(MACRO, DATA,\
    BOOST_PP_SEQ_FILTER(_IS_VALUE_TYPE_PREDICATE, VALUE_TYPE, \
      BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__) \
    )\
  )

/* Iterate on the macro MACRO chosen type of elements and return the result as a comma separated list */
#define _ITERATE_ON_VALUE_TYPE_COMMA(MACRO, DATA, VALUE_TYPE, ...) \
  BOOST_PP_TUPLE_ENUM(\
    BOOST_PP_SEQ_TO_TUPLE(\
      _ITERATE_ON_VALUE_TYPE(MACRO, DATA, VALUE_TYPE, __VA_ARGS__)\
    )\
  )

/* Iterate MACRO on all elements */
#define _ITERATE_ON_ALL(MACRO, DATA, ...) \
  BOOST_PP_SEQ_FOR_EACH(MACRO, DATA,\
    BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__) \
  )

/* dump SoA fields information; these should expand to, for columns:
 *
 *   std::cout << "  x_[" << SoA::size << "] at " 
 *             << offsetof(SoA, SoA::x_) << " has size " << sizeof(SoA::x_) << std::endl;
 *
 * and for scalars:
 *
 *   std::cout << "  x_ at " 
 *             << offsetof(SoA, SoA::x_) << " has size " << sizeof(SoA::x_) << std::endl;
 *
 */

#define _DECLARE_SOA_DUMP_INFO_IMPL(IS_COLUMN, TYPE, NAME)                                                                          \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    std::cout << "  " BOOST_PP_STRINGIZE(NAME) "_ at offset " << offset                                                             \
              <<  " has size " << sizeof(TYPE) * nElements << " and padding "                                                       \
              << ((nElements * sizeof(TYPE) / byteAlignment) + 1) * byteAlignment - (sizeof(TYPE) * nElements) <<  std::endl;       \
  ,                                                                                                                                 \
    std::cout << "  " BOOST_PP_STRINGIZE(NAME) "_ at offset " << offset                                                             \
              <<  " has size " << sizeof(TYPE) << " and padding "                                                                   \
              << (sizeof(TYPE) / byteAlignment + 1) * byteAlignment - sizeof(TYPE) <<  std::endl;                                   \
  )

#define _DECLARE_SOA_DUMP_INFO(R, DATA, TYPE_NAME)                                                                                  \
  BOOST_PP_EXPAND(_DECLARE_SOA_DUMP_INFO_IMPL TYPE_NAME)

#define _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL(IS_COLUMN, TYPE, NAME)                                                                    \
  BOOST_PP_CAT(NAME, BaseOffset_) = curOffset;                                                                                      \
    BOOST_PP_IIF(IS_COLUMN,                                                                                                         \
    curOffset += (((nElements_ * sizeof(TYPE) - 1) / byteAlignment_) + 1) * byteAlignment_;                                         \
  ,                                                                                                                                 \
    curOffset += (((sizeof(TYPE) - 1) / byteAlignment_) + 1) * byteAlignment_;                                                      \
  )

#define _ASSIGN_SOA_COLUMN_OR_SCALAR(R, DATA, TYPE_NAME)                                                                            \
  _ASSIGN_SOA_COLUMN_OR_SCALAR_IMPL TYPE_NAME

#define _ACCUMULATE_SOA_ELEMENT_IMPL(IS_COLUMN, TYPE, NAME)                                                                         \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    ret += (((nElements * sizeof(TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                                                  \
  ,                                                                                                                                 \
    ret += (((sizeof(TYPE) - 1) / byteAlignment) + 1) * byteAlignment;                                                              \
  )

#define _ACCUMULATE_SOA_ELEMENT(R, DATA, TYPE_NAME)                                                                                 \
  _ACCUMULATE_SOA_ELEMENT_IMPL TYPE_NAME

#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                             \
  SOA_HOST_DEVICE_INLINE                                                                                                            \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE const & NAME() { return * (soa_. NAME () + index_); }                                                                      \
  ,                                                                                                                                 \
    TYPE const & NAME() { return soa_. NAME (); }                                                                                   \
  )

#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR(R, DATA, TYPE_NAME)                                                                     \
  BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL TYPE_NAME)

/* declare AoS-like element value aregs for contructor; these should expand,for columns only */
#define _DECLARE_ELEMENT_VALUE_ARG_IMPL(IS_COLUMN, TYPE, NAME)                                                                      \
  BOOST_PP_IIF(IS_COLUMN,  \
    (size_t BOOST_PP_CAT (NAME, BaseOffset))\
  ,\
    BOOST_PP_EMPTY()\
  )

#define _DECLARE_ELEMENT_VALUE_ARG(R, DATA, TYPE_NAME)                                                                              \
  BOOST_PP_EXPAND(_DECLARE_ELEMENT_VALUE_ARG_IMPL TYPE_NAME)

/* declare AoS-like element value members; these should expand,for columns only */
/* We filter the value list beforehand to avoid having a comma inside a macro parameter */
#define _DECLARE_ELEMENT_VALUE_COPY_IMPL(IS_COLUMN, TYPE, NAME)                                                                     \
  static_cast<TYPE &>(NAME) = static_cast<std::add_const<TYPE>::type __restrict__ &>(other.NAME);

#define _DECLARE_ELEMENT_VALUE_COPY(R, DATA, TYPE_NAME)                                                                             \
  _DECLARE_ELEMENT_VALUE_COPY_IMPL TYPE_NAME

/* declare AoS-like element value members; these should expand,for columns only */
/* We filter the value list beforehand to avoid having a comma inside a macro parameter */
#define _DECLARE_ELEMENT_VALUE_MEMBER_IMPL(IS_COLUMN, TYPE, NAME, CLASS)                                                             \
  SoAValue<TYPE, CLASS> NAME;

#define _DECLARE_ELEMENT_VALUE_MEMBER(R, CLASS, TYPE_NAME)                                                                           \
  BOOST_PP_EXPAND(_DECLARE_ELEMENT_VALUE_MEMBER_IMPL BOOST_PP_TUPLE_PUSH_BACK (TYPE_NAME, CLASS))

#define _DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION_IMPL(IS_COLUMN, TYPE, NAME, INDEX, PARENT)                                      \
  (NAME (BOOST_PP_CAT(NAME, BaseOffset), INDEX, PARENT))

/* declare AoS-like element value args for contructor; these should expand,for columns only */
/* This macro returns a tuple and with the other results of the iteration will be turned into a comma separated list */
/* Format is: (<name> (<name>BaseOffset, index, parent)) */
/* index and parent are provided as members of the DATA tuple */
/* This macro creates single tuple out of TYPE_NAME and DATA and call its helper _IMPL macro */
#define _DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION(R, DATA, TYPE_NAME)                                                            \
  _DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION_IMPL                                                                                 \
      BOOST_PP_SEQ_TO_TUPLE(BOOST_PP_TUPLE_TO_SEQ(TYPE_NAME) BOOST_PP_TUPLE_TO_SEQ(DATA))

/* declare AoS-like element value args for contructor; these should expand,for columns only */
#define _DECLARE_ELEMENT_CONSTR_CALL_IMPL(IS_COLUMN, TYPE, NAME)                                                                    \
  (BOOST_PP_CAT(NAME, BaseOffset_))

#define _DECLARE_ELEMENT_CONSTR_CALL(R, DATA, TYPE_NAME)                                                                            \
  BOOST_PP_EXPAND(_DECLARE_ELEMENT_CONSTR_CALL_IMPL TYPE_NAME)

#define _DECLARE_SOA_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                                           \
  SOA_HOST_DEVICE_INLINE                                                                                                            \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE* NAME() { return reinterpret_cast<TYPE *>(&mem_[BOOST_PP_CAT(NAME, BaseOffset_)]); }                                       \
  ,                                                                                                                                 \
    TYPE& NAME() { return reinterpret_cast<TYPE &>(mem_[BOOST_PP_CAT(NAME, BaseOffset_)]); }                                        \
  )

#define _DECLARE_SOA_ACCESSOR(R, DATA, TYPE_NAME)                                                                                   \
  BOOST_PP_EXPAND(_DECLARE_SOA_ACCESSOR_IMPL TYPE_NAME)

#define _DECLARE_SOA_CONST_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                                     \
  SOA_HOST_DEVICE_INLINE                                                                                                            \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE const* NAME() const { return reinterpret_cast<TYPE *>(&mem_[BOOST_PP_CAT(NAME, BaseOffset_)]); }                           \
  ,                                                                                                                                 \
    TYPE const& NAME() const { return reinterpret_cast<TYPE &>(mem_[BOOST_PP_CAT(NAME, BaseOffset_)]); }                            \
  )

#define _DECLARE_SOA_CONST_ACCESSOR(R, DATA, TYPE_NAME)                                                                             \
  BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ACCESSOR_IMPL TYPE_NAME)

#define _DECLARE_SOA_DATA_MEMBER_IMPL(IS_COLUMN, TYPE, NAME)                                                                        \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    size_t BOOST_PP_CAT(NAME, BaseOffset_);                                                                                         \
  ,                                                                                                                                 \
    size_t BOOST_PP_CAT(NAME, BaseOffset_);                                                                                         \
  )

#define _DECLARE_SOA_DATA_MEMBER(R, DATA, TYPE_NAME)                                                                                \
  BOOST_PP_EXPAND(_DECLARE_SOA_DATA_MEMBER_IMPL TYPE_NAME)

#ifdef DEBUG
#define _DO_RANGECHECK true
#else
#define _DO_RANGECHECK false
#endif

/*
 * A macro defining a SoA (structure of variable sized arrays variant).
 */
#define declare_SoA_template(CLASS, ...)                                                                                            \
struct CLASS {                                                                                                                      \
  /* Value class direct access */                                                                                                   \
  template <typename T, class P>\
  friend class SoAValue;                                                                                                            \
                                                                                                                                    \
  /* these could be moved to an external type trait to free up the symbol names */                                                  \
  using self_type = CLASS;                                                                                                          \
                                                                                                                                    \
  /* For CUDA applications, we align to the 128 bytes of the cache lines.                                                           \
   * See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0 this is still valid                     \
   * up to compute capability 8.X.                                                                                                  \
   */                                                                                                                               \
  constexpr static size_t defaultAlignment = 128;                                                                                   \
                                                                                                                                    \
  /* dump the SoA internaul structure */                                                                                            \
  SOA_HOST_ONLY                                                                                                                     \
  static void dump(size_t nElements, size_t byteAlignment = defaultAlignment) {                                                     \
    std::cout << #CLASS "(" << nElements << ", " << byteAlignment << "): " << '\n';                                                 \
    std::cout << "  sizeof(" #CLASS "): " << sizeof(CLASS) << '\n';                                                                 \
    std::cout << "  computeDataSize(...): " << computeDataSize(nElements, byteAlignment);                                           \
    size_t offset=0;                                                                                                                \
    _ITERATE_ON_ALL(_DECLARE_SOA_DUMP_INFO, ~, __VA_ARGS__)                                                                         \
    std::cout << std::endl;                                                                                                         \
  }                                                                                                                                 \
  /* Helper function used by caller to externally allocate the storage */                                                           \
  static size_t computeDataSize(size_t nElements, size_t byteAlignment = defaultAlignment) {                                        \
    size_t ret = 0;                                                                                                                 \
    _ITERATE_ON_ALL(_ACCUMULATE_SOA_ELEMENT, ~, __VA_ARGS__)                                                                        \
    return ret;                                                                                                                     \
  }                                                                                                                                 \
                                                                                                                                    \
  SOA_HOST_DEVICE_INLINE size_t nElements() const { return nElements_; }                                                            \
  SOA_HOST_DEVICE_INLINE size_t byteAlignment() const { return byteAlignment_; }                                                    \
                                                                                                                                    \
  /* Constructor relying on user provided storage */                                                                                \
  CLASS(std::byte* mem, size_t nElements, size_t byteAlignment = defaultAlignment):                                                 \
      mem_(mem), nElements_(nElements), byteAlignment_(byteAlignment) {                                                             \
    size_t curOffset = 0;                                                                                                           \
    _ITERATE_ON_ALL(_ASSIGN_SOA_COLUMN_OR_SCALAR, ~, __VA_ARGS__)                                                                   \
    /* Sanity check: we should have reached the computed size; */                                                                   \
    if(mem_ + computeDataSize(nElements_, byteAlignment_) != &mem[curOffset])                                                       \
      throw std::out_of_range("In " #CLASS "::" #CLASS ": unexpected end pointer.");                                                \
  }                                                                                                                                 \
                                                                                                                                    \
  /* AoS-like accessor to individual elements */                                                                                    \
  struct const_element {                                                                                                            \
    SOA_HOST_DEVICE_INLINE                                                                                                          \
    const_element(CLASS const& soa, int index) :                                                                                    \
      soa_(soa),                                                                                                                    \
      index_(index)                                                                                                                 \
    { }                                                                                                                             \
                                                                                                                                    \
    _ITERATE_ON_ALL(_DECLARE_SOA_CONST_ELEMENT_ACCESSOR, ~, __VA_ARGS__)                                                            \
                                                                                                                                    \
  private:                                                                                                                          \
    CLASS const& soa_;                                                                                                              \
    const int index_;                                                                                                               \
  };                                                                                                                                \
                                                                                                                                    \
  struct element {                                                                                                                  \
    SOA_HOST_DEVICE_INLINE                                                                                                          \
    element(size_t index, CLASS &parent,                                                                                            \
      /* Turn Boost PP */                                                                                                           \
      _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_VALUE_ARG, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__)                                  \
    ):                                                                                                                              \
      _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION, (index, parent), _VALUE_TYPE_COLUMN, __VA_ARGS__)  \
       {}                                                                                                                           \
    SOA_HOST_DEVICE_INLINE                                                                                                          \
    element& operator=(const element& other) {                                                                                      \
      _ITERATE_ON_VALUE_TYPE(_DECLARE_ELEMENT_VALUE_COPY, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__)                                       \
      return *this;                                                                                                                 \
    }                                                                                                                               \
    _ITERATE_ON_VALUE_TYPE(_DECLARE_ELEMENT_VALUE_MEMBER, CLASS, _VALUE_TYPE_COLUMN, __VA_ARGS__)                                   \
  };                                                                                                                                \
                                                                                                                                    \
  /* AoS-like accessor */                                                                                                           \
  SOA_HOST_DEVICE_INLINE                                                                                                            \
  element operator[](size_t index) {                                                                                                \
    rangeCheck(index);                                                                                                              \
    return element(index, *this,                                                                                                    \
        _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_CONSTR_CALL, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__) );                           \
  }                                                                                                                                 \
                                                                                                                                    \
  SOA_HOST_DEVICE_INLINE                                                                                                            \
  const element operator[](size_t index) const {                                                                                    \
    rangeCheck(index);                                                                                                              \
    return element(index, *const_cast<CLASS *>(this),                                                                               \
        _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_CONSTR_CALL, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__) );                           \
  }                                                                                                                                 \
                                                                                                                                    \
  /* accessors */                                                                                                                   \
  _ITERATE_ON_ALL(_DECLARE_SOA_ACCESSOR, ~, __VA_ARGS__)                                                                            \
  _ITERATE_ON_ALL(_DECLARE_SOA_CONST_ACCESSOR, ~, __VA_ARGS__)                                                                      \
                                                                                                                                    \
  /* dump the SoA internal structure */                                                                                             \
  template <typename T> SOA_HOST_ONLY friend void dump();                                                                           \
                                                                                                                                    \
private:                                                                                                                            \
  /* Range checker conditional to the macro _DO_RANGECHECK */                                                                       \
  SOA_HOST_DEVICE_INLINE                                                                                                            \
  void rangeCheck(size_t index) const {                                                                                             \
    if constexpr (_DO_RANGECHECK) {                                                                                                 \
      if (index >= nElements_) {                                                                                                    \
        printf("In " #CLASS "::rangeCheck(): index out of range: %zu with nElements: %zu\n", index, nElements_);                    \
        assert(false);                                                                                                              \
      }                                                                                                                             \
    }                                                                                                                               \
  }                                                                                                                                 \
                                                                                                                                    \
  /* data members */                                                                                                                \
  std::byte* mem_;                                                                                                                  \
  size_t nElements_;                                                                                                                \
  size_t byteAlignment_;                                                                                                            \
  _ITERATE_ON_ALL(_DECLARE_SOA_DATA_MEMBER, ~, __VA_ARGS__)                                                                         \
}
