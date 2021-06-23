/*
 * Structure-of-Arrays template with "columns" and "scalars", defined through preprocessor macros,
 * with compile-time size and alignment, and accessors to the "rows" and "columns".
 */

#include <iostream>
#include <type_traits>

#include "boost/preprocessor.hpp"

// CUDA attributes
#ifdef __CUDACC__
#define SOA_HOST_ONLY __host__
#define SOA_HOST_DEVICE __host__ __device__
#else
#define SOA_HOST_ONLY
#define SOA_HOST_DEVICE
#endif

// compile-time sized SoA

// Helper template managing the value within it column
template<typename T, size_t sz>
class SoAValue {
public:
  SoAValue(size_t i, T(& col)[sz]): idx_(i), col_(col) {}
  operator T&() { return col_[idx_]; }
  operator const T&() const { return col_[idx_]; }
  T* operator& () { return &col_[idx_]; }
  const T* operator& () const { return &col_[idx_]; }
  template <typename T2>
  T& operator= (const T2& v) { col_[idx_] = v; return col_[idx_]; }
  typedef T valueType;
  static constexpr auto valueSize = sizeof(T);
private:
  size_t idx_;
  T (&col_)[sz];
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

/* Element member declaration */
#define _DECLARE_ELEMENT_MEMBER_IMPL(IS_COLUMN, TYPE, NAME)\
  SoAValue<TYPE BOOST_PP_COMMA() SIZE> NAME;

#define _DECLARE_ELEMENT_MEMBER(R, DATA, ELEM)\
  _DECLARE_ELEMENT_MEMBER_IMPL ELEM

/* declare SoA data members; these should exapnd to, for columns:
 *
 *   alignas(ALIGN) double x_[SIZE];
 *
 * and for scalars:
 *
 *   double x_;
 *
 */

#define _DECLARE_SOA_DATA_MEMBER_IMPL(IS_COLUMN, TYPE, NAME)                                                                        \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    alignas(ALIGN) TYPE BOOST_PP_CAT(NAME, _[SIZE]);                                                                                \
  ,                                                                                                                                 \
    TYPE BOOST_PP_CAT(NAME, _);                                                                                                     \
  )

#define _DECLARE_SOA_DATA_MEMBER(R, DATA, TYPE_NAME)                                                                                \
  BOOST_PP_EXPAND(_DECLARE_SOA_DATA_MEMBER_IMPL TYPE_NAME)


/* declare SoA accessors; these should expand to, for columns:
 *
 *   double* x() { return x_; }
 *
 * and for scalars:
 *
 *   double& x() { return x_; }
 *
 */

#define _DECLARE_SOA_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                                           \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE* NAME() { return BOOST_PP_CAT(NAME, _); }                                                                                  \
  ,                                                                                                                                 \
    TYPE& NAME() { return BOOST_PP_CAT(NAME, _); }                                                                                  \
  )

#define _DECLARE_SOA_ACCESSOR(R, DATA, TYPE_NAME)                                                                                   \
  BOOST_PP_EXPAND(_DECLARE_SOA_ACCESSOR_IMPL TYPE_NAME)

#define _DECLARE_SOA_CONST_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                                     \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE const* NAME() const { return BOOST_PP_CAT(NAME, _); }                                                                      \
  ,                                                                                                                                 \
    TYPE const& NAME() const { return BOOST_PP_CAT(NAME, _); }                                                                      \
  )

#define _DECLARE_SOA_CONST_ACCESSOR(R, DATA, TYPE_NAME)                                                                             \
  BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ACCESSOR_IMPL TYPE_NAME)

/* assignment of individual fields; these should expand to, for columns
 *
 *   x() = other.x();
 *
 * and to nothing for scalars.
 */

#define _DECLARE_SOA_ELEMENT_ASSIGNMENT_IMPL(IS_COLUMN, TYPE, NAME)                                                                 \
    NAME = other.NAME;

#define _DECLARE_SOA_ELEMENT_ASSIGNMENT(R, DATA, TYPE_NAME)                                                                         \
  BOOST_PP_EXPAND(_DECLARE_SOA_ELEMENT_ASSIGNMENT_IMPL TYPE_NAME)

/* declare AoS-like element accessors; these should expand to, for columns:
 *
 *   double & x() { return * (soa_.x() + index_); }
 *
 * and for scalars:
 *
 *   double & x() { return soa_.x(); }
 *
 */

#define _DECLARE_SOA_ELEMENT_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                                   \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE & NAME() { return * (soa_. NAME () + index_); }                                                                            \
  ,                                                                                                                                 \
    TYPE & NAME() { return soa_. NAME (); }                                                                                         \
  )

#define _DECLARE_SOA_ELEMENT_ACCESSOR(R, DATA, TYPE_NAME)                                                                           \
  BOOST_PP_EXPAND(_DECLARE_SOA_ELEMENT_ACCESSOR_IMPL TYPE_NAME)

#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                             \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE const & NAME() { return * (soa_. NAME () + index_); }                                                                      \
  ,                                                                                                                                 \
    TYPE const & NAME() { return soa_. NAME (); }                                                                                   \
  )

#define _DECLARE_SOA_CONST_ELEMENT_ACCESSOR(R, DATA, TYPE_NAME)                                                                     \
  BOOST_PP_EXPAND(_DECLARE_SOA_CONST_ELEMENT_ACCESSOR_IMPL TYPE_NAME)


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
    std::cout << "  " BOOST_PP_STRINGIZE(NAME) "_[" << SoA::size << "] at "                                                         \
              << offsetof(SoA, SoA:: BOOST_PP_CAT(NAME, _)) << " has size " << sizeof(SoA:: BOOST_PP_CAT(NAME, _)) << std::endl;    \
  ,                                                                                                                                 \
    std::cout << "  " BOOST_PP_STRINGIZE(NAME) "_ at "                                                                              \
              << offsetof(SoA, SoA:: BOOST_PP_CAT(NAME, _)) << " has size " << sizeof(SoA:: BOOST_PP_CAT(NAME, _)) << std::endl;    \
  )

#define _DECLARE_SOA_DUMP_INFO(R, DATA, TYPE_NAME)                                                                                  \
  BOOST_PP_EXPAND(_DECLARE_SOA_DUMP_INFO_IMPL TYPE_NAME)

/* declare AoS-like element value aregs for contructor; these should expand,for columns only */
#define _DECLARE_ELEMENT_VALUE_ARG_IMPL(IS_COLUMN, TYPE, NAME)                                                                      \
  BOOST_PP_IIF(IS_COLUMN,  \
    (TYPE (&NAME)[SIZE])\
  ,\
    BOOST_PP_EMPTY()\
  )

#define _DECLARE_ELEMENT_VALUE_ARG(R, DATA, TYPE_NAME)                                                                              \
  BOOST_PP_EXPAND(_DECLARE_ELEMENT_VALUE_ARG_IMPL TYPE_NAME)

/* declare AoS-like element value args for contructor; these should expand,for columns only */
#define _DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION(R, DATA, TYPE_NAME)                                                            \
  (BOOST_PP_TUPLE_ELEM(2, TYPE_NAME)(DATA, BOOST_PP_TUPLE_ELEM(2, TYPE_NAME)))

/* declare AoS-like element value members; these should expand,for columns only */
/* We filter the value list beforehand to avoid having a comma inside a macro parameter */
#define _DECLARE_ELEMENT_VALUE_MEMBER_IMPL(IS_COLUMN, TYPE, NAME)                                                                   \
  SoAValue<TYPE BOOST_PP_COMMA() SIZE> NAME;

#define _DECLARE_ELEMENT_VALUE_MEMBER(R, DATA, TYPE_NAME)                                                                           \
  _DECLARE_ELEMENT_VALUE_MEMBER_IMPL TYPE_NAME

/* declare AoS-like element value members; these should expand,for columns only */
/* We filter the value list beforehand to avoid having a comma inside a macro parameter */
#define _DECLARE_ELEMENT_VALUE_COPY_IMPL(IS_COLUMN, TYPE, NAME)                                                                     \
  static_cast<TYPE &>(NAME) = static_cast<std::add_const<TYPE>::type &>(other.NAME);

#define _DECLARE_ELEMENT_VALUE_COPY(R, DATA, TYPE_NAME)                                                                             \
  _DECLARE_ELEMENT_VALUE_COPY_IMPL TYPE_NAME

/* declare AoS-like element value args for contructor; these should expand,for columns only */
#define _DECLARE_ELEMENT_CONSTR_CALL_IMPL(IS_COLUMN, TYPE, NAME)                                                                    \
  (BOOST_PP_CAT(NAME, _))

#define _DECLARE_ELEMENT_CONSTR_CALL(R, DATA, TYPE_NAME)                                                                            \
  BOOST_PP_EXPAND(_DECLARE_ELEMENT_CONSTR_CALL_IMPL TYPE_NAME)

#define _DECLARE_SOA_ELEMENT_ACCESSOR_IMPL(IS_COLUMN, TYPE, NAME)                                                                   \
  SOA_HOST_DEVICE                                                                                                                   \
  BOOST_PP_IIF(IS_COLUMN,                                                                                                           \
    TYPE & NAME() { return * (soa_. NAME () + index_); }                                                                            \
  ,                                                                                                                                 \
    TYPE & NAME() { return soa_. NAME (); }                                                                                         \
  )

#define _DECLARE_SOA_ELEMENT_ACCESSOR(R, DATA, TYPE_NAME)                                                                           \
  BOOST_PP_EXPAND(_DECLARE_SOA_ELEMENT_ACCESSOR_IMPL TYPE_NAME)

#define declare_SoA_template(CLASS, ...)                                                                                            \
                                                                                                                                    \
/* dump the SoA internaul structure */                                                                                              \
template <typename T>                                                                                                               \
SOA_HOST_ONLY                                                                                                                       \
void dump() {                                                                                                                       \
  using SoA = T;                                                                                                                    \
  std::cout << #CLASS "<" << SoA::size << ", " << SoA::alignment << ">: " << '\n';                                                  \
  std::cout << "  sizeof(...): " << sizeof(CLASS) << '\n';                                                                          \
  std::cout << "  alignof(...): " << alignof(CLASS) << '\n';                                                                        \
  _ITERATE_ON_ALL(_DECLARE_SOA_DUMP_INFO, ~, __VA_ARGS__)                                                                           \
  std::cout << std::endl;                                                                                                           \
}                                                                                                                                   \
                                                                                                                                    \
template <size_t SIZE, size_t ALIGN=0>                                                                                              \
struct CLASS {                                                                                                                      \
                                                                                                                                    \
  /* these could be moved to an external type trait to free up the symbol names */                                                  \
  using self_type = CLASS;                                                                                                          \
  static const size_t size = SIZE;                                                                                                  \
  static const size_t alignment = ALIGN;                                                                                            \
                                                                                                                                    \
  /* AoS-like accessor to individual elements */                                                                                    \
  struct const_element {                                                                                                            \
    SOA_HOST_DEVICE                                                                                                                 \
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
/*  struct element {                                                                                                                \
    SOA_HOST_DEVICE                                                                                                                 \
    element(CLASS & soa, size_t index) :                                                                                            \
      soa_(soa),                                                                                                                    \
      index_(index)                                                                                                                 \
    { }                                                                                                                             \
                                                                                                                                    \
    SOA_HOST_DEVICE                                                                                                                 \
    element& operator=(element const& other) {                                                                                      \
      _ITERATE_ON_VALUE_TYPE(_DECLARE_SOA_ELEMENT_ASSIGNMENT, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__)                                   \
      return *this;                                                                                                                 \
    }                                                                                                                               \
                                                                                                                                    \
    SOA_HOST_DEVICE                                                                                                                 \
    element& operator=(element && other) {                                                                                          \
      _ITERATE_ON_VALUE_TYPE(_DECLARE_SOA_ELEMENT_ASSIGNMENT, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__)                                   \
      return *this;                                                                                                                 \
    }                                                                                                                               \
                                                                                                                                    \
    SOA_HOST_DEVICE                                                                                                                 \
    element& operator=(const_element const& other) {                                                                                \
      _ITERATE_ON_VALUE_TYPE(_DECLARE_SOA_ELEMENT_ASSIGNMENT, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__)                                   \
      return *this;                                                                                                                 \
    }                                                                                                                               \
                                                                                                                                    \
    SOA_HOST_DEVICE                                                                                                                 \
    element& operator=(const_element && other) {                                                                                    \
      _ITERATE_ON_VALUE_TYPE(_DECLARE_SOA_ELEMENT_ASSIGNMENT, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__)                                   \
      return *this;                                                                                                                 \
    }                                                                                                                               \
                                                                                                                                    \
    _ITERATE_ON_ALL(_DECLARE_SOA_CONST_ELEMENT_ACCESSOR, ~, __VA_ARGS__)                                                            \
                                                                                                                                    \
  private:                                                                                                                          \
    CLASS & soa_;                                                                                                                   \
    const size_t index_;                                                                                                            \
  };*/                                                                                                                              \
  struct element {                                                                                                                  \
    element(size_t index,                                                                                                           \
      /* Turn Boost PP */                                                                                                           \
      _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_VALUE_ARG, index, _VALUE_TYPE_COLUMN, __VA_ARGS__)                              \
    ):                                                                                                                              \
      _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_VALUE_MEMBER_INITIALISATION, index, _VALUE_TYPE_COLUMN, __VA_ARGS__)            \
       {}                                                                                                                           \
    element& operator=(const element& other) {                                                                                      \
      _ITERATE_ON_VALUE_TYPE(_DECLARE_ELEMENT_VALUE_COPY, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__)                                       \
      return *this;                                                                                                                 \
    }                                                                                                                               \
    _ITERATE_ON_VALUE_TYPE(_DECLARE_ELEMENT_VALUE_MEMBER, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__)                                       \
  };                                                                                                                                \
                                                                                                                                    \
  /* AoS-like accessor */                                                                                                           \
  SOA_HOST_DEVICE                                                                                                                   \
  element operator[](size_t index) { return element(index,                                                                          \
    _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_CONSTR_CALL, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__) );                               \
    }                                                                                                                               \
  const element operator[](size_t index) const { return element(index,                                                              \
    _ITERATE_ON_VALUE_TYPE_COMMA(_DECLARE_ELEMENT_CONSTR_CALL, ~, _VALUE_TYPE_COLUMN, __VA_ARGS__) );                               \
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
  /* data members */                                                                                                                \
  _ITERATE_ON_ALL(_DECLARE_SOA_DATA_MEMBER, ~, __VA_ARGS__)                                                                         \
}
