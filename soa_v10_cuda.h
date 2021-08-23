#pragma once

/* Common definitions shared by .cu and .cc files for the test */
#include "soa_v10.h"


#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <type_traits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Core>

/* Test definitions */

class testSoA: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSoA);
  CPPUNIT_TEST(initialTest);
  CPPUNIT_TEST(checkAlignment);
  CPPUNIT_TEST(fill);
  CPPUNIT_TEST(randomFill);
  CPPUNIT_TEST(crossProduct);
  CPPUNIT_TEST(randomCrossProduct);
  CPPUNIT_TEST(randomCrossProductEigen);
  CPPUNIT_TEST(randomCrossProductEmbeddedVector);
  CPPUNIT_TEST_SUITE_END();

/* SoA and AoS structures definitions */  
public:
  void setUp() {}
  void tearDown() {}
  void initialTest();
  void checkAlignment();
  void fill();
  void randomFill();
  void crossProduct();
  void randomCrossProduct();
  void randomCrossProductEigen();
  void randomCrossProductEmbeddedVector();
  
  declare_SoA_template(SoA,
    // predefined static scalars
    // size_t size;
    // size_t alignment;

    // columns: one value per element
    SoA_FundamentalTypeColumn(double, x),
    SoA_FundamentalTypeColumn(double, y),
    SoA_FundamentalTypeColumn(double, z),
    SoA_eigenColumn(Eigen::Vector3d, a),
    SoA_eigenColumn(Eigen::Vector3d, b),
    SoA_eigenColumn(Eigen::Vector3d, r),
    SoA_column(uint16_t, colour),
    SoA_column(int32_t, value),
    SoA_column(double *, py),
    SoA_FundamentalTypeColumn(uint32_t, count),
    SoA_FundamentalTypeColumn(uint32_t, anotherCount),

    // scalars: one value for the whole structure
    SoA_scalar(const char *, description),
    SoA_scalar(uint32_t, someNumber)
  );
  
  // declare equivalent struct
  struct AoSelement {
    double x;
    double y;
    double z;
    Eigen::Vector3d a;
    Eigen::Vector3d b;
    Eigen::Vector3d r;
    uint16_t colour;
    int32_t value;
    double * py;
  };
  
private:
  // Constants
  static constexpr int defaultDevice = 0;
  static constexpr size_t elementsCount = 10000;
  
  // Helper functions
  template <typename T>
  void checkValuesAlignment(SoA &soa, T SoA::element::*member, const std::string & memberName, size_t byteAlignment) {
    for (size_t i=0; i<soa.nElements(); i++) {
      // Check that each value is aligned
      if (reinterpret_cast<std::uintptr_t>(&(soa[i].*member)) % byteAlignment
              != (i * T::valueSize) %byteAlignment ) {
        std::stringstream err;
        err << "Misaligned value: " <<  memberName << " at index=" << i
                << " address=" << &(soa[i].*member) << " byteAlignment=" << byteAlignment
                << " address lower part: " << reinterpret_cast<std::uintptr_t>(&(soa[i].*member)) % byteAlignment
                << " expected address lower part: " << ((i * T::valueSize) % byteAlignment)
                << " size=" << soa.nElements() << " align=" << soa.byteAlignment();
        CPPUNIT_FAIL(err.str());
      }
      // Check that all values except the first-in rows (address 0 modulo alignment)
      // are contiguous to their predecessors in memory (this will detect cutting
      // memory/cache/etc... lines in unexpected places (for blocked SoA like AoSoA)
      if ((reinterpret_cast<std::uintptr_t>(&(soa[i].*member)) % byteAlignment)
           && (reinterpret_cast<std::uintptr_t>(&(soa[i - 1].*member)) + T::valueSize
                != reinterpret_cast<std::uintptr_t>(&(soa[i].*member)))) {
        std::stringstream err;
        err << "Unexpected non-contiguity: " <<  memberName << " at index=" << i
                << " address=" << &(soa[i].*member) << " is not contiguous to "
                << memberName << " at index=" << i - 1 << "address=" << &(soa[i - 1].*member)
                << " size=" << soa.nElements() << " align=" << soa.byteAlignment() << " valueSize=" << T::valueSize;
        CPPUNIT_FAIL(err.str());
      }
    }
  }
  
  // Helper macro to check alignement of reference-like members
#define CHECK_REFERENCED_VALUE_ALIGNMENT(SOA, MEMBER, BYTE_ALIGMENT)                                                                \
  {                                                                                                                                 \
    for (size_t i=0; i<SOA.nElements(); i++) {                                                                                      \
      /* Check that each value is aligned */                                                                                        \
      if (reinterpret_cast<std::uintptr_t>(&(SOA[i].MEMBER)) % byteAlignment                                                        \
              != (i *sizeof(decltype(SOA[i].MEMBER))) %byteAlignment ) {                                                            \
        std::stringstream err;                                                                                                      \
        err << "Misaligned value: " <<  BOOST_PP_STRINGIZE(MEMBER) << " at index=" << i                                             \
                << " address=" << &(soa[i].MEMBER) << " byteAlignment=" << byteAlignment                                            \
                << " address lower part: " << reinterpret_cast<std::uintptr_t>(&(SOA[i].MEMBER)) % byteAlignment                    \
                << " expected address lower part: " << ((i * sizeof(decltype(SOA[i].MEMBER))) % byteAlignment)                      \
                << " size=" << SOA.nElements() << " align=" << SOA.byteAlignment();                                                 \
        CPPUNIT_FAIL(err.str());                                                                                                    \
      }                                                                                                                             \
      /* Check that all values except the first-in rows (address 0 modulo alignment)                                                \
         are contiguous to their predecessors in memory (this will detect cutting                                                   \
         memory/cache/etc... lines in unexpected places (for blocked SoA like AoSoA)*/                                              \
      if ((reinterpret_cast<std::uintptr_t>(&(SOA[i].MEMBER)) % byteAlignment)                                                      \
           && (reinterpret_cast<std::uintptr_t>(&(SOA[i - 1].MEMBER)) + sizeof(decltype(SOA[i].MEMBER))                             \
                != reinterpret_cast<std::uintptr_t>(&(SOA[i].MEMBER)))) {                                                           \
        std::stringstream err;                                                                                                      \
        err << "Unexpected non-contiguity: " <<  BOOST_PP_STRINGIZE(MEMBER) << " at index=" << i                                    \
                << " address=" << &(SOA[i].MEMBER) << " is not contiguous to "                                                      \
                << BOOST_PP_STRINGIZE(MEMBER) << " at index=" << i - 1 << "address=" << &(SOA[i - 1].MEMBER)                        \
                << " size=" << SOA.nElements() << " align=" << SOA.byteAlignment()                                                  \
                << " valueSize=" << sizeof(decltype(SOA[i].MEMBER));                                                                \
        CPPUNIT_FAIL(err.str());                                                                                                    \
      }                                                                                                                             \
    }                                                                                                                               \
  }
  
  void checkSoAAlignment(size_t nElements, size_t byteAlignment);
  
  std::unique_ptr<std::byte, std::function<void(void*)>> make_aligned_unique(size_t size, size_t alignment) {
    return std::unique_ptr<std::byte, std::function<void(void*)>> (
            static_cast<std::byte*>(std::aligned_alloc(size, alignment)), [](void*p){std::free(p);});
  }
  
  class bad_alloc: public std::bad_alloc {
  public:
    bad_alloc(const std::string& w) noexcept: what_(w) {}
    const char* what() const noexcept override { return what_.c_str(); } 
  private:
    const std::string what_;
  };
  
  std::unique_ptr<std::byte, std::function<void(void*)>> make_device_unique(size_t size) {
    void *p = nullptr;
    cudaError_t e = cudaMalloc(&p, size);
    if (e != cudaSuccess) {
      std::string m("Failed to allocate device memory: ");
      m+= cudaGetErrorName(e);
      [[unlikely]] throw bad_alloc(m);
    }
    return std::unique_ptr<std::byte, std::function<void(void*)>> (
            static_cast<std::byte*>(p), [](void*p){cudaFree(p);});
  }
  
  std::unique_ptr<std::byte, std::function<void(void*)>> make_host_unique(size_t size) {
    void *p = nullptr;
    cudaError_t e = cudaMallocHost(&p, size);
    if (e != cudaSuccess) {
      std::string m("Failed to allocate page-locked host memory: ");
      m+= cudaGetErrorName(e);
      [[unlikely]] throw bad_alloc(m);
    }
    return std::unique_ptr<std::byte, std::function<void(void*)>> (
            static_cast<std::byte*>(p), [](void*p){cudaFreeHost(p);});
  }  
};

/* SoA and AoS structures definitions */

