#pragma once

/* Common definitions shared by .cu and .cc files for the test */
#include "soa_v8.h"


#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <type_traits>
#include <cuda.h>
#include <cuda_runtime.h>

/* Test definitions */

class testSoA: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSoA);
  //CPPUNIT_TEST(initialTest);
  CPPUNIT_TEST(checkAlignment);
  CPPUNIT_TEST(fill);
  CPPUNIT_TEST(randomFill);
  CPPUNIT_TEST(crossProduct);
  CPPUNIT_TEST(randomCrossProduct);
  CPPUNIT_TEST(randomCrossProductEigen);
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
  
  declare_SoA_template(SoA,
    // predefined static scalars
    // size_t size;
    // size_t alignment;

    // columns: one value per element
    SoA_column(double, x),
    SoA_column(double, y),
    SoA_column(double, z),
    SoA_column(uint16_t, colour),
    SoA_column(int32_t, value),
    SoA_column(double *, py),

    // scalars: one value for the whole structure
    SoA_scalar(const char *, description)
  );
  
  // declare equivalent struct
  struct AoSelement {
    double x;
    double y;
    double z;
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

