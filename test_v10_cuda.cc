

#include "soa_v10_cuda.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <cuda.h>
#include <functional>
/*
 * Unit tests skeleton based on DataFormats/Common/test/ (testDataFormatsCommon)
 */

#include "soa_v10_cuda.h"
#include "soa_v10.h"

CPPUNIT_TEST_SUITE_REGISTRATION(testSoA);


#define check(X) \
  do { std::cout << #X " is " << (X) << std::endl; } while(false)

// declare a statically-sized SoA, templated on the column size and (optional) alignment

void testSoA::initialTest() {
  std::cout << std::boolalpha;



  SoA::dump(10, 8);
  // CPPUNIT_ASSERT_EQUAL_MESSAGE("Size of SoA<1>", 
  //     3 * sizeof(double)
  //     + (sizeof(uint16_t) / sizeof(int32_t) + 1) * sizeof(int32_t) // Take into account the padding to align the next element
  //     + sizeof(int32_t)
  //     + 2 * sizeof(const char *), 
  //   sizeof(SoA<1>));
  //check(sizeof(SoA<1>));
  std::cout << std::endl;

  SoA::dump(10);
  SoA::dump(31);
  SoA::dump(32);
  std::cout << std::endl;

  SoA::dump(1, 64);
  SoA::dump(10, 64);
  SoA::dump(31, 64);
  SoA::dump(32, 64);
  std::cout << std::endl;

  SoA soa(new std::byte[SoA::computeDataSize(10,32)], 10, 32);
  check(& soa.z()[7] == & (soa[7].z));

  soa[7].x = 0.;
  soa[7].y = 3.1416;
  soa[7].z = -1.;
  soa[7].colour = 42;
  soa[7].value = 9999;

  soa[9] = soa[7];

  CPPUNIT_ASSERT(& soa.z()[7] == & (soa[7].z));
}



void testSoA::checkSoAAlignment(size_t nElements, size_t byteAlignment) {
  auto soaBlock = make_aligned_unique(SoA::computeDataSize(nElements,byteAlignment), byteAlignment);
  SoA soa(soaBlock.get(), nElements, byteAlignment);
  checkValuesAlignment(soa, &SoA::element::x, "x", byteAlignment);
  checkValuesAlignment(soa, &SoA::element::y, "y", byteAlignment);
  checkValuesAlignment(soa, &SoA::element::z, "z", byteAlignment);
  checkValuesAlignment(soa, &SoA::element::colour, "colour", byteAlignment);
  checkValuesAlignment(soa, &SoA::element::value, "value", byteAlignment);
  checkValuesAlignment(soa, &SoA::element::py, "py", byteAlignment);
}

void testSoA::checkAlignment() {
  checkSoAAlignment(1, 1);
  checkSoAAlignment(10, 1);
  checkSoAAlignment(31, 1);
  checkSoAAlignment(32, 1);

  checkSoAAlignment(1,64);
  checkSoAAlignment(10,64);
  checkSoAAlignment(31,64);
  checkSoAAlignment(32,64);
}
