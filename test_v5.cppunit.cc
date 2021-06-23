#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include "cppunit/extensions/HelperMacros.h"

/*
 * Unit tests skeleton based on DataFormats/Common/test/ (testDataFormatsCommon)
 */

#include "soa_v5.h"

class testSoA: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSoA);
  CPPUNIT_TEST(initialTest);
  CPPUNIT_TEST_SUITE_END();
  
public:
  void setUp() {}
  void tearDown() {}
  void initialTest();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testSoA);


#define check(X) \
  do { std::cout << #X " is " << (X) << std::endl; } while(false)

// declare a statically-sized SoA, templated on the column size and (optional) alignment

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
  SoA_column(const char *, name),

  // scalars: one value for the whole structure
  SoA_scalar(const char *, description)
);

using LargeSoA = SoA<1024>;

void testSoA::initialTest(void) {
  std::cout << std::boolalpha;



  dump<SoA<1>>();
  CPPUNIT_ASSERT_EQUAL_MESSAGE("Size of SoA<1>", 
      3 * sizeof(double) 
      + (sizeof(uint16_t) / sizeof(int32_t) + 1) * sizeof(int32_t) // Take into account the padding to align the next element
      + sizeof(int32_t)
      + 2 * sizeof(const char *), 
    sizeof(SoA<1>));
  //check(sizeof(SoA<1>));
  std::cout << std::endl;

  dump<SoA<10>>();
  dump<SoA<31>>();
  dump<SoA<32>>();
  std::cout << std::endl;

  dump<SoA<1, 64>>();
  dump<SoA<10, 64>>();
  dump<SoA<31, 64>>();
  dump<SoA<32, 64>>();
  std::cout << std::endl;

  SoA<10, 32> soa;
  check(& soa.z()[7] == & (soa[7].z));

  soa[7].x = 0.;
  soa[7].y = 3.1416;
  soa[7].z = -1.;
  soa[7].colour = 42;
  soa[7].value = 9999;
  soa[7].name = "element";

  soa[9] = soa[7];

  CPPUNIT_ASSERT(& soa.z()[7] == & (soa[7].z));
}
