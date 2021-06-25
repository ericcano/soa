#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <stdexcept>
#include "cppunit/extensions/HelperMacros.h"

/*
 * Unit tests skeleton based on DataFormats/Common/test/ (testDataFormatsCommon)
 */

#include "soa_v5.h"

class testSoA: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSoA);
  CPPUNIT_TEST(initialTest);
  CPPUNIT_TEST(checkAlignment);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void initialTest();
  void checkAlignment();
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

void testSoA::initialTest() {
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

template <typename SoA_t, typename T>
void checkValuesAlignment(SoA_t &soa, T SoA_t::element::*member, const std::string & memberName, size_t size, uint32_t bitAlignment) {
  if (bitAlignment % 8) CPPUNIT_FAIL("bitAlignment not byte aligned.");
  size_t byteAlignment = bitAlignment / 8;
  for (size_t i=0; i<size; i++) {
    // Check that each value is aligned
    if (reinterpret_cast<std::uintptr_t>(&(soa[i].*member)) % byteAlignment
            != (i * T::valueSize) %byteAlignment ) {
      std::stringstream err;
      err << "Misaligned value: " <<  memberName << " at index=" << i
              << " address=" << &(soa[i].*member) << " byteAlignment=" << byteAlignment
              << " address lower part: " << reinterpret_cast<std::uintptr_t>(&(soa[i].*member)) % byteAlignment
              << " expected address lower part: " << ((i * T::valueSize) % byteAlignment)
              << " size=" << SoA_t::size << " align=" << SoA_t::alignment;
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
              << " size=" << SoA_t::size << " align=" << SoA_t::alignment << " valueSize=" << T::valueSize;
      CPPUNIT_FAIL(err.str());
    }
  }
}

template <typename T>
void checkSoAAlignment(uint32_t bitAlignment) {
  T soa;
  checkValuesAlignment(soa, &T::element::x, "x", T::size, bitAlignment);
  checkValuesAlignment(soa, &T::element::y, "y", T::size, bitAlignment);
  checkValuesAlignment(soa, &T::element::z, "z", T::size, bitAlignment);
  checkValuesAlignment(soa, &T::element::colour, "colour", T::size, bitAlignment);
  checkValuesAlignment(soa, &T::element::value, "value", T::size, bitAlignment);
  checkValuesAlignment(soa, &T::element::name, "name", T::size, bitAlignment);
}
void testSoA::checkAlignment() {
  checkSoAAlignment<SoA<1>>(8);
  checkSoAAlignment<SoA<10>>(8);
  checkSoAAlignment<SoA<31>>(8);
  checkSoAAlignment<SoA<32>>(8);

  checkSoAAlignment<SoA<1,64>>(8*64);
  checkSoAAlignment<SoA<10,64>>(8*64);
  checkSoAAlignment<SoA<31,64>>(8*64);
  checkSoAAlignment<SoA<32,64>>(8*64);
}
