#include <limits>

#include "soa_v7_cuda.h"

#include <curand_kernel.h>

#define CUDA_UNIT_CHECK(A) CPPUNIT_ASSERT_EQUAL(cudaSuccess, A)

namespace {
  // fill element
  template <class T>
  __host__ __device__ __forceinline__ void fillElement(T & e, size_t i) {
    e.x = 11.0 * i;
    e.y = 22.0 * i;
    e.z = 33.0 * i;
    e.colour = i;
    e.value = 0x10001 * i;
    e.py = &e.y;
  }

  // Fill up the elements of the SoA
  __global__ void fillSoA(testSoA::SoA soa) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= soa.nElements()) return;
    // compiler does not belive we can use a temporary soa[i] to store results.
    // So make an lvalue.
    auto e = soa[i];
    fillElement(e, i);
  }
  
  // Fill elements with random data.
  __global__ void randomFillSoA(testSoA::SoA soa, uint64_t seed) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= soa.nElements()) return;
    curandState state;
    curand_init(seed, i, 0, &state);
    soa[i].x = curand_uniform_double(&state);
    soa[i].y = curand_uniform_double(&state);
    soa[i].z = curand_uniform_double(&state);
  }

  __global__ void fillAoS(testSoA::AoSelement *aos, size_t nElements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElements) return;
    fillElement(aos[i], i);
  }

  // Simple cross product for elements
  template <typename T, typename T2>
  __host__ __device__ __forceinline__ void crossProduct(T & r, const T2 & a, const T2 & b) {
    r.x = a.y * b.z - a.z * b.y;
    r.y = a.z * b.x - a.x * b.z;
    r.z = a.x * b.y - a.y * b.x;
  }

  // Simple cross product (SoA)
  __global__ void indirectCrossProductSoA(testSoA::SoA r, const testSoA::SoA a, const testSoA::SoA b) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= min(r.nElements(), min(a.nElements(), b.nElements()))) return;
    // C++ does not allow creating non-const references to temporary variables
    // this workaround makes the temporary variable 
    auto ri = r[i];
    // Print addresses for a few samples
    if (not i%10) {
      // TODO: the output of this is fishy, we expect equality. The rest seems to work though (ongoing).
      printf ("i=%zd &r[i].x=%p &ri.x=%p\n", i, &r[i].x, &ri.x);
    }
    crossProduct(ri, a[i], b[i]);
  }

  // Simple cross product (SoA on CPU)
  __host__ void indirectCPUcrossProductSoA(testSoA::SoA r, const testSoA::SoA a, const testSoA::SoA b) {
    for (size_t i =0; i< min(r.nElements(), min(a.nElements(), b.nElements())); ++i) {
      // This version is also affected.
      auto ri = r[i];
      crossProduct(ri, a[i], b[i]);
    }
  }

  // Simple cross product (AoS)
  __global__ void crossProductAoS(testSoA::AoSelement *r,
          testSoA::AoSelement *a, testSoA::AoSelement *b, size_t nElements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElements) return;
    crossProduct(r[i], a[i], b[i]);
  }

  void hexdump(void *ptr, int buflen) {
    /* From https://stackoverflow.com/a/29865 with adaptations */
    unsigned char *buf = (unsigned char*)ptr;
    int i, j;
    for (i=0; i<buflen; i+=16) {
      printf("%06x: ", i);
      for (j=0; j<16; j++) {
        if (i+j < buflen)
          printf("%02x ", buf[i+j]);
        else
          printf("   ");
        if ((i+j) % 4 == 3) printf (" ");
      }
      printf(" ");
  //  for (j=0; j<16; j++)
  //    if (i+j < buflen)
  //      printf("%c", isprint(buf[i+j]) ? buf[i+j] : '.');
      printf("\n");
    }
  }
  // Check we find what we wanted to initialize.
  // Pass should be initialized to true.
  template <class T>
  __host__ __device__ __forceinline__ void checkSoAelement(T soa, size_t i, bool & pass) {
    if (i >= soa.nElements() || !pass) return;
    if (soa[i].x != 11.0 * i) { pass = false; return; }
    if (soa[i].y != 22.0 * i) { pass = false; return; }
    if (soa[i].z != 33.0 * i) { pass = false; return; }
    if (soa[i].colour != i) { pass = false; return; }
    if (soa[i].value != static_cast<int32_t>(0x10001 * i)) { pass = false; return; }
  }

  // Check r[i].{x, y, z} are close enough to zero compared to a[i].{x,y,z} and b[i].{x,y,z}
  // to validate a cross product of a vector with itself produced a zero (enough) result.
  template <class T>
  __host__ __device__ __forceinline__ void checkSoAelementNullityRealtiveToSquare(T resSoA, T referenceSoA, size_t i, double epsilon, bool & pass) {
    if (i >= resSoA.nElements() || !pass) return;
    auto ref = max (abs(referenceSoA[i].x), 
                    max(abs(referenceSoA[i].y), 
                        abs(referenceSoA[i].z)));
    ref *= ref * epsilon;
    if (abs(resSoA[i].x) > ref) { pass = false; return; }
    if (abs(resSoA[i].y) > ref) { pass = false; return; }
    if (abs(resSoA[i].z) > ref) { pass = false; return; }    
  }

  // Check r[i].{x, y, z} are close enough to zero compared to a[i].{x,y,z} and b[i].{x,y,z}
  // to validate a cross product of a vector with itself produced a zero (enough) result.
  template <class T>
  __host__ __device__ __forceinline__ void checkCrossProduct(T resultSoA, T aSoA, T bSoA, size_t i, double epsilon, bool & pass) {
    if (i >= resultSoA.nElements() || !pass) return;
    auto refA = max (abs(aSoA[i].x), 
                     max(abs(aSoA[i].y), 
                         abs(aSoA[i].z)));
    auto refB = max (abs(bSoA[i].x), 
                     max(abs(bSoA[i].y), 
                         abs(bSoA[i].z)));
    auto ref = max(refA, refB);
    ref *= ref * epsilon;
    testSoA::AoSelement myRes;
    crossProduct(myRes, aSoA[i], bSoA[i]);
    if (abs(myRes.x - resultSoA[i].x) > ref) { pass = false; return; }
    if (abs(myRes.y - resultSoA[i].y) > ref) { pass = false; return; }
    if (abs(myRes.z - resultSoA[i].z) > ref) { pass = false; return; }
  }
}

void testSoA::fill() {
  // Get device, stream, memory
  cudaDeviceProp deviceProperties;
  int deviceCount=0;
  CUDA_UNIT_CHECK(cudaGetDeviceCount(&deviceCount));
  CPPUNIT_ASSERT(deviceCount > 0);
  CUDA_UNIT_CHECK(cudaGetDeviceProperties(&deviceProperties, defaultDevice));
  CUDA_UNIT_CHECK(cudaSetDevice(defaultDevice));  
  cudaStream_t stream;
  auto e = cudaStreamCreate(&stream);
  CUDA_UNIT_CHECK(cudaStreamCreate(&stream));
  
  // Allocate memory and populate SoA descriptors
  auto deviceSoABlock = make_device_unique(SoA::computeDataSize(elementsCount));
  auto hostSoABlock = make_host_unique(SoA::computeDataSize(elementsCount));
  SoA deviceSoA(deviceSoABlock.get(), elementsCount);
  SoA hostSoA(hostSoABlock.get(), elementsCount);
  
  // Call kernel, get result
  fillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoA);
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlock.get(), deviceSoABlock.get(), SoA::computeDataSize(hostSoA.nElements()), cudaMemcpyDeviceToHost, stream));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(stream));

  // Validate result
  bool pass = true;
  size_t i = 0;
  for (; pass && i< hostSoA.nElements(); i++) checkSoAelement(hostSoA, i, pass);
  if (!pass) {
    std::cout << "In " << typeid(*this).name() << " check failed at i= " << i << ")" << std::endl;
    hexdump(hostSoABlock.get(), SoA::computeDataSize(hostSoA.nElements()));
    printf("base=%p, &y=%p\n", deviceSoABlock.get(), deviceSoA.y());
  }
  CPPUNIT_ASSERT(pass);
}

void testSoA::randomFill() {
  // Get device, stream, memory
  cudaDeviceProp deviceProperties;
  int deviceCount=0;
  CUDA_UNIT_CHECK(cudaGetDeviceCount(&deviceCount));
  CPPUNIT_ASSERT(deviceCount > 0);
  CUDA_UNIT_CHECK(cudaGetDeviceProperties(&deviceProperties, defaultDevice));
  CUDA_UNIT_CHECK(cudaSetDevice(defaultDevice));  
  cudaStream_t stream;
  auto e = cudaStreamCreate(&stream);
  CUDA_UNIT_CHECK(cudaStreamCreate(&stream));
  
  // Allocate memory and populate SoA descriptors
  auto deviceSoABlock = make_device_unique(SoA::computeDataSize(elementsCount));
  auto hostSoABlock = make_host_unique(SoA::computeDataSize(elementsCount));
  SoA deviceSoA(deviceSoABlock.get(), elementsCount);
  SoA hostSoA(hostSoABlock.get(), elementsCount);
  
  // Call kernel, get result
  randomFillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoA, 0xbaddeed5);
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlock.get(), deviceSoABlock.get(), SoA::computeDataSize(hostSoA.nElements()), cudaMemcpyDeviceToHost, stream));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(stream));
}

void testSoA::crossProduct() {
  // Get device, stream, memory
  cudaDeviceProp deviceProperties;
  int deviceCount=0;
  CUDA_UNIT_CHECK(cudaGetDeviceCount(&deviceCount));
  CPPUNIT_ASSERT(deviceCount > 0);
  CUDA_UNIT_CHECK(cudaGetDeviceProperties(&deviceProperties, defaultDevice));
  cudaStream_t stream;
  CUDA_UNIT_CHECK(cudaStreamCreate(&stream));
  
  // Allocate memory and populate SoA descriptors (device A as source and R as result of cross product)
  auto deviceSoABlockA = make_device_unique(SoA::computeDataSize(elementsCount));
  auto deviceSoABlockR = make_device_unique(SoA::computeDataSize(elementsCount));
  auto hostSoABlockA = make_host_unique(SoA::computeDataSize(elementsCount));
  auto hostSoABlockR = make_host_unique(SoA::computeDataSize(elementsCount));
  SoA deviceSoAA(deviceSoABlockA.get(), elementsCount);
  SoA deviceSoAR(deviceSoABlockR.get(), elementsCount);
  SoA hostSoAA(hostSoABlockA.get(), elementsCount);
  SoA hostSoAR(hostSoABlockR.get(), elementsCount);
  
  // Call kernels, get result. Also fill up result SoA to ensure the results go in the right place.
  fillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoAA);
  fillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoAR);
  indirectCrossProductSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoAR, deviceSoAA, deviceSoAA);
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockA.get(), deviceSoABlockA.get(), SoA::computeDataSize(hostSoAA.nElements()), cudaMemcpyDeviceToHost, stream));
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockR.get(), deviceSoABlockR.get(), SoA::computeDataSize(hostSoAR.nElements()), cudaMemcpyDeviceToHost, stream));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(stream));

  // Validate result
  bool pass = true;
  size_t i = 0;
  for (; pass && i< hostSoAR.nElements(); i++)
    checkSoAelementNullityRealtiveToSquare(hostSoAR, hostSoAA, i, std::numeric_limits<double>::epsilon(), pass);
  if (!pass) {
    std::cout << "In " << typeid(*this).name() << " check failed at i= " << i << ")" << std::endl;
    std::cout << "result[" << i << "].x=" << hostSoAR[i].x << " .y=" << hostSoAR[i].y << " .z=" << hostSoAR[i].z << std::endl;
  } else {
    std::cout << std::endl;
    for (size_t j=0; j<10 ; ++j) {
      std::cout << "result[" << j << "]].x=" << hostSoAR[j].x << " .y=" << hostSoAR[j].y << " .z=" << hostSoAR[j].z << std::endl;
      std::cout << "A[" << j << "]].x=" << hostSoAA[j].x << " .y=" << hostSoAA[j].y << " .z=" << hostSoAA[j].z << std::endl;
    }
  }
  CPPUNIT_ASSERT(pass);
}

void testSoA::randomCrossProduct() {
  // Get device, stream, memory
  cudaDeviceProp deviceProperties;
  int deviceCount=0;
  CUDA_UNIT_CHECK(cudaGetDeviceCount(&deviceCount));
  CPPUNIT_ASSERT(deviceCount > 0);
  CUDA_UNIT_CHECK(cudaGetDeviceProperties(&deviceProperties, defaultDevice));
  cudaStream_t stream;
  CUDA_UNIT_CHECK(cudaStreamCreate(&stream));
  
  // Allocate memory and populate SoA descriptors (device A as source and R as result of cross product)
  auto deviceSoABlockA = make_device_unique(SoA::computeDataSize(elementsCount));
  auto deviceSoABlockB = make_device_unique(SoA::computeDataSize(elementsCount));
  auto deviceSoABlockR = make_device_unique(SoA::computeDataSize(elementsCount));
  auto hostSoABlockA = make_host_unique(SoA::computeDataSize(elementsCount));
  auto hostSoABlockB = make_host_unique(SoA::computeDataSize(elementsCount));
  auto hostSoABlockR = make_host_unique(SoA::computeDataSize(elementsCount));
  SoA deviceSoAA(deviceSoABlockA.get(), elementsCount);
  SoA deviceSoAB(deviceSoABlockB.get(), elementsCount);
  SoA deviceSoAR(deviceSoABlockR.get(), elementsCount);
  SoA hostSoAA(hostSoABlockA.get(), elementsCount);
  SoA hostSoAB(hostSoABlockB.get(), elementsCount);
  SoA hostSoAR(hostSoABlockR.get(), elementsCount);
  
  // Call kernels, get result. Also fill up result SoA to ensure the results go in the right place.
  randomFillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoAA, 0xdeadbeef);
  randomFillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoAB, 0xcafefade);
  randomFillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoAR, 0xfadedcab);
  indirectCrossProductSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoAR, deviceSoAA, deviceSoAB);
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockA.get(), deviceSoABlockA.get(), SoA::computeDataSize(hostSoAA.nElements()), cudaMemcpyDeviceToHost, stream));
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockB.get(), deviceSoABlockB.get(), SoA::computeDataSize(hostSoAA.nElements()), cudaMemcpyDeviceToHost, stream));
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockR.get(), deviceSoABlockR.get(), SoA::computeDataSize(hostSoAR.nElements()), cudaMemcpyDeviceToHost, stream));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(stream));

  // Validate result
  bool pass = true;
  size_t i = 0;
  for (; pass && i< hostSoAR.nElements(); i++) {
    checkCrossProduct(hostSoAR, hostSoAA, hostSoAB, i, std::numeric_limits<double>::epsilon(), pass);
  }
  if (!pass) {
    // Recompute the expected result
    testSoA::AoSelement expected;
    ::crossProduct(expected, hostSoAA[i], hostSoAB[i]);
    std::cout << "In " << __FUNCTION__ << " check failed at i= " << i << std::endl;
    std::cout << "result= ("   << hostSoAR[i].x << ", " << hostSoAR[i].y << ", " << hostSoAR[i].z << ")" << std::endl;
    std::cout << "expected= (" << expected.x << ", " << expected.y << ", " << expected.z << ")" << std::endl;
    std::cout << "A= (" << hostSoAA[i].x << ", " << hostSoAA[i].y << ", " << hostSoAA[i].z << ")" << std::endl;
    std::cout << "B= (" << hostSoAB[i].x << ", " << hostSoAB[i].y << ", " << hostSoAB[i].z << ")" << std::endl;
  } else {
    std::cout << std::endl;
    for (size_t j=0; j<10 && j<hostSoAR.nElements(); ++j) {
      testSoA::AoSelement expected;
      // Mixed computation AoS(single row) / SoA / SoA 
      ::crossProduct(expected, hostSoAA[j], hostSoAB[j]);
      std::cout << "In " << __FUNCTION__ << " check was OK. Sampling j= " << j << std::endl;
      std::cout << "result= ("   << hostSoAR[j].x << ", " << hostSoAR[j].y << ", " << hostSoAR[j].z << ")" << std::endl;
      std::cout << "expected= (" << expected.x << ", " << expected.y << ", " << expected.z << ")" << std::endl;
      std::cout << "difference= (" << expected.x - hostSoAR[j].x << ", " << expected.y - hostSoAR[j].y  
              << ", " << expected.z - hostSoAR[j].z << ")" << std::endl;
      std::cout << "A= (" << hostSoAA[j].x << ", " << hostSoAA[j].y << ", " << hostSoAA[j].z << ")" << std::endl;
      std::cout << "B= (" << hostSoAB[j].x << ", " << hostSoAB[j].y << ", " << hostSoAB[j].z << ")" << std::endl;
    }
  }
  CPPUNIT_ASSERT(pass);
}