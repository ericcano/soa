#include <limits>

#include "soa_v9_cuda.h"

#include <curand_kernel.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

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
  [[maybe_unused]] __global__ void fillSoA(testSoA::SoA soa) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= soa.nElements()) return;
    // compiler does not belive we can use a temporary soa[i] to store results.
    // So make an lvalue.
    auto e = soa[i];
    fillElement(e, i);
  }
  // A read only decorator adding x,y,z interface to Eigen vectors
  /*template <class C>
  class EigenXYZW {
  public:
    EigenXYZW(const C &v): x(v, &C::x()), y(v, &C::y()), z(v, &C::z()), w(v, &C::w())  {}
    class Accessor {
      friend EigenXYZW<C>;
    public:
      operator C::Scalar() const { return (v_.*m_)(); }
    private:
      Accessor(const C& v, C::Scalar (C::Type::*m)()): v_(v), m_(m) {}
      const C & v_;
      C::Scalar (C::Type::*m_)();
    };
    Accessor x;
    Accessor y;
    Accessor z;
    Accessor w;
  };*/
  
  // Fill elements with random data.
  [[maybe_unused]] __global__ void randomFillSoA(testSoA::SoA soa, uint64_t seed) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= soa.nElements()) return;
    curandState state;
    curand_init(seed + i, 0, 0, &state);
    soa[i].x = curand_uniform_double(&state);
    soa[i].y = curand_uniform_double(&state);
    soa[i].z = curand_uniform_double(&state);
    soa[i].a()(0) = curand_uniform_double(&state);
    soa[i].a()(1) = curand_uniform_double(&state);
    soa[i].a()(2) = curand_uniform_double(&state);
    soa[i].b()(0) = curand_uniform_double(&state);
    soa[i].b()(1) = curand_uniform_double(&state);
    soa[i].b()(2) = curand_uniform_double(&state);
  }

  [[maybe_unused]] __global__ void fillAoS(testSoA::AoSelement *aos, size_t nElements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElements) return;
    fillElement(aos[i], i);
  }
  
  // Simple cross product for elements
  template <typename T, typename T2>
  [[maybe_unused]] __host__ __device__ __forceinline__ void crossProduct(T & r, const T2 & __restrict__ a, const T2 & __restrict__ b) {
    r.x = a.y * b.z - a.z * b.y;
    r.y = a.z * b.x - a.x * b.z;
    r.z = a.x * b.y - a.y * b.x;
  }

  // Simple indirect cross product (SoA)
  [[maybe_unused]] __global__ void indirectCrossProductSoA(testSoA::SoA r, const testSoA::SoA a, const testSoA::SoA b, size_t nElements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElements) return;
    // C++ does not allow creating non-const references to temporary variables
    // this workaround makes the temporary variable 
    auto ri = r[i];
    crossProduct(ri, a[i], b[i]);
  }

  // Simple direct cross product (SoA)
  [[maybe_unused]] __global__ void directCrossProductSoA(testSoA::SoA r, const testSoA::SoA a, const testSoA::SoA b, size_t nElements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElements) return;
    r[i].x = a[i].y * b[i].z - a[i].z * b[i].y;
    r[i].y = a[i].z * b[i].x - a[i].x * b[i].z;
    r[i].z = a[i].x * b[i].y - a[i].y * b[i].x;
  }

  // Hand-made cross product as a reference (SoA)
  [[maybe_unused]] __global__ void handcraftedCrossProductSoA(testSoA::SoA r, const testSoA::SoA a, const testSoA::SoA b, size_t nElements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElements) return;
    r.x()[i] = a.y()[i] * b.z()[i] - a.z()[i] * b.y()[i];
    r.y()[i] = a.z()[i] * b.x()[i] - a.x()[i] * b.z()[i];
    r.z()[i] = a.x()[i] * b.y()[i] - a.y()[i] * b.x()[i];
  }
  
  using V3 = Eigen::Vector3d;
  using DynStride = Eigen::InnerStride<Eigen::Dynamic>;
  using CStride = Eigen::InnerStride<1024>;
  using MapV3 =  Eigen::Map<V3,0, DynStride>;
  using CMapV3 =  Eigen::Map<const V3,0,  DynStride>;
  
   // Eigen based cross product
  [[maybe_unused]] __global__ void eigenCrossProductSoA(double* rx, const double* __restrict__ ax, const double* __restrict__ bx, size_t nElements, size_t stride) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElements) return;

    CMapV3 ma(ax+i, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(stride));
    CMapV3 mb(bx+i, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(stride));
    MapV3 mr(rx+i, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(stride));
    mr = ma.cross(mb);
  }
  
  // Eigen based cross product on embedded vectors
  [[maybe_unused]] __global__ void embeddedCrossProductSoA(testSoA::SoA soa) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= soa.nElements()) return;
#if 0
    soa[i].r() = soa[i].a().cross(soa[i].b());
#else
    using V3 = Eigen::Vector3d;
    using DynStride = Eigen::InnerStride<Eigen::Dynamic>;
    using CMapV3 =  Eigen::Map<const V3,0,  DynStride>;
    const V3::Scalar * __restrict__ mad = soa[i].a().data();
    const V3::Scalar * __restrict__ mbd = soa[i].b().data();
    CMapV3 ma(mad, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(soa[i].a.stride()));
    CMapV3 mb(mbd, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(soa[i].b.stride()));
    soa[i].r() = ma.cross(mb);
#endif
  }

   // Eigen based cross product on embedded vectors
  [[maybe_unused]] __global__ void embeddedCrossProductLocalObjectSoA(std::byte * data, size_t nElements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElements) return;
    bool deviceConstructor = true;
    testSoA::SoA soa(deviceConstructor, data, nElements);
#if 0
    soa[i].r() = soa[i].a().cross(soa[i].b());
#else
    using V3 = Eigen::Vector3d;
    using DynStride = Eigen::InnerStride<Eigen::Dynamic>;
    using CMapV3 =  Eigen::Map<const V3,0,  DynStride>;
    const V3::Scalar * __restrict__ mad = soa[i].a().data();
    const V3::Scalar * __restrict__ mbd = soa[i].b().data();
    CMapV3 ma(mad, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(soa[i].a.stride()));
    CMapV3 mb(mbd, V3::RowsAtCompileTime, V3::ColsAtCompileTime, DynStride(soa[i].b.stride()));
    soa[i].r() = ma.cross(mb);
#endif
  }

  // Simple cross product (SoA on CPU)
  [[maybe_unused]] __host__ void indirectCPUcrossProductSoA(testSoA::SoA r, const testSoA::SoA a, const testSoA::SoA b, size_t nElements) {
    for (size_t i =0; i< nElements; ++i) {
      // This version is also affected.
      auto ri = r[i];
      crossProduct(ri, a[i], b[i]);
    }
  }

  // Simple cross product (AoS)
  [[maybe_unused]] __global__ void crossProductAoS(testSoA::AoSelement *r,
          testSoA::AoSelement *a, testSoA::AoSelement *b, size_t nElements) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nElements) return;
    crossProduct(r[i], a[i], b[i]);
  }

  [[maybe_unused]] void hexdump(void *ptr, int buflen) {
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
  [[maybe_unused]] __host__ __device__ __forceinline__ void checkSoAelement(T soa, size_t i, bool & pass) {
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
  [[maybe_unused]] __host__ __device__ __forceinline__ void checkSoAelementNullityRealtiveToSquare(T resSoA, T referenceSoA, size_t i, double epsilon, bool & pass) {
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
  [[maybe_unused]] __host__ __device__ __forceinline__ void checkCrossProduct(T resultSoA, T aSoA, T bSoA, size_t i, double epsilon, bool & pass) {
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
  
  template <class T>
  [[maybe_unused]] __host__ __device__ __forceinline__ void checkEmbeddedCrossProduct(T soa, size_t i, double epsilon, bool & pass) {
    if (i >= soa.nElements() || !pass) return;
    auto & a = soa[i].a();
    auto & b = soa[i].b();
    auto & r = soa[i].r();
    auto refA = max (abs(a.x()), max(abs(a.y()), abs(a.z())));
    auto refB = max (abs(b.x()), max(abs(b.y()), abs(b.z())));
    auto ref = max(refA, refB);
    ref *= ref * epsilon;
    testSoA::AoSelement myRes, AoSa, AoSb;
    AoSa.x = a.x();
    AoSa.y = a.y();
    AoSa.z = a.z();
    AoSb.x = b.x();
    AoSb.y = b.y();
    AoSb.z = b.z();
    crossProduct(myRes, AoSa, AoSb);
    if (abs(myRes.x - r.x()) > ref) { pass = false; return; }
    if (abs(myRes.y - r.y()) > ref) { pass = false; return; }
    if (abs(myRes.z - r.z()) > ref) { pass = false; return; }
  }

  class StreamTimer {
  public:
    StreamTimer() {
      if (cudaSuccess != cudaEventCreate(&beginning_) 
          || cudaSuccess != cudaEventCreate(&end_))
        throw std::bad_alloc();
    }
    
    ~StreamTimer() {
      cudaEventDestroy(beginning_);
      cudaEventDestroy(end_);
    }
    
    void start(cudaStream_t s) {
      CUDA_UNIT_CHECK(cudaEventRecord(beginning_, s));
    }
    
    void stop(cudaStream_t s) {
      CUDA_UNIT_CHECK(cudaEventRecord(end_, s));
      
    }
    
    float mSecs() {
      CUDA_UNIT_CHECK(cudaEventSynchronize(beginning_));
      CUDA_UNIT_CHECK(cudaEventSynchronize(end_));
      float ret;
      CUDA_UNIT_CHECK(cudaEventElapsedTime(&ret, beginning_, end_));
      return ret;
    }
    
  private:
    cudaEvent_t beginning_, end_;
  };
} // Anonymous namesapce

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
  indirectCrossProductSoA<<<
    (elementsCount - 1)/deviceProperties.warpSize + 1,
    deviceProperties.warpSize,
    0, stream
  >>>(deviceSoAR, deviceSoAA, deviceSoAA, elementsCount);
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
  cudaStream_t streamA, streamB, streamR;
  CUDA_UNIT_CHECK(cudaStreamCreate(&streamA));
  CUDA_UNIT_CHECK(cudaStreamCreate(&streamB));
  CUDA_UNIT_CHECK(cudaStreamCreate(&streamR));
  
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
  
  // Timer to measure performance
  ::StreamTimer timer;
  // Call kernels, get result. Also fill up result SoA to ensure the results go in the right place.
  randomFillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, streamA>>>(deviceSoAA, 0xdeadbeef);
  randomFillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, streamB>>>(deviceSoAB, 0xcafefade);
  randomFillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, streamR>>>(deviceSoAR, 0xfadedcab);
  cudaEvent_t eventA, eventB;
  CUDA_UNIT_CHECK(cudaEventCreate(&eventA));
  CUDA_UNIT_CHECK(cudaEventCreate(&eventB));
  CUDA_UNIT_CHECK(cudaEventRecord(eventA, streamA));
  CUDA_UNIT_CHECK(cudaEventRecord(eventB, streamB));
  CUDA_UNIT_CHECK(cudaStreamWaitEvent(streamR, eventA));
  CUDA_UNIT_CHECK(cudaStreamWaitEvent(streamR, eventB));
  // Run more to gather statistics
  timer.start(streamR);
  for (size_t i=0; i<20; ++i) {
    indirectCrossProductSoA<<<
      (elementsCount - 1)/deviceProperties.warpSize + 1,
      deviceProperties.warpSize,
      0, streamR
    >>>(deviceSoAR, deviceSoAA, deviceSoAB, elementsCount);
  }
  timer.stop(streamR);
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockA.get(), deviceSoABlockA.get(), SoA::computeDataSize(hostSoAA.nElements()), cudaMemcpyDeviceToHost, streamA));
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockB.get(), deviceSoABlockB.get(), SoA::computeDataSize(hostSoAA.nElements()), cudaMemcpyDeviceToHost, streamB));
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockR.get(), deviceSoABlockR.get(), SoA::computeDataSize(hostSoAR.nElements()), cudaMemcpyDeviceToHost, streamR));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(streamR));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(streamA));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(streamB));

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
  }
  std::cout << "indirectCrossProductSoA time=" << timer.mSecs() * 1000 << " us." << std::endl;
  CPPUNIT_ASSERT(pass);
}


void testSoA::randomCrossProductEmbeddedVector() {
  // Get device, stream, memory
  cudaDeviceProp deviceProperties;
  int deviceCount=0;
  CUDA_UNIT_CHECK(cudaGetDeviceCount(&deviceCount));
  CPPUNIT_ASSERT(deviceCount > 0);
  CUDA_UNIT_CHECK(cudaGetDeviceProperties(&deviceProperties, defaultDevice));
  cudaStream_t stream;
  CUDA_UNIT_CHECK(cudaStreamCreate(&stream));
  
  // Allocate memory and populate SoA descriptors (device A as source and R as result of cross product)
  auto deviceSoABlock = make_device_unique(SoA::computeDataSize(elementsCount));
  auto hostSoABlock = make_host_unique(SoA::computeDataSize(elementsCount));
  SoA deviceSoA(deviceSoABlock.get(), elementsCount);
  SoA hostSoA(hostSoABlock.get(), elementsCount);
 
  
  // Timer to measure performance
  ::StreamTimer timer, timer2;
  // Call kernels, get result. Also fill up result SoA to ensure the results go in the right place.
  randomFillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, stream>>>(deviceSoA, 0xdeadbeef);
  // Run more to gather statistics
  timer.start(stream);
  for (size_t i=0; i<20; ++i) {
    embeddedCrossProductLocalObjectSoA<<<
      (elementsCount - 1)/deviceProperties.warpSize + 1,
      deviceProperties.warpSize,
      0, stream
    >>>(deviceSoABlock.get(), elementsCount);
  }
  timer.stop(stream);
  timer2.start(stream);
  for (size_t i=0; i<20; ++i) {
    embeddedCrossProductSoA<<<
      (elementsCount - 1)/deviceProperties.warpSize + 1,
      deviceProperties.warpSize,
      0, stream
    >>>(deviceSoA);
  }
  timer2.stop(stream);
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlock.get(), deviceSoABlock.get(), SoA::computeDataSize(hostSoA.nElements()), cudaMemcpyDeviceToHost, stream));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(stream));

  // Validate result
  bool pass = true;
  size_t i = 0;
  for (; pass && i< hostSoA.nElements(); i++) {
    checkEmbeddedCrossProduct(hostSoA, i, std::numeric_limits<double>::epsilon(), pass);
  }
  if (!pass) {
    // Recompute the expected result
    Eigen::Vector3d expected;
    auto a = hostSoA[i].a();
    auto b = hostSoA[i].b();
    auto r = hostSoA[i].r();
    expected =  a.cross(b);
    std::cout << "In " << __FUNCTION__ << " check failed at i= " << i << std::endl;
    std::cout << "result= ("   << r.x() << ", " << r.y() << ", " << r.z() << ")" << std::endl;
    std::cout << "expected= (" << expected.x() << ", " << expected.y() << ", " << expected.z() << ")" << std::endl;
    std::cout << "A= (" << a.x() << ", " << a.y() << ", " << a.z() << ")" << std::endl;
    std::cout << "B= (" << b.x() << ", " << b.y() << ", " << b.z() << ")" << std::endl;
  }
  std::cout << "embeddedCrossProductLocalObjectSoA time=" << timer.mSecs() * 1000 << " us." << std::endl;
  std::cout << "embeddedCrossProductSoA time=" << timer2.mSecs() * 1000 << " us." << std::endl;
  CPPUNIT_ASSERT(pass);
}

void testSoA::randomCrossProductEigen() {
  // Get device, stream, memory
  cudaDeviceProp deviceProperties;
  int deviceCount=0;
  CUDA_UNIT_CHECK(cudaGetDeviceCount(&deviceCount));
  CPPUNIT_ASSERT(deviceCount > 0);
  CUDA_UNIT_CHECK(cudaGetDeviceProperties(&deviceProperties, defaultDevice));
  cudaStream_t streamA, streamB, streamR;
  CUDA_UNIT_CHECK(cudaStreamCreate(&streamA));
  CUDA_UNIT_CHECK(cudaStreamCreate(&streamB));
  CUDA_UNIT_CHECK(cudaStreamCreate(&streamR));
  
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
  
  // Timer to measure performance
  ::StreamTimer timer;
  // Call kernels, get result. Also fill up result SoA to ensure the results go in the right place.
  randomFillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, streamA>>>(deviceSoAA, 0xdeadbeef);
  randomFillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, streamB>>>(deviceSoAB, 0xcafefade);
  randomFillSoA<<<(elementsCount - 1)/deviceProperties.warpSize + 1, deviceProperties.warpSize, 0, streamR>>>(deviceSoAR, 0xfadedcab);
  cudaEvent_t eventA, eventB;
  CUDA_UNIT_CHECK(cudaEventCreate(&eventA));
  CUDA_UNIT_CHECK(cudaEventCreate(&eventB));
  CUDA_UNIT_CHECK(cudaEventRecord(eventA, streamA));
  CUDA_UNIT_CHECK(cudaEventRecord(eventB, streamB));
  CUDA_UNIT_CHECK(cudaStreamWaitEvent(streamR, eventA));
  CUDA_UNIT_CHECK(cudaStreamWaitEvent(streamR, eventB));
  const size_t stride = (((elementsCount * sizeof(double) - 1) / deviceSoAA.byteAlignment() ) + 1) * deviceSoAA.byteAlignment() / sizeof(double);
  // Run more to gather statistics
  timer.start(streamR);
  for (size_t i=0; i<20; ++i) {
    eigenCrossProductSoA<<<
      (elementsCount - 1)/deviceProperties.warpSize + 1,
      deviceProperties.warpSize,
      0, streamR
    >>>(deviceSoAR.x(), deviceSoAA.x(), deviceSoAB.x(), elementsCount, stride);
  }
  timer.stop(streamR);
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockA.get(), deviceSoABlockA.get(), SoA::computeDataSize(hostSoAA.nElements()), cudaMemcpyDeviceToHost, streamA));
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockB.get(), deviceSoABlockB.get(), SoA::computeDataSize(hostSoAA.nElements()), cudaMemcpyDeviceToHost, streamB));
  CUDA_UNIT_CHECK(cudaMemcpyAsync(hostSoABlockR.get(), deviceSoABlockR.get(), SoA::computeDataSize(hostSoAR.nElements()), cudaMemcpyDeviceToHost, streamR));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(streamR));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(streamA));
  CUDA_UNIT_CHECK(cudaStreamSynchronize(streamB));

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
  }
  std::cout << "eigenCrossProductSoA time=" << timer.mSecs() * 1000 << " us." << std::endl;
  CPPUNIT_ASSERT(pass);
}
