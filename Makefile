SRC=$(wildcard *.cc *.cu)
OBJ=$(SRC:%=.tmp/%.o)
DEP=$(SRC:%=.tmp/%.d)
TEST=test_v0 test_v1 test_v2 test_v3 test_v4 test_v5
CPPUNIT_TEST=test_v5 test_v6
CUDA_TEST=test_v7 test_v8

# Can be overridden on the command line with CXX=/opt/rh/devtoolset-10/root/bin/g++
# on Centos7 (with package devtools-10).
CXX=g++-9
LD=g++-9

# Piggy back on CVMFS for external dependencies
EXTERNAL_BASE=/cvmfs/cms.cern.ch/cc8_amd64_gcc9/external
BOOST=$(EXTERNAL_BASE)/boost/1.75.0/include
CPPUNIT=$(EXTERNAL_BASE)/cppunit/1.15.x-cc43a055582fbd2170ba53a421bc6433/include
EIGEN=$(EXTERNAL_BASE)/eigen/011e0db31d1bed8b7f73662be6d57d9f30fa457a-llifpc/include/eigen3
LIBS=/cvmfs/cms.cern.ch/cc8_amd64_gcc9/cms/cmssw/CMSSW_11_3_2/external/cc8_amd64_gcc9/lib
CLANG_FORMAT=clang-format
NVCC=/usr/local/cuda/bin/nvcc
CUDA_INCLUDE=/usr/local/cuda/include

CXXFLAGS=-std=c++17 -O3 -g -Wall -Wno-attributes -pedantic -fPIC -MMD -march=native -mtune=native -I$(CUDA_INCLUDE) -I$(BOOST) -I$(CPPUNIT) -I$(EIGEN)

# Code generated by nvcc for the system compiler does not pass pedantic checks and
# is a temporary files that should not be added to dependencies.
NVCC_CXXFLAGS=--compiler-options '$(filter-out -pedantic -MMD, $(CXXFLAGS))'

NVCCARCH= -gencode arch=compute_60,code=[sm_60,compute_60] -gencode arch=compute_70,code=[sm_70,compute_70] -gencode arch=compute_75,code=[sm_75,compute_75]
NVCCFLAGS=-std=c++17 -O3 --generate-line-info -MMD --compiler-bindir $(CXX)
LDFLAGS=-lrt -lcppunit -lcuda -L$(LIBS)

.PHONY: all clean distclean dump

CPPUNIT_EXECUTABLES=$(patsubst %, %_cppunit, $(CPPUNIT_TEST))
CUDA_EXECUTABLES=$(patsubst %, %_cuda, $(CUDA_TEST))


all: $(TEST) $(CPPUNIT_EXECUTABLES) $(CUDA_EXECUTABLES)


clean:
	rm -rf .tmp/ $(TEST) $(CPPUNIT_EXECUTABLES) $(CUDA_EXECUTABLES)

distclean: clean
	rm -f $(TEST)

$(TEST): %: .tmp/%.cc.o Makefile
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

$(CPPUNIT_EXECUTABLES): %: .tmp/%.cc.o .tmp/cppunit_runner.cc.o Makefile
	$(CXX) $(CXXFLAGS) $(filter-out Makefile, $+) $(LDFLAGS) -o $@


$(CUDA_EXECUTABLES): %: .tmp/%.cc.o .tmp/%.cu.o .tmp/cppunit_runner.cc.o Makefile
	echo Deps: $+
	$(NVCC) $(NVCCFLAGS) $(NVCCARCH) $(LDFLAGS) $(filter-out Makefile, $+) -o $@

.tmp/%.cc.o: %.cc Makefile
	@mkdir -p .tmp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.tmp/%.cu.o: %.cu Makefile
	@mkdir -p .tmp
	$(NVCC)  -dc $(NVCCFLAGS) $(NVCCARCH) $(NVCC_CXXFLAGS) --verbose -c $< -o $@

.tmp/%.cu.ptx: %.cu Makefile
	@mkdir -p .tmp
	$(NVCC) $(NVCCFLAGS) --ptx --source-in-ptx -gencode arch=compute_75,code=[sm_75,compute_75] $(NVCC_CXXFLAGS) $< -o $@
	
.tmp/%.cc.i: %.cc Makefile
	 @mkdir -p .tmp
	$(CXX) $(CXXFLAGS) -E $<  | grep -v ^# | $(CLANG_FORMAT) > $@

.tmp/%.cu.i: %.cu Makefile
	 @mkdir -p .tmp
	$(NVCC) $(NVCCFLAGS) -E $<  | grep -v ^# | $(CLANG_FORMAT) > $@

-include $(DEP)
