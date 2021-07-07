SRC=$(wildcard *.cc *.cu)
OBJ=$(SRC:%=.tmp/%.o)
DEP=$(SRC:%=.tmp/%.d)
TEST=test_v0 test_v1 test_v2 test_v3 test_v4 test_v5
CPPUNIT_TEST=test_v5 test_v6
CUDA_TEST=test_v7

# Can be overridden on the command line with CXX=/opt/rh/devtoolset-10/root/bin/g++
# on Centos7 (with package devtools-10).
CXX=g++-9
LD=g++-9
BOOST=/cvmfs/cms.cern.ch/slc7_amd64_gcc10/external/boost/1.75.0/include
CLANG_FORMAT=clang-format
NVCC=/usr/local/cuda/bin/nvcc
CUDA_INCLUDE=/usr/local/cuda/include

CXXFLAGS=-std=c++17 -O3 -g -Wall -Wno-attributes -pedantic -fPIC -MMD -march=native -mtune=native -I$(CUDA_INCLUDE) -I$(BOOST)

# Code generated by nvcc for the system compiler does not pass pedantic checks and
# is a temporary files that should not be added to dependencies.
NVCC_CXXFLAGS=--compiler-options '$(filter-out -pedantic -MMD, $(CXXFLAGS))'

NVCCFLAGS=-std=c++17 -O3 -g --compiler-bindir $(CXX)
LDFLAGS=-lrt -lcppunit -lcuda

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
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(filter-out Makefile, $+) -o $@

.tmp/%.cc.o: %.cc Makefile
	@mkdir -p .tmp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.tmp/%.cu.o: %.cu Makefile
	@mkdir -p .tmp
	$(NVCC)  -dc $(NVCCFLAGS) $(NVCC_CXXFLAGS) --verbose -c $< -o $@

.tmp/%.cu.ptx: %.cu Makefile
	@mkdir -p .tmp
	$(NVCC) $(NVCCFLAGS) --ptx $< -o $@
	
.tmp/%.cc.i: %.cc Makefile
	 @mkdir -p .tmp
	$(CXX) $(CXXFLAGS) -E $<  | grep -v ^# | $(CLANG_FORMAT) > $@

.tmp/%.cu.i: %.cu Makefile
	 @mkdir -p .tmp
	$(NVCC) $(NVCCFLAGS) -E $<  | grep -v ^# | $(CLANG_FORMAT) > $@

-include $(DEP)
