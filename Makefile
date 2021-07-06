SRC=$(wildcard *.cc *.cu)
OBJ=$(SRC:%=.tmp/%.o)
DEP=$(SRC:%=.tmp/%.d)
TEST=test_v0 test_v1 test_v2 test_v3 test_v4 test_v5
CPPUNIT_TEST=test_v5_cppunit test_v6_cppunit

CXX=g++-9
LD=g++-9
NVCC=/usr/local/cuda/bin/nvcc
CUDA_INCLUDE=/usr/local/cuda/include


CXXFLAGS=-std=c++17 -O3 -g -Wall -Wno-attributes -pedantic -fPIC -MMD -march=native -mtune=native -I$(CUDA_INCLUDE)
NVCCFLAGS=-std=c++17 -O3 -g --compiler-bindir $(CXX)
LDFLAGS=-lrt -lcppunit -lcuda

.PHONY: all clean distclean dump

all: $(TEST) $(CPPUNIT_TEST) test_v7_cppunit


clean:
	rm -rf .tmp/

distclean: clean
	rm -f $(TEST)

$(TEST): %: .tmp/%.cc.o Makefile
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

$(CPPUNIT_TEST): %: .tmp/%.cc.o Makefile .tmp/cppunit_runner.cc.o
	$(CXX) $(CXXFLAGS) $< .tmp/cppunit_runner.cc.o $(LDFLAGS) -o $@

test_v7_cppunit: .tmp/test_v7_cppunit.cc.o .tmp/test_v7_kernels.cu.o Makefile
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) .tmp/test_v7_cppunit.cc.o .tmp/test_v7_kernels.cu.o .tmp/cppunit_runner.cc.o -o $@

.tmp/%.cc.o: %.cc Makefile
	@mkdir -p .tmp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.tmp/%.cu.o: %.cu Makefile
	@mkdir -p .tmp
	$(NVCC)  -dc $(NVCCFLAGS) --compiler-options '$(filter-out -pedantic -MMD, $(CXXFLAGS))' -c $< -o $@

.tmp/%.cu.ptx: %.cu Makefile
	@mkdir -p .tmp
	$(NVCC) $(NVCCFLAGS) --ptx $< -o $@

-include $(DEP)
