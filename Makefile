SRC=$(wildcard *.cc *.cu)
OBJ=$(SRC:%=.tmp/%.o)
DEP=$(SRC:%=.tmp/%.d)
TEST=test_v0 test_v1 test_v2 test_v3 test_v4 test_v5

CXX=g++-9
LD=g++-9

CXXFLAGS=-std=c++17 -O3 -g -Wall -Wno-attributes -pedantic -fPIC -MMD -march=native -mtune=native
LDFLAGS=-lrt

.PHONY: all clean distclean dump

all: $(TEST) test_v5_cppunit


clean:
	rm -rf .tmp/

distclean: clean
	rm -f $(TEST)

$(TEST): %: .tmp/%.cc.o Makefile
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

test_v5_cppunit: .tmp/test_v5.cppunit.cc.o .tmp/test_v5.cppunitrunner.cc.o
	$(CXX) $(CXXFLAGS) $+ $(LDFLAGS) -o $@

.tmp/%.cc.o: %.cc Makefile
	@mkdir -p .tmp
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(DEP)
