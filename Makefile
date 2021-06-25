SRC=$(wildcard *.cc *.cu)
OBJ=$(SRC:%=.tmp/%.o)
DEP=$(SRC:%=.tmp/%.d)
TEST=test_v0 test_v1 test_v2 test_v3 test_v4 test_v5
CPPUNIT_TEST=test_v5_cppunit test_v6_cppunit

CXX=g++-9
LD=g++-9

CXXFLAGS=-std=c++17 -O3 -g -Wall -Wno-attributes -pedantic -fPIC -MMD -march=native -mtune=native
LDFLAGS=-lrt

.PHONY: all clean distclean dump

all: $(TEST) $(CPPUNIT_TEST)


clean:
	rm -rf .tmp/

distclean: clean
	rm -f $(TEST)

$(TEST): %: .tmp/%.cc.o Makefile
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

$(CPPUNIT_TEST): %: .tmp/%.cc.o Makefile .tmp/cppunit_runner.cc.o
	$(CXX) $(CXXFLAGS) $< .tmp/cppunit_runner.cc.o $(LDFLAGS) -o $@

.tmp/%.cc.o: %.cc Makefile
	@mkdir -p .tmp
	$(CXX) $(CXXFLAGS) -c $< -o $@

-include $(DEP)
