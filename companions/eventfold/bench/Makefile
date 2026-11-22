CXX ?= c++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic
# Override on systems with a runtime-only Debian/Ubuntu LZ4 installation.
LZ4_LIB ?= -llz4
LDLIBS ?= -lzstd -lz $(LZ4_LIB)
.PHONY: all
all: ../build/bench
../build/bench: run.cpp codecs.hpp lz4frame_abi.h ../eventfold.h
	mkdir -p ../build
	$(CXX) $(CXXFLAGS) run.cpp -o $@ $(LDLIBS)
