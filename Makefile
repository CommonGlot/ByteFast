#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

CXX = c++
CXXFLAGS = -pthread -std=c++17 -march=native
OBJS = args.o safetensors.o matrix.o dictionary.o loss.o productquantizer.o densematrix.o quantmatrix.o vector.o model.o utils.o meter.o bytefast.o
INCLUDES = -I.

opt: CXXFLAGS += -O3 -funroll-loops -DNDEBUG
opt: bytefast

# Static library with the C API (src/bytefast_c.h), and its test.
lib: CXXFLAGS += -O3 -funroll-loops -DNDEBUG
lib: libbytefast.a

libbytefast.a: $(OBJS) bytefast_c.o
	ar rcs libbytefast.a $(OBJS) bytefast_c.o

bytefast_c.o: src/bytefast_c.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/bytefast_c.cc

c_api_test: lib tests/c_api_test.c
	cc -O2 -Wall -Wextra -Werror -Isrc -c tests/c_api_test.c -o c_api_test.o
	$(CXX) -pthread c_api_test.o libbytefast.a -o c_api_test

coverage: CXXFLAGS += -O0 -fno-inline -fprofile-arcs --coverage
coverage: bytefast

debug: CXXFLAGS += -g -O0 -fno-inline
debug: bytefast

wasm: bindings/js/bytefast_wasm.js


args.o: src/args.cc src/args.h src/*.h
	$(CXX) $(CXXFLAGS) -c src/args.cc

safetensors.o: src/safetensors.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/safetensors.cc


matrix.o: src/matrix.cc src/matrix.h src/*.h
	$(CXX) $(CXXFLAGS) -c src/matrix.cc

dictionary.o: src/dictionary.cc src/dictionary.h src/args.h src/*.h
	$(CXX) $(CXXFLAGS) -c src/dictionary.cc

loss.o: src/loss.cc src/loss.h src/matrix.h src/real.h src/*.h
	$(CXX) $(CXXFLAGS) -c src/loss.cc

productquantizer.o: src/productquantizer.cc src/productquantizer.h src/utils.h src/*.h
	$(CXX) $(CXXFLAGS) -c src/productquantizer.cc

densematrix.o: src/densematrix.cc src/densematrix.h src/utils.h src/matrix.h src/*.h
	$(CXX) $(CXXFLAGS) -c src/densematrix.cc

quantmatrix.o: src/quantmatrix.cc src/quantmatrix.h src/utils.h src/matrix.h src/*.h
	$(CXX) $(CXXFLAGS) -c src/quantmatrix.cc

vector.o: src/vector.cc src/vector.h src/utils.h src/*.h
	$(CXX) $(CXXFLAGS) -c src/vector.cc

model.o: src/model.cc src/model.h src/args.h src/*.h
	$(CXX) $(CXXFLAGS) -c src/model.cc

utils.o: src/utils.cc src/utils.h src/*.h
	$(CXX) $(CXXFLAGS) -c src/utils.cc

meter.o: src/meter.cc src/meter.h src/*.h
	$(CXX) $(CXXFLAGS) -c src/meter.cc

bytefast.o: src/bytefast.cc src/*.h
	$(CXX) $(CXXFLAGS) -c src/bytefast.cc

bytefast: $(OBJS) src/bytefast.cc src/main.cc
	$(CXX) $(CXXFLAGS) $(OBJS) src/main.cc -o bytefast

clean:
	rm -rf *.o *.gcno *.gcda bytefast libbytefast.a c_api_test bindings/js/bytefast_wasm.js bindings/js/bytefast_wasm.wasm


# WebAssembly build of the C API for the npm package (bindings/js).
LIB_SOURCES = $(filter-out src/main.cc,$(wildcard src/*.cc))
WASM_EXPORTS = _bytefast_load,_bytefast_free,_bytefast_free_error,_bytefast_predict,_bytefast_num_labels,_bytefast_label,_bytefast_dimension,_malloc,_free
WASM_RUNTIME = FS,HEAPU8,getValue,setValue,UTF8ToString,stringToUTF8,lengthBytesUTF8

bindings/js/bytefast_wasm.js: $(LIB_SOURCES) src/*.h Makefile
	em++ -std=c++17 -O3 -flto -DNDEBUG -fexceptions -Isrc $(LIB_SOURCES) -o $@ \
	  -sWASM=1 -sALLOW_MEMORY_GROWTH=1 -sMODULARIZE=1 -sEXPORT_ES6=1 \
	  -sEXPORT_NAME=createByteFast -sFORCE_FILESYSTEM=1 \
	  -sEXPORTED_FUNCTIONS=$(WASM_EXPORTS) -sEXPORTED_RUNTIME_METHODS=$(WASM_RUNTIME)
