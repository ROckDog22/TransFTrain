# # NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
# default:
# 	c++ -O3 -Wall -shared -std=c++11 -fPIC $$(python3 -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so
.PHONY: lib, pybind, clean, format, all

all: lib


lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

format:
	python3 -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	rm -rf build python/needle/backend_ndarray/ndarray_backend*.so
