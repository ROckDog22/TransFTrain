#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace TransFTrain {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT boundaries in
 * memory.  This alignment should be at least TILE * ELEM_SIZE, though we make it even larger
 * here by default.
 */
struct AlignedArray {
  AlignedArray(const size_t size) {
    // 在构造函数中调用“posix_memalign函数来分配对齐的内存
    // ptr指向内存的指针，要求的内存对齐字节数
    // ALIGMENT是一个指向内存分配时的对齐要求，只定义在内存块中的起始地址应该满足对齐边界，256
    // size *  elesize 要分配的内存块大小，以字节为单位
    // ret为0表示成功
    int ret = posix_memalign((void**)&ptr, ALIGNMENT, size * ELEM_SIZE);
    if (ret != 0) throw std::bad_alloc();
    this->size = size;
  }
  ~AlignedArray() { free(ptr); }
  size_t ptr_as_int() {return (size_t)ptr; }
  scalar_t* ptr;
  size_t size;
};



void Fill(AlignedArray* out, scalar_t val) {
  /**
   * Fill the values of an aligned array with val
   */
  for (int i = 0; i < out->size; i++) {
    out->ptr[i] = val;
  }
}

void incIndices(uint32_t *indices, const std::vector<uint32_t> shape){
    int i;
    for (i = shape.size()-1; i>=0; i--){
      if(indices[i] < shape[i] - 1)
          break;
    }
    if(i>=0)
      indices[i]++;
    for(int j= i+1; j< shape.size(); ++j)
      indices[j]= 0;
}

uint32_t getOffset(const uint32_t *indices, const std::vector<uint32_t>& strides, uint32_t start=0){
  uint32_t ret = start;
  for(size_t i=0; i<strides.size(); i++){
    ret += strides[i] * indices[i];
  }
  return ret;
}



void Compact(const AlignedArray& a, AlignedArray* out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
  /**
   * Compact an array in memory
   *
   * Args:
   *   a: non-compact representation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   *
   * Returns:
   *  void (you need to modify out directly, rather than returning anything; this is true for all the
   *  function will implement here, so we won't repeat this note.)
   */
  // 函数用于在内存中分配指定数量的连续字节，并将其初始化为0，需要分配的元素数量，每个元素的字节大小
   uint32_t *indices = (uint32_t*)calloc(shape.size(), sizeof(uint32_t));
   for(size_t i = 0; i< out->size; ++i, incIndices(indices, shape)){
    out->ptr[i] = a.ptr[getOffset(indices, strides, offset)];
   }
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array
   *
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
   uint32_t* indices = (uint32_t*)calloc(shape.size(), sizeof(uint32_t));
   for(size_t i = 0; i< a.size; ++i, incIndices(indices, shape)){
    out->ptr[getOffset(indices, strides, offset)] = a.ptr[i];
   }
   free(indices);
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out, std::vector<uint32_t> shape,
                   std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   *
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
   *         product of items in shape, but convenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */

  uint32_t* indices = (uint32_t*)calloc(shape.size(), sizeof(uint32_t));
  for (size_t i = 0; i<size; ++i, incIndices(indices, shape)){
    out->ptr[getOffset(indices, strides, offset)] = val;
  }
  free(indices);
}

void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of correspondings entires in a and b.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + b.ptr[i];
  }
}

void ScalarAdd(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  /**
   * Set entries in out to be the sum of corresponding entry in a plus the scalar val.
   */
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = a.ptr[i] + val;
  }
}

void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for (size_t i=0; i<a.size; i++){
    out->ptr[i] = a.ptr[i] * b.ptr[i];
  }
}

void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for(size_t i=0; i<a.size; i++) {
    out->ptr[i] = a.ptr[i] * val;
  } 
}

void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for(size_t i=0; i<a.size; i++)
    out->ptr[i] = a.ptr[i] / b.ptr[i];
}

void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for(size_t i=0; i<a.size; i++)
   out->ptr[i] = a.ptr[i] / val;
}

void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
    for(size_t i=0; i<a.size; i++){
      out->ptr[i] = std::pow(a.ptr[i], val);
    }
}

void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  for(size_t i=0; i<a.size; i++){
    out->ptr[i] = std::max(a.ptr[i], b.ptr[i]);
  }
}

void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  for (size_t i =0; i<a.size; i++){
    out->ptr[i] = std::max(a.ptr[i], val);
  }
}

void EwiseEq(const AlignedArray& a, const  AlignedArray& b, AlignedArray* out){
  for(size_t i=0; i<a.size; i++){
    out->ptr[i] = a.ptr[i] == b.ptr[i];
  }
}

void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out){
  for(size_t i=0; i<a.size; i++){
    out->ptr[i] = a.ptr[i] == val;
  }
}

void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out){
  for(size_t i=0; i< a.size; ++i){
    out->ptr[i] = a.ptr[i] > b.ptr[i];
  }
}

void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out){
  for(size_t i=0; i<a.size; i++){
    out->ptr[i] = a.ptr[i] > val;
  }
}

void EwiseLog(const AlignedArray& a, AlignedArray* out){
  for(size_t i=0; i<a.size; i++){
    out->ptr[i] = std::log(a.ptr[i]);
  }
}

void EwiseExp(const AlignedArray& a, AlignedArray* out){
  for(size_t i=0; i<a.size; i++){
    out->ptr[i] = std::exp(a.ptr[i]);
  }
}

void EwiseTanh(const AlignedArray& a, AlignedArray* out){
  for(size_t i=0; i<a.size; i++){
    out->ptr[i] = std::tanh(a.ptr[i]);
  }
}
void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
   * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
   * you can use the "naive" three-loop algorithm.
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: compact 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   */
    std::fill(out->ptr, out->ptr + m*n, 0.0);
    for (size_t i=0; i<m; i++){
      for (size_t j=0; j<p; j++){
        for (size_t k=0; k<n; k++){
          out->ptr[i*p+j] += a.ptr[i*n + k] * b.ptr[k*p + j];
        }
      }
    }
}

// restrict 是一种特殊的限定符，通常告知编译器该指针是一个独占访问的指针，即在指针的生命周期内
// 没有其他指针会访问相同内存的区域，这样的声明可以帮助编译器进行优化，例如对指针进行指令重排，寄存器分配
inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out) {

  /**
   * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
   * the result to the existing out, which you should not set to zero beforehand).  We are including
   * the compiler flags here that enable the compile to properly use vector operators to implement
   * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
   * out don't have any overlapping memory (which is necessary in order for vector operations to be
   * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
   * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
   * compiler that the input array will be aligned to the appropriate blocks in memory, which also
   * helps the compiler vectorize the code.
   *
   * Args:
   *   a: compact 2D array of size TILE x TILE
   *   b: compact 2D array of size TILE x TILE
   *   out: compact 2D array of size TILE x TILE to write to
   */
// 特殊语法，在编译器使用，gcc 和 clang ，它用于告知编译器，指针‘a’是按照特定的对齐方式进行对齐的，
// 从而帮助编译器进行优化 builtinassumealigned是编译器内置的函数，接受一个指针a和一个对齐方式的表达式
// tile * elem_size， 该函数用于提示编译器，指针是按照什么方式进行对齐
// 优化代码，使用对齐的加载/存储指令
  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);
  
  for (size_t i=0; i<TILE; ++i){
    for (size_t j=0; j<TILE; ++j){
      for (size_t k=0; k<TILE; ++k){
        out[i*TILE+j] = a[i*TILE+k] * a[k*TILE+j];
      }
    }
  }
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
   * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
   * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
   *   a[m/TILE][n/TILE][TILE][TILE]
   * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
   * function should call `AlignedDot()` implemented above).
   *
   * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
   * assume that this division happens without any remainder.
   *
   * Args:
   *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
   *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
   *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
   *   m: rows of a / out
   *   n: columns of a / rows of b
   *   p: columns of b / out
   *
   */
    std::fill(out->ptr, out->ptr + m * p, 0);
    for (size_t i = 0, offset = 0; i < m / TILE; ++i, offset += p * TILE)
        for (size_t j = 0, o = offset; j < p / TILE; ++j, o += TILE * TILE)
            for (size_t k = 0; k < n / TILE; ++k)
                AlignedDot(a.ptr + i * n * TILE + k * TILE * TILE,
                           b.ptr + k * p * TILE + j * TILE * TILE,
                           out->ptr + o);
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */
//  用于查找给定范围内最大的元素， 
  for(size_t i=0 ; i<a.size/reduce_size; i++){
    out[i] = *std::max_element(a.ptr + i * reduce_size, a.ptr + (i+1) * reduce_size);
  }
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
   * Reduce by taking sum over `reduce_size` contiguous blocks.
   *
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   reduce_size: size of the dimension to reduce over
   */

    for(size_t i =0; i<a.size / reduce_size; i++){
      scalar_t init = 0;
      // 对元素进行累加
      // 第一个版本接受一个范围[first, last), init将范围内的元素累加起来，返回累加结果
      out->ptr[i] = std::accumulate(a.ptr + i * reduce_size, a.ptr + (i+1) * reduce_size, init);
    }
}

}  // namespace cpu
}  // namespace TransFTrain

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace TransFTrain;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);
  m.def("matmul_tiled", MatmulTiled);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
