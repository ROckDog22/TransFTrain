#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
// global, device, host是用于cuda c/c++代码中的限定符， 用于定义函数或变量的执行位置和内存类型，
//  global 用于定义在gpu上执行的函数，被称为内核函数，它们可以由主机cpu调用，并在GPU上的多个线程中并行执行，
// 内核函数通常用于对GPU数据进行计算
//  device这个限定符用于定义只能在GPU上执行并且只能从GPU代码访问的函数或变量，device的函数可以从其他设备代码，如
// 设备函数或内核函数调用，设备变量存储在GPU的内存中，并可以被GPU线程访问
// host 用于定义只能在CPU上执行，并且只能从主机代码访问的函数或变量，host的函数可以从主机CPU代码调用，主机变量存储
// 在主机的内存中，并可以被CPU代码访问
// 这些限定符允许开发者指定函数或变量的存放位置和访问方式，通过适当地使用这些限定符号，您可以控制代码的执行位置和内存类型
// 
namespace TransFTrain {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  // dim3 是一个用于描述线程块和网格维度的数据类型， 它是cuda_runtime api中定义的一个结构题
  // dim3 结构体包含三个无符号整数成员变量 x,y,z 分别表示三个维度的大小，这三个维度可以用来描述线程块block和网格grid的结构
  // 在CUDA程序中个，线程块和网格的概念是用于并行执行代码的，线程块是一组并发执行的现场，而网格则是由多个线程块组成的集合
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  uint32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

__device__ uint32_t GetOffset(uint32_t gid, const CudaVec& shape, const CudaVec& strides, uint32_t initial){
    size_t idx = initial;

    for(int i = shape.size -1; i>=0; --i){
      idx += strides.data[i] * (gid % shape.data[i]);
      gid /= shape.data[i];
    }

    return idx;
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  ssize_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(gid < size)
    out[gid] = a[GetOffset(gid, shape, strides, offset)];
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
             std::vector<uint32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape, CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    out[GetOffset((gid), shape, strides, offset)] = a[gid];
  }
}


void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
                  std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  CudaDims dim = CudaOneDim(a.size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape), VecToCuda(strides), offset);
}

__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape, CudaVec strides, size_t offset){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[GetOffset((gid), shape, strides, offset)] = val;
  }
}


void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<uint32_t> shape,
                   std::vector<uint32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape), VecToCuda(strides), offset);
}


__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size)
    out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


__global__ void  ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    out[gid] = a[gid] * val;
  }
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  size_t gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid < size){
    out[gid] = a[gid] / b[gid];
  }
}

void EwiseDiv(const CudaArray& a, CudaArray& b, CudaArray * out){
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t * a, scalar_t val, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = a[gid] / val;
  }
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t * out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    out[gid] = max(a[gid], b[gid]);
  }
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t * out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    out[gid] = max(a[gid], val);
  }
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}


__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t * out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid<size){
    out[gid] = a[gid] == b[gid];
  }
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}


__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid<size){
    out[gid] = a[gid] == val;
  }
}


void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size)
    out[gid] = a[gid] >= b[gid];
}

void EwiseGe(const CudaArray& a, CudaArray& b, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  out[gid] = a[gid] >= val;
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = pow(a[gid], val);
  }
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = log(a[gid]);
  }
}

void EwiseLog(const CudaArray& a, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = exp(a[gid]);
  }
}

void EwiseExp(const CudaArray& a,  CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size){
    out[gid] = tanh(a[gid]);
  }
}

void EwiseTanh(const CudaArray& a, CudaArray* out){
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}


__global__ void MatmulKernel(const float* a, const float* b, float* out, uint32_t M, uint32_t N, uint32_t P){
    size_t gid = blockIdx.x* blockDim.x + threadIdx.x;
    if (gid < M*P){
      size_t i = gid / P;
      size_t j = gid % P;
      out[gid] = 0;
      for (size_t k = 0; k <N; ++k){
        out[gid] += static_cast<double>(a[i*N+k] * b[k*P+j]);
      }
    }
}
void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
  CudaDims dim = CudaOneDim(M*P);
  MatmulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}


__global__ void ReduceMaxKernel(const float* a, float* out, size_t size, size_t reduce_size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size / reduce_size){
    out[gid] = a[reduce_size*gid];
    for (size_t i=1; i< reduce_size; i++){
      out[gid] = max(a[reduce_size*gid + i], out[gid]);
    }
  }
}


void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
    CudaDims dim = CudaOneDim(a.size / reduce_size);
    ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, reduce_size);
}

__global__ void ReduceSumKernel(const float* a, float* out, size_t size, size_t reduce_size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size / reduce_size){
    out[gid] = 0;
    for (size_t i=0; i< reduce_size; i++){
      out[gid] += a[reduce_size*gid + i];
    }
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
    CudaDims dim = CudaOneDim(a.size / reduce_size);
    ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, reduce_size);
}

}  // namespace cuda
}  // namespace TransFTrain

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace TransFTrain;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
