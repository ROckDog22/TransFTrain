cmake_minimum_required(VERSION 3.2)
project(TransFTrain C CXX)

# find correct version of Python
# 执行外部命令，并将结果输出到 Python_Root_DIR变量中去
execute_process(COMMAND python3-config --prefix
  OUTPUT_VARIABLE Python_ROOT_DIR)
# 指明要查找的软件包是python，加载开发和解释器组件的
find_package(Python COMPONENTS Development Interpreter REQUIRED)
# 前面找到的包，添加python头文件目录
include_directories(${Python_INCLUDE_DIRS})

# find pybind
# 找到pybind11的cmake目录 Result_variable 存储退出码，OUTPUT_STRIP_TRAILING_WHITESPACE去除末尾的空白字符
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE __pybind_path
  OUTPUT_STRIP_TRAILING_WHITESPACE)
# 查找并加载pybing11软件包 PATHS是参数
find_package(pybind11 PATHS ${__pybind_path})


# 根据构建环境不同，设置了不同的编译标志，是否为MSVC 微软Visual C++编译器
if(NOT MSVC)
# c++11标准，优化级别2，native 生成针对本机架构的代码， 用于将任何现有的c++编译标志添加到新的标志中
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

# 使用SYSTEM 选项，将头文件添加到编译器的搜索路径中去
# 系统目录，告知编译器忽略其中的一些特定警告，通常有用，但是第三方和系统的就忽略了
include_directories(SYSTEM ${pybind11_INCLUDE_DIRS}) 
# 用于向列表变量添加元素， 
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
# 定义一个库目标，并将源代码文件编译为库文件，将文件编译为库目标
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)
# 将第三方库和target目标进行链接
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})
#生成pybing11模块和配置 生成扩展模块
pybind11_extension(ndarray_backend_cpu)
# 去除pybind11模块的调试信息和符号
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
# 用于设置目标的属性， 设置目标库文件的输出目录
# hidden  设置c++的可见性规则，hidden表示隐藏c++目标的符号和可见性，有助于隐藏实现细节和内部符号，提高库的封装性
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/TransFTrain/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)
# 使用条件语句来根据系统名称设置目标的属性， darwin 代表macos系统
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
# -und ,,, 允许动态链接时解析符号
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/TransFTrain/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

endif()
