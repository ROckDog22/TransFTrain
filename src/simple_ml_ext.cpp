#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iostream>

namespace py = pybind11;

void mat_mul(float* result, const float *X, const float *theta, size_t m, size_t n, size_t k);
void mat_exp(float* mat, size_t size);
void mat_normalize(float* mat, size_t row, size_t columns);
void mat_transpose(const float* mat, float* transposed_mat, size_t row, size_t column);
void mat_sub(float* mat1, float* mat2, size_t size, float constant);

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t index, i = 0 ; 
    while(i<m ){
        index = (i+batch<=m) ?  i+batch : m ;
        
        float* outputs = (float *)calloc((index-i)*k,sizeof(float));
        mat_mul(outputs,&X[i*n],theta,index-i,n,k);
        
        
        mat_exp(outputs,(index-i)*k); 
        mat_normalize(outputs,(index-i),k);
            

        float* identity = (float *)calloc((index-i)*k,sizeof(float));   
        for(size_t j = i ; j< index; j++ ){identity[(j-i)*k+y[j]]=1;}
        
        mat_sub(outputs,identity,(index-i)*k,1.0);


        float* X_transpose = (float *)calloc((index-i)*n,sizeof(float));
        mat_transpose( &X[i*n],X_transpose,(index-i),n);


        float* results = (float *)calloc(n*k,sizeof(float));
        mat_mul(results ,X_transpose,outputs,n,(index-i),k);    
        


        mat_sub(theta,results,n*k,(lr/(index-i)));

        free(results);
        free (X_transpose);
        free(identity);
        free(outputs);
        i+=batch;
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}

void mat_mul(float * result, const float *X, const float *theta, size_t m, size_t n, size_t k){
    /**
     * A Naive matrix multiplication function. This should save in result
     * the multiplication defined by X and theta (and sizes m, n, k).
     * 
     * Args:
     *     result (float *): pointer to result data, of size m*k, stored in C format
     *     x (const float *): pointer to X data, of size m*n, stored in row major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     * 
     * 
     * Returns:
     *     (None)
    */
    for (size_t i=0; i<m; i++){
        for(size_t j=0; j<k; j++){
            for(size_t l=0; l<n; l++){
                result[i*k+j]+=X[i*n+l]*theta[l*k+j];
            }
        }
    }
}

void mat_exp(float* mat, size_t size){
    for(size_t i=0; i<size; i++){
        mat[i] = exp(mat[i]);
    }
}

void mat_normalize(float * mat, size_t row, size_t columns){
    float* normalizer = (float*)calloc(row, sizeof(float));
    for(size_t i=0; i<row; i++){
        for(size_t j=0; j<columns; j++){
            normalizer[i]+=mat[i*columns+j];
        }
    }
    for(size_t i=0; i<row; i++){
        for(size_t j=0; j<columns; j++){
            mat[i*columns+j]/=normalizer[i];
        }
    }
    free(normalizer);
}

void mat_sub(float* mat1, float* mat2, size_t size, float constant){
    for(size_t i=0; i<size;i++){
        mat1[i]-=mat2[i]*constant;
    }
}

void mat_transpose(const float* mat, float* transposed_mat, size_t row, size_t column){
    for(size_t i=0; i<row; i++){
        for(size_t j=0; j<column; j++){
            transposed_mat[j*row+i]=mat[i*column+j];
        }
    }
}