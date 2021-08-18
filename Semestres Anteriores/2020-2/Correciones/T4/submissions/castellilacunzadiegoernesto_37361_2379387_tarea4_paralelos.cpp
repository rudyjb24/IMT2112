// Compilation as: gcc -framework OpenCL opencl-vecadd.cpp

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdlib>
#include <math.h>
#include <cmath>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif



// Aqui creamos los dos Kernel el primero recibe c=c1 + i c2; y itera con respecto a cn
// pero como necesito el termino anterior ocupo cn_1 que vendria siendo c_{n-1}
// Retorna d que es un array con el modulo de cn, en el segundo es el reduce con grupos habitual
// basado en el de clases con el profe.
const char* src =
"__kernel void num_conv(                          \n"
"	__global float *c1,                           \n"
"	__global float *c2,                           \n"
"	__global float *c1n,                          \n"
"	__global float *c2n,                          \n"
"	__global float *c1n_1,                        \n"
"	__global float *c2n_1,                        \n"
"	__global float *d,                            \n"
"	__const unsigned int n_it,                    \n"
"	__const unsigned int K)                       \n"
"{                                                \n"
"	int id = get_global_id(0);                    \n"
"	if(id < K) {                                  \n"
"       for (int i = 0; i < n_it; i++) {          \n"
"           c1n_1[id] = c1n[id];                  \n"
"           c2n_1[id] = c2n[id];                  \n"
"           c1n[id] = (c1n_1[id] * c1n_1[id]) - (c2n_1[id] * c2n_1[id]) + c1[id];     \n"
"           c2n[id] = (2 * c1n_1[id] * c2n_1[id]) + c2[id];                \n"
"       }                                         \n"
"       d[id] = (c1n[id]*c1n[id]) + (c2n[id]*c2n[id]);            \n"
"   }                                             \n"
"}                                                \n"
"__kernel void reduce(                                        \n"
"    __global int *vec,                                       \n"
"    __const unsigned int K,                                  \n"
"    __global int* partial_sums)                              \n"
"{                                                            \n"
"    int group_size = get_local_size(0);                      \n"
"    int local_id = get_local_id(0);                          \n"
"    int group_id = get_group_id(0);                          \n"
"    int global_id = get_global_id(0);                        \n"
"    if (local_id == 0) {                                     \n"
"        int sum = 0;                                         \n"
"        for (int i = 0; i < group_size; i++) {               \n"
"            sum += vec[global_id+i];                         \n"
"        }                                                    \n"
"        partial_sums[group_id] = sum;                        \n"
"    }                                                        \n"
"}                                                            \n"
;





int main() {

    // Definimos variables
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue myqueue;

    // bla bla
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    ctx = clCreateContext(cps, 1, &device, NULL, NULL, &err);
    myqueue = clCreateCommandQueueWithProperties(ctx, device, 0, &err);

    // printeo el nombre del device
    size_t valueSize;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    char* value = (char*)malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device name: %s\n", value);
    free(value);

    // Ahora el Host aqui se pude jugar con el K y el n_it :)
    const unsigned int K = 20000;
    const unsigned int n_it = 55000;
    float area = 0;
    float* h_c1 = (float*)malloc(sizeof(float) * K);
    float* h_c2 = (float*)malloc(sizeof(float) * K);
    float* h_c1n = (float*)malloc(sizeof(float) * K);
    float* h_c2n = (float*)malloc(sizeof(float) * K);
    float* h_c1n_1 = (float*)malloc(sizeof(float*) * K);
    float* h_c2n_1 = (float*)malloc(sizeof(float*) * K);
    float* h_d = (float*)malloc(sizeof(float) * K);
    srand(time(0));
    for (int i = 0; i < K; i++) {
        h_c1[i] = 3 * ((float)rand() / (float)RAND_MAX) - 2;
        h_c2[i] = 2 * ((float)rand() / (float)RAND_MAX) - 1;
    }

    // Los grupos para el reduce
    const unsigned int size_group = 16;
    int num_groups = K / size_group;
    if (num_groups * size_group != K) {
        num_groups += 1;
    }

    // aqui se prepara el vector que se entrega a vec, donde se comienza con todos 1,
    // pues asi se ve como un paradigma de probar divergencia, es decir todos pertenecen hasta 
    // probar lo contrario.
    const unsigned int n = num_groups * size_group;
    int n_pad = n - K;
    int* h_vec = (int*)malloc(sizeof(int) * n);
    int* h_partial_sums = (int*)calloc(num_groups, sizeof(int));
    for (int i = 0; i < K; i++) {
        h_vec[i] = 1;
    }
    for (int i = K; i < n; i++) {
        h_vec[i] = 0;
    }



    cl_mem d_c1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
    cl_mem d_c2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
    cl_mem d_c1n = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
    cl_mem d_c2n = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
    cl_mem d_c1n_1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
    cl_mem d_c2n_1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
    cl_mem d_d = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
    cl_mem d_vec = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n * sizeof(int), NULL, &err);
    cl_mem d_partial_sums = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int) * num_groups, NULL, &err);

    // Aqui escribo solo el c, y fijate que se hace todo lo del primer kernel pero no el segundo.
    err = clEnqueueWriteBuffer(myqueue, d_c1, CL_TRUE, 0, K * sizeof(float), (void*)h_c1, 0, NULL, NULL);

    // creo el programa
    cl_program myprog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    err = clBuildProgram(myprog, 0, NULL, NULL, NULL, NULL);

    // los kernel
    cl_kernel mykernelConv = clCreateKernel(myprog, "num_conv", &err);
    cl_kernel mykernelReduce = clCreateKernel(myprog, "reduce", &err);

    // defino las variables del primer kernel
    clSetKernelArg(mykernelConv, 0, sizeof(cl_mem), (void*)&d_c1);
    clSetKernelArg(mykernelConv, 1, sizeof(cl_mem), (void*)&d_c2);
    clSetKernelArg(mykernelConv, 2, sizeof(cl_mem), (void*)&d_c1n);
    clSetKernelArg(mykernelConv, 3, sizeof(cl_mem), (void*)&d_c2n);
    clSetKernelArg(mykernelConv, 4, sizeof(cl_mem), (void*)&d_c1n_1);
    clSetKernelArg(mykernelConv, 5, sizeof(cl_mem), (void*)&d_c2n_1);
    clSetKernelArg(mykernelConv, 6, sizeof(cl_mem), (void*)&d_d);
    clSetKernelArg(mykernelConv, 7, sizeof(unsigned int), (void*)&n_it);
    clSetKernelArg(mykernelConv, 8, sizeof(unsigned int), (void*)&K);


    // ejecuto el kernel
    size_t localSize = 8;
    size_t globalSize = (size_t)ceil(K / (float)localSize) * localSize;
    err = clEnqueueNDRangeKernel(myqueue, mykernelConv, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    // recupero d, donde estan los modulos de la n-esima iteración de c
    err = clEnqueueReadBuffer(myqueue, d_d, CL_TRUE, 0, K * sizeof(float), (void*)h_d, 0, NULL, NULL);

    // Y aqui evaluo el criterio de divergencia
    for (int i = 0; i < K; i++) {
        if ((h_d[i] > 2) || isnan(h_d[i])) {
            h_vec[i] = 0;
        }
    }

    // Escribo Vec que es el vector que es 1 si c pertenece al conjunto, 0 en otro caso
    err = clEnqueueWriteBuffer(myqueue, d_vec, CL_TRUE, 0, n * sizeof(int), (void*)h_vec, 0, NULL, NULL);

    // preparo el segundo kernel
    clSetKernelArg(mykernelReduce, 0, sizeof(cl_mem), (void*)&d_vec);
    clSetKernelArg(mykernelReduce, 1, sizeof(unsigned int), (void*)&K);
    clSetKernelArg(mykernelReduce, 2, sizeof(cl_mem), (void*)&d_partial_sums);


    // Ejecuto el segundo Kernel
    err = clEnqueueNDRangeKernel(myqueue, mykernelReduce, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    // Leo las sumas parciales
    err = clEnqueueReadBuffer(myqueue, d_partial_sums, CL_TRUE, 0, num_groups * sizeof(float), (void*)h_partial_sums, 0, NULL, NULL);

    // Junto los trozos de suma
    for (int i = 0; i < num_groups; i++) {
        area += h_partial_sums[i];
    }



    // Divido por K para sacar la proporción, luego por el area del rectangulo para tener
    // la aproximación deseada.
    area = 6 * area / K;



    printf("Estimate Area is: %f\n", area);





    // Y finalmente los free
    clReleaseKernel(mykernelConv);
    clReleaseKernel(mykernelReduce);
    clReleaseProgram(myprog);
    clReleaseMemObject(d_d);
    clReleaseMemObject(d_c1);
    clReleaseMemObject(d_c1n);
    clReleaseMemObject(d_c1n_1);
    clReleaseMemObject(d_c2);
    clReleaseMemObject(d_c2n);
    clReleaseMemObject(d_c2n_1);
    clReleaseMemObject(d_vec);
    clReleaseMemObject(d_partial_sums);
    clReleaseCommandQueue(myqueue);
    clReleaseContext(ctx);
    clReleaseDevice(device);
    free(h_d);
    free(h_c1);
    free(h_c1n);
    free(h_c1n_1);
    free(h_c2);
    free(h_c2n);
    free(h_c2n_1);
    free(h_vec);
    free(h_partial_sums);

    return 0;
}