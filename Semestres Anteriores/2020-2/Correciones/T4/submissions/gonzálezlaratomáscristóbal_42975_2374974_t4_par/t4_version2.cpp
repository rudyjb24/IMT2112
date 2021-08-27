#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <math.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define _USE_MATH_DEFINES
using namespace std;

float sample_angle(){
  return 2*M_PI*((float)rand()/(float)RAND_MAX);
}

float sample_radious(){
  float u = (float)rand()/(float)RAND_MAX+(float)rand()/(float)RAND_MAX;
  float r;
  if (u>1){r = 2-u;}
  else {r = u;}
  return 2*r;
}

// Create the kernel code in the form of a text string
const char * src =
"__kernel void recursion(                                   \n"
"	__global float *a,                                        \n"
"	__global float *b,                                        \n"
"	__global float *c,                                        \n"
"	__global float *d,                                        \n"
"	__global float *e,                                        \n"
"	__global float *f,                                        \n"
"	__global int *indic,                                      \n"
"	__const unsigned int n)                                   \n"
"{                                                          \n"
"int id = get_global_id(0);                                 \n"
"for(int s=0; s < 10000; s++){                               \n"
"	  if(a[id] *a[id] + b[id]* b[id] < 4){       \n"
"		 c[id] = a[id]*a[id] - b[id]* b[id] + e[id];            \n"
"		 d[id] = 2*a[id]*b[id] + f[id];                         \n"
"		 a[id] = c[id];                                         \n"
"		 b[id] = d[id];                                         \n"
"    }                                                      \n"
"   else if (a[id]*a[id] + b[id]*b[id] >= 4){  \n"
"     indic[id] = 1;                                        \n"
"    }                                                      \n"
"  }                                                        \n"
"}                                                          \n"
"__kernel void reduction(                                          \n"
"    __global int *vec,                                            \n"
"    __const unsigned int n,                                       \n"
"    __const unsigned int chunk_size,                              \n"
"    __local int* local_sums,                                      \n"
"    __global int* partial_sums)                                   \n"
"{                                                                 \n"
"    int group_size = get_local_size(0);                           \n"
"    int local_id = get_local_id(0);                               \n"
"    int group_id = get_group_id(0);                               \n"
"    int global_id = get_global_id(0);                             \n"
"                                                                  \n"
"    int start_id_group = group_id * group_size * chunk_size;      \n"
"    int start_id_thread = start_id_group + local_id * chunk_size; \n"
"    local_sums[local_id] = 0;                                     \n"
"    for (int i = 0; i < chunk_size; i++) {                        \n"
"        local_sums[local_id] += vec[start_id_thread + i];         \n"
"    }                                                             \n"
"                                                                  \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                 \n"
"                                                                  \n"
"    if (local_id == 0) {                                          \n"
"        int sum = 0;                                              \n"
"        for (int i = 0; i < group_size; i++) {                    \n"
"            sum += local_sums[i];                                 \n"
"        }                                                         \n"
"        partial_sums[group_id] = sum;                             \n"
"    }                                                             \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                 \n"
"}                                                                 \n"
;



int main(){

  // Allocate the OpenCL variables for the heterogeneous processors
	cl_int err;
	cl_platform_id platform;
	cl_device_id device;
	cl_context ctx;
	cl_command_queue myqueue;

	// Boilerplate functionality
	err = clGetPlatformIDs(1, &platform, NULL);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
	ctx = clCreateContext(cps, 1, &device, NULL, NULL, &err);
	myqueue = clCreateCommandQueue(ctx, device, 0, &err);

  srand(time(0)); //esto es para los randoms

  const unsigned int _K = 1000000; // cantidad de números c que vamos a samplear.
  const unsigned int size_group = 20; // Specify the size (#threads) of each work group.
  const unsigned int size_chunk = 5; // Specify the size (#vector elements) of each chunk.
  // Calculate the number of work groups.
  int num_groups = _K / (size_group * size_chunk);
  if (num_groups * size_group * size_chunk != _K){
      num_groups += 1;
  }
  const unsigned int K = num_groups * size_group * size_chunk;// global size, despues de paddear al final con ceros.



	// Create data on the host
	float *h_c_parte_real = (float *)malloc(sizeof(float) * K);
	float *h_c_parte_imaginaria = (float *)malloc(sizeof(float) * K);
  float *h_x_parte_real = (float *)malloc(sizeof(float) * K);
  float *h_x_parte_imaginaria = (float *)malloc(sizeof(float) * K);
  int *h_indic = (int *)malloc(sizeof(int) * K);
  for (int i = 0; i < _K; i++) {
      float angle = sample_angle();
      float radio = sample_radious();
      h_c_parte_real[i] = radio*cos(angle);
      h_c_parte_imaginaria[i] = radio*sin(angle);
      //printf("%f\n", h_c_parte_real[i]*h_c_parte_real[i] + h_c_parte_imaginaria[i]*h_c_parte_imaginaria[i]);
      h_x_parte_real[i] = 0;
      h_x_parte_imaginaria[i] = 0;
      h_indic[i] = 0;
  }
  for (int i = _K; i < K; i++) {
      h_c_parte_real[i] = 0;
      h_c_parte_imaginaria[i] = 0;
      h_x_parte_real[i] = 0;
      h_x_parte_imaginaria[i] = 0;
      h_indic[i] = 0;
  }
  float *h_aux1 = (float *)malloc(sizeof(float) * K);
  float *h_aux2 = (float *)malloc(sizeof(float) * K);
  float *h_indic2 = (float *)malloc(sizeof(float) * K);
  int *h_partial_sums = (int *)malloc(sizeof(int) * num_groups);


  // Allocate memory on the device
	cl_mem d_c_parte_real = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
	cl_mem d_c_parte_imaginaria = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
	cl_mem d_x_parte_real = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
  cl_mem d_x_parte_imaginaria = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
  cl_mem d_aux1= clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
  cl_mem d_aux2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
  cl_mem d_indic = clCreateBuffer(ctx, CL_MEM_READ_ONLY, K * sizeof(float), NULL, &err);
  cl_mem d_partial_sums = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(int) * num_groups, NULL, &err);



  // Initialise on device memory
	err = clEnqueueWriteBuffer(myqueue, d_c_parte_real, CL_TRUE, 0, K * sizeof(float), (void*)h_c_parte_real, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(myqueue, d_c_parte_imaginaria, CL_TRUE, 0, K * sizeof(float), (void*)h_c_parte_imaginaria, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(myqueue, d_x_parte_real, CL_TRUE, 0, K * sizeof(float), (void*)h_x_parte_real, 0, NULL, NULL);
	err = clEnqueueWriteBuffer(myqueue, d_x_parte_imaginaria, CL_TRUE, 0, K * sizeof(float), (void*)h_x_parte_imaginaria, 0, NULL, NULL);
  err = clEnqueueWriteBuffer(myqueue, d_indic, CL_TRUE, 0, K * sizeof(float), (void*)h_indic, 0, NULL, NULL);


  // Create the compute program
  cl_program myprog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
  err = clBuildProgram(myprog, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
      size_t len;
      char buffer[2048];
      clGetProgramBuildInfo(myprog, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      printf("%s\n", buffer);
      return EXIT_FAILURE;
  }

  // Create the kernel function
  cl_kernel mykernelRec = clCreateKernel(myprog, "recursion", &err);
  cl_kernel mykernelRed = clCreateKernel(myprog, "reduction", &err);

  // Set the arguments of the kernel function
  clSetKernelArg(mykernelRec, 0, sizeof(cl_mem), (void *)&d_x_parte_real);
  clSetKernelArg(mykernelRec, 1, sizeof(cl_mem), (void *)&d_x_parte_imaginaria);
  clSetKernelArg(mykernelRec, 2, sizeof(cl_mem), (void *)&d_aux1);
  clSetKernelArg(mykernelRec, 3, sizeof(cl_mem), (void *)&d_aux2);
  clSetKernelArg(mykernelRec, 4, sizeof(cl_mem), (void *)&d_c_parte_real);
  clSetKernelArg(mykernelRec, 5, sizeof(cl_mem), (void *)&d_c_parte_imaginaria);
  clSetKernelArg(mykernelRec, 6, sizeof(cl_mem), (void *)&d_indic);
  clSetKernelArg(mykernelRec, 7, sizeof(unsigned int), (void *)&K);

  clSetKernelArg(mykernelRed, 0, sizeof(cl_mem), (void *)&d_indic);
  clSetKernelArg(mykernelRed, 1, sizeof(unsigned int), (void *)&K);
  clSetKernelArg(mykernelRed, 2, sizeof(unsigned int), (void *)&size_chunk);
  clSetKernelArg(mykernelRed, 3, sizeof(int) * size_group, NULL);
  clSetKernelArg(mykernelRed, 4, sizeof(cl_mem), (void *)&d_partial_sums);

  // Execute the kernel in chunks of size_group
  size_t globalSize = (size_t)K;
  size_t localSize = (size_t)size_group;
  err = clEnqueueNDRangeKernel(myqueue, mykernelRec, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
  err = clEnqueueNDRangeKernel(myqueue, mykernelRed, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

  // Read back the results to the device
  err = clEnqueueReadBuffer(myqueue, d_indic,	CL_TRUE, 0, sizeof(int) * num_groups, h_indic2, 0, NULL, NULL);
  err = clEnqueueReadBuffer(myqueue, d_partial_sums,	CL_TRUE, 0, sizeof(int) * num_groups, h_partial_sums, 0, NULL, NULL);



  // The final reduction putting the workgroup results
  float global_sum = 0;
  for (int i = 0; i < num_groups; i++)
      global_sum += h_partial_sums[i];
  printf("Proportion: %f\n", 1 - (global_sum/_K));
  printf("Aproximacion: %f\n", 4*M_PI*( 1 - (global_sum/_K)));
  printf("Error: %f\n", abs(1.50659177 - 4*M_PI*( 1 - (global_sum/_K))));

  //printf("indicador = [");
  //for (int i = 0; i < K; i++) {
  //    printf("%.1f  ", h_indic2[i]);
  //}
  //printf("]\n");

  // Shutdown and cleanup
  clReleaseKernel(mykernelRec);
  clReleaseKernel(mykernelRed);
  clReleaseMemObject(d_aux2);
  clReleaseMemObject(d_aux1);
  clReleaseMemObject(d_c_parte_imaginaria);
  clReleaseMemObject(d_c_parte_real);
  clReleaseMemObject(d_x_parte_imaginaria);
  clReleaseMemObject(d_x_parte_real);
  clReleaseMemObject(d_partial_sums);
  clReleaseProgram(myprog);
  clReleaseCommandQueue(myqueue);
  clReleaseContext(ctx);
  clReleaseDevice(device);
  free(h_c_parte_imaginaria);
  free(h_c_parte_real);
  free(h_x_parte_imaginaria);
  free(h_x_parte_real);
  free(h_aux1);
  free(h_aux2);
}
