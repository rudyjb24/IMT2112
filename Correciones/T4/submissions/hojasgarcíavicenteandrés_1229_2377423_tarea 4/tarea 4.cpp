// Compilation as: gcc -framework OpenCL opencl-sum.cpp

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

// Functionality to print error messages
void exitmsg(const char* msg) { printf("ERROR: %s\n", msg); exit(EXIT_FAILURE); }
void checkError(cl_int status, const char* msg) { if (status != CL_SUCCESS) exitmsg(msg); }

// Function that approximates area of mandelbrot set.
const char* src =
"__kernel void reduce(                                             \n"
"    __global float* a_random,                                     \n"
"    __global float* b_random,                                     \n"
"    __const unsigned int n_iter,                                  \n"
"    __const unsigned int n,                                       \n"
"    __const unsigned int chunk_size,                              \n"
"    __local float* local_sums,                                    \n"
"    __global float* partial_sums)                                 \n"
"{                                                                 \n"
"    int group_size = get_local_size(0);                           \n"
"    int local_id = get_local_id(0);                               \n"
"    int group_id = get_group_id(0);                               \n"
"    int global_id = get_global_id(0);                             \n"
"                                                                  \n"
"    int start_id_group = group_id * group_size * chunk_size;      \n"
"    int start_id_thread = start_id_group + local_id * chunk_size; \n"
"    local_sums[local_id] = 0;                                     \n"
"                                                                  \n"
"    double a, b, partial;                                         \n"
"    for (int i = 0; i < chunk_size; i++) {                        \n"
"        a = 0;                                                    \n"
"        b = 0;                                                    \n"
"        partial = 0;                                              \n"
"        for (int j=0; j<n_iter; j++) {                            \n"
"            partial = a*a-b*b;                                    \n"
"            b = 2*a*b;                                            \n"
"            a = partial;                                          \n"
"            a += a_random[start_id_thread + i];                   \n"
"            b += b_random[start_id_thread + i];                   \n"
"        }                                                         \n"
"        if (a*a+b*b < 4.0) {                                      \n"
"            local_sums[local_id] += 1.0/(float) chunk_size;       \n"
"        }                                                         \n"
"    }                                                             \n"
"                                                                  \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                 \n"
"                                                                  \n"
"    if (local_id == 0) {                                          \n"
"        float sum = 0;                                            \n"
"        for (int i = 0; i < group_size; i++) {                    \n"
"            sum += local_sums[i]/(float)group_size;               \n"
"        }                                                         \n"
"        partial_sums[group_id] = sum;                             \n"
"    }                                                             \n"
"}                                                                 \n"
;

int main(int argc, char* argv[])
{
    // Create data on the host
    const unsigned int size_vector = 100000;
    printf("Size of sum: %i\n", size_vector);

    float* a_random = (float*) calloc(size_vector, sizeof(float));
    float* b_random = (float*) calloc(size_vector, sizeof(float));
    srand((int)time(0));
    for (int i = 0; i < size_vector; i++) {
        a_random[i] = 2.0 * 2.0 * ((float)rand() / (float)RAND_MAX) - 2.0;
        b_random[i] = 2.0 * 2.0 * ((float)rand() / (float)RAND_MAX) - 2.0;
    }

    // Specify the size (#threads) of each work group.
    const unsigned int size_group = 8;
    printf("Size of work group: %i\n", size_group);

    // Specify the size (#vector elements) of each chunk.
    const unsigned int size_chunk = 128;
    printf("Size of chunks: %i\n", size_chunk);

    // Calculate the number of work groups.
    int num_groups = size_vector / (size_group * size_chunk);
    if (num_groups * size_group * size_chunk != size_vector) {
        num_groups += 1;
    }
    printf("Number of groups: %i\n", num_groups);

    // Pad the vector with zeros at the end.
    const unsigned int n = num_groups * size_group * size_chunk;
    int n_pad = n - size_vector;
    printf("Size of vector: %i\n", n);

    float* h_a = (float*) calloc(n, sizeof(float));
    float* h_b = (float*) calloc(n, sizeof(float));
    for (int i = 0; i < size_vector; i++) {
        h_a[i] = a_random[i];
        h_b[i] = b_random[i];
    }

    // Print the values of the vector.
    // printf("Vector: ");
    // for (int i = 0; i < n; i++) {
        // printf("%f + i%f", h_a[i], h_b[i]);
    // }
    // printf("\n");

    // Allocate the OpenCL variables for the heterogeneous processors
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue commands;

    // Boilerplate functionality
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
    context = clCreateContext(cps, 1, &device, NULL, NULL, &err);
    commands = clCreateCommandQueue(context, device, 0, &err);

    // print device name
    size_t valueSize;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &valueSize);
    char* value = (char*)malloc(valueSize);
    clGetDeviceInfo(device, CL_DEVICE_NAME, valueSize, value, NULL);
    printf("Device name: %s\n", value);
    free(value);

    // Allocate memory on the device
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &err);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &err);

    // Initialise on device memory
    err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, n * sizeof(float), (void*)h_a, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, n * sizeof(float), (void*)h_b, 0, NULL, NULL);

    // Create the compute program from the source buffer
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&src, NULL, &err);
    checkError(err, "Creating program");

    // Build the program. If things go wrong, OpenCL provides a method that allows
    // extraction of the error message generated by the device code compiler.
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        return EXIT_FAILURE;
    }

    // Extract the kernel (function) from the program (library)
    cl_kernel kernel_reduce = clCreateKernel(program, "reduce", &err);
    checkError(err, "Creating kernel");

    // Allocate memory for the partial sums on the device
    cl_mem d_partial_sums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * num_groups, NULL, &err);
    checkError(err, "Creating buffer d_partial_sums");

    // changes number of iterations
    const unsigned int n_iter = 50000;

    // Set kernel arguments
    clSetKernelArg(kernel_reduce, 0, sizeof(cl_mem), (void*)&d_a);
    clSetKernelArg(kernel_reduce, 1, sizeof(cl_mem), (void*)&d_b);
    clSetKernelArg(kernel_reduce, 2, sizeof(unsigned int), (void*)&n_iter);
    clSetKernelArg(kernel_reduce, 3, sizeof(unsigned int), (void*)&n);
    clSetKernelArg(kernel_reduce, 4, sizeof(unsigned int), (void*)&size_chunk);
    clSetKernelArg(kernel_reduce, 5, sizeof(float) * size_group, NULL);
    clSetKernelArg(kernel_reduce, 6, sizeof(cl_mem), (void*)&d_partial_sums);
    // Local memory is not allocated from within the host code. Instead,
    // the OpenCL runtime will make sure it is visible to the workgroup
    // it belongs to when that workgroup is active.

    // Set the local and global size
    size_t globalSize = (size_t)n;
    size_t localSize = (size_t)size_group;
    printf("Global size of kernel: %i\n", (int)globalSize);
    printf("Local size of kernel: %i\n", (int)localSize);

    // Run the kernel
    err = clEnqueueNDRangeKernel(commands, kernel_reduce, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    checkError(err, "Enqueueing kernel");

    // One output per workgroup
    float* h_partial_sums = (float*)malloc(sizeof(float) * num_groups);
    err = clEnqueueReadBuffer(commands, d_partial_sums, CL_TRUE, 0, sizeof(float) * num_groups, h_partial_sums, 0, NULL, NULL);
    checkError(err, "Reading back d_partial_sums");

    printf("Local sums: ");
    for (int i = 0; i < num_groups; i++)
        printf("%f  ", h_partial_sums[i]);
    printf("\n");

    // The final reduction putting the workgroup results
    // together is done by the CPU.
    float global_sum = 0;
    for (int i = 0; i < num_groups; i++)
        global_sum += h_partial_sums[i]/(float) num_groups;
    // Aplica coreccion porque los 0 que se agregan a los vectores siempre caen dentro del set
    global_sum = ((global_sum * (float)n) - (float)n_pad) / (float)size_vector;
    global_sum = global_sum * 16;
    printf("Calculated sum: %f\n", global_sum);

    // Shutdown and cleanup
    clReleaseMemObject(d_partial_sums);
    clReleaseProgram(program);
    clReleaseKernel(kernel_reduce);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    clReleaseDevice(device);
    free(h_partial_sums);
    free(h_a);
    free(h_b);
    free(a_random);
    free(b_random);

    return 0;
}
