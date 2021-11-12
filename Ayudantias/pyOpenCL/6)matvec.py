import time as tm
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

platform = cl.get_platforms()[0]
device = platform.get_devices(cl.device_type.GPU)[0]
print("Platform name:", platform.name)
print("Device name:", device.name)
print("Maximum work group size:", device.max_work_group_size)
ctx = cl.Context()
queue = cl.CommandQueue(ctx)

workgroup_size = 2**5
n_workgroups = 8
matrix_size = workgroup_size * n_workgroups
print("\nWorkgroup size:", workgroup_size)
print("Matrix size:", matrix_size)

kernel = """
__kernel void matvec(__global const double *matrix,
                     __global const double *vector,
                     __global double *output)
{
  int dim = get_global_size(0);
  int global_id = get_global_id(0);

  double local_sum = 0.0;
  int offset = global_id * dim;
  for(int i = 0; i < dim; i++){
    local_sum += matrix[i + offset] * vector[i];
  }

  output[global_id] = local_sum;
}
"""

prg = cl.Program(ctx, kernel).build()

np.random.seed(0)
random_matrix = np.random.rand(matrix_size, matrix_size)
random_vector = np.random.rand(matrix_size)

## Calculate the matvec with Numpy.
start = tm.time()
matvec_np = random_matrix @ random_vector
time_np = tm.time() - start

## Calculate the matvec with OpenCL, in single precision.

start = tm.time()

cl_matrix = cl_array.to_device(queue, random_matrix.ravel())
cl_vector = cl_array.to_device(queue, random_vector)
cl_output = cl_array.empty(queue, matrix_size, dtype=np.float64)

event = prg.matvec(queue,
                   (matrix_size,), (workgroup_size,),
                   cl_matrix.data, cl_vector.data, cl_output.data)

matvec_cl = cl_output.get()

time_cl = tm.time() - start

print(matvec_np[:5])
print(matvec_cl[:5])

print("\nElapsed time Numpy :", time_np)
print("Elapsed time OpenCL:", time_cl)
