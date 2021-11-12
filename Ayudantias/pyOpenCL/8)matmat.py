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

workgroup_size = 2**4
n_workgroups = 8
matrix_size = workgroup_size * n_workgroups
print("\nWorkgroup size:", workgroup_size)
print("Matrix size:", matrix_size)

kernel = """
__kernel void matmat(__global const double *inputA,
                     __global const double *inputB,
                     __global double *outputC)
{
  int n_cols = get_global_size(1);
  int global_id_row = get_global_id(0);
  int global_id_col = get_global_id(1);

  double local_sum = 0.0;
  for(int i = 0; i < n_cols; i++){
    local_sum += inputA[global_id_row * n_cols + i] * inputB[global_id_col + i * n_cols];
  }

  outputC[global_id_col + global_id_row * n_cols] = local_sum;
}
"""

prg = cl.Program(ctx, kernel).build()

np.random.seed(0)
np_matrix_A = np.random.rand(matrix_size, matrix_size)
np_matrix_B = np.random.rand(matrix_size, matrix_size)

## Calculate the matmat with Numpy.
start = tm.time()

matmat_np = np_matrix_A @ np_matrix_B

time_np = tm.time() - start

## Calculate the matmat with OpenCL.

start = tm.time()

cl_matrix_A = cl_array.to_device(queue, np_matrix_A.ravel())
cl_matrix_B = cl_array.to_device(queue, np_matrix_B.ravel())
cl_output = cl_array.empty(queue, matrix_size*matrix_size, dtype=np.float64)

event = prg.matmat(queue,
                   (matrix_size, matrix_size), (workgroup_size, workgroup_size),
                   cl_matrix_A.data, cl_matrix_B.data, cl_output.data)

matmat_cl = cl_output.get().reshape(matrix_size, matrix_size)

time_cl = tm.time() - start

print(matmat_np[:5,:5])
print(matmat_cl[:5,:5])

print("\nElapsed time Numpy :", time_np)
print("Elapsed time OpenCL:", time_cl)
