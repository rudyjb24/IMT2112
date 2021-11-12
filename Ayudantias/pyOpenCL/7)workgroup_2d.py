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

workgroup_size = 4
n_workgroups = 2
matrix_size = workgroup_size * n_workgroups
print("Workgroup size:", workgroup_size)
print("Matrix size:", matrix_size)

kernel = """
__kernel void get_my_id_2d(__global int *row,
                           __global int *col)
{
  int id0 = get_global_id(0);
  int id1 = get_global_id(1);
  int size0 = get_global_size(0);
  int size1 = get_global_size(1);

  int id = id0 * size1 + id1;

  row[id] = get_local_id(0);
  col[id] = get_local_id(1);

}
"""

prg = cl.Program(ctx, kernel).build()

cl_row = cl_array.empty(queue, matrix_size*matrix_size, dtype=np.int32)
cl_col = cl_array.empty(queue, matrix_size*matrix_size, dtype=np.int32)

event = prg.get_my_id_2d(queue, (matrix_size, matrix_size), (workgroup_size, workgroup_size),
                         cl_row.data, cl_col.data)

print("\nLocal id along dimension 0:")
print(cl_row.reshape(matrix_size, matrix_size))
print("\nLocal id along dimension 1:")
print(cl_col.reshape(matrix_size, matrix_size))