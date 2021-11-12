import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
PYOPENCL_CTX='0'

platform = cl.get_platforms()[0]
device = platform.get_devices(cl.device_type.GPU)[0]
print("Platform name:", platform.name)
print("Device name:", device.name)
print("Maximum work group size:", device.max_work_group_size)

ctx = cl.Context()
queue = cl.CommandQueue(ctx)

kernel = """
__kernel void get_id(__global int *a,
                     __global int *b,
                     __global int *c,
                     __global int *d)
{
  int id = get_global_id(0);

  a[id] = get_global_id(0);
  b[id] = get_local_id(0);
  c[id] = get_group_id(0);
  d[id] = get_local_size(0);
}
"""

prg = cl.Program(ctx, kernel).build()

workgroup_size = 6
n_workgroups = 4
n_vector = workgroup_size * n_workgroups

cl_global_id = cl_array.empty(queue, n_vector, dtype=np.int32)
cl_local_id = cl_array.empty(queue, n_vector, dtype=np.int32)
cl_group_id = cl_array.empty(queue, n_vector, dtype=np.int32)
cl_local_size = cl_array.empty(queue, n_vector, dtype=np.int32)

event = prg.get_id(queue,
                   (n_vector,),
                   (workgroup_size,),
                   cl_global_id.data,
                   cl_local_id.data,
                   cl_group_id.data,
                   cl_local_size.data)

print("Global ID:")
print(cl_global_id)
print("Local ID:")
print(cl_local_id)
print("Group ID:")
print(cl_group_id)
print("Local size:")
print(cl_local_size)