import time as tm
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

platform = cl.get_platforms()[0]
device = platform.get_devices(cl.device_type.GPU)[0]
print("Maximum work group size:", device.max_work_group_size)
ctx = cl.Context()
queue = cl.CommandQueue(ctx)

workgroup_size = 2**8
n_workgroups = 4
vector_size = workgroup_size * n_workgroups
print("\nWorkgroup size:", workgroup_size)
print("Vector size:", vector_size)

kernel = """
__kernel void sum1(__global const long int *vec,
                   __global long int *partial_sums)
{
  int group_size = get_local_size(0);
  int local_id = get_local_id(0);
  int group_id = get_group_id(0);
  int global_id = get_global_id(0);

  if (local_id == 0){
    long int sum = 0;
    for(int i = 0; i < group_size; i++){
      sum += vec[global_id + i];
    }
    partial_sums[group_id] = sum;
  }
}
__kernel void sum2(__global long int *vec,
                   __global long int *partial_sums)
{
  int group_size = get_local_size(0);
  int local_id = get_local_id(0);
  int group_id = get_group_id(0);
  int global_id = get_global_id(0);

  int step = 2;
  while (step <= group_size) {
    if (local_id % step == 0) {
      vec[global_id] += vec[global_id + step / 2];
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    step *= 2;
  }
  if (local_id == 0){
    partial_sums[group_id] = vec[global_id];
  }
}
"""

prg = cl.Program(ctx, kernel).build()

np_vector = np.arange(vector_size, dtype=np.int64)
cl_vector = cl_array.to_device(queue, np_vector)

cl_partial_sums1 = cl_array.empty(queue, n_workgroups, dtype=np.int64)
cl_partial_sums2 = cl_array.empty(queue, n_workgroups, dtype=np.int64)

t0 = tm.time()
event = prg.sum1(queue,
                 (vector_size,),
                 (workgroup_size,),
                 cl_vector.data,
                 cl_partial_sums1.data
                )
t1 = tm.time()
event = prg.sum2(queue,
                 (vector_size,),
                 (workgroup_size,),
                 cl_vector.data,
                 cl_partial_sums2.data
                )
t2 = tm.time()
np_partial_sums1 = cl_partial_sums1.get()
vector_sum1 = np.sum(np_partial_sums1)
np_partial_sums2 = cl_partial_sums2.get()
vector_sum2 = np.sum(np_partial_sums2)

print("The sum calculated by OpenCL:", vector_sum1, "and execution time: ", t1-t0)
print("The sum calculated by OpenCL:", vector_sum2, "and execution time: ", t2-t1)
print("The exact value of the sum:  ", vector_size*(vector_size-1)//2)