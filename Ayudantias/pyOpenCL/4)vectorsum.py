import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

platform = cl.get_platforms()[0]
device = platform.get_devices(cl.device_type.GPU)[0]
ctx = cl.Context()
queue = cl.CommandQueue(ctx)

n = 1000
np_a = np.random.rand(n)
np_b = np.random.rand(n)

print("Variable np_a is of type:", type(np_a))
print("Variable np_b is of type:", type(np_b))

cl_a = cl_array.to_device(queue, np_a)
cl_b = cl_array.to_device(queue, np_b)
cl_c = cl_array.empty_like(cl_a)

print("\nVariable cl_a is of type:", type(cl_a))
print("Variable cl_b is of type:", type(cl_b))
print("Variable cl_c is of type:", type(cl_c))

kernel = """
__kernel void sum(__global const double *a,
                  __global const double *b,
                  __global double *c)
{
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
}
"""

prg = cl.Program(ctx, kernel).build()

event = prg.sum(queue, cl_a.shape, None, cl_a.data, cl_b.data, cl_c.data)

print("\nVariable cl_a.data is of type:", type(cl_a.data))
print("Variable cl_b.data is of type:", type(cl_b.data))
print("Variable cl_c.data is of type:", type(cl_c.data))

print("\na =", cl_a[:4])
print("b =", cl_b[:4])
print("c =", cl_c[:4])