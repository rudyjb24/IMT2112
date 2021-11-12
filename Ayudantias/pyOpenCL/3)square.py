import numpy as np
import pyopencl as cl

platform = cl.get_platforms()[0]
device = platform.get_devices(cl.device_type.GPU)[0]
ctx = cl.Context()
queue = cl.CommandQueue(ctx)

n = 10
h_array_input = np.arange(n, dtype=np.int32)
h_array_output = np.zeros(n, dtype=np.int32)

d_array_input = cl.Buffer(ctx,
                          cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                          hostbuf=h_array_input)
d_array_output = cl.Buffer(ctx,
                           cl.mem_flags.WRITE_ONLY,
                           h_array_output.nbytes)

kernel = """
__kernel void square(__global int* a, __global int* b) {
    int i = get_global_id(0);
    b[i] = a[i]*a[i];
}
"""

prg = cl.Program(ctx, kernel).build()

event_kernel = prg.square(queue, h_array_input.shape, None, d_array_input, d_array_output)

print("Input array:")
print(h_array_input)
print("Squared values:")
print(h_array_output)
print("\n")
event_copy = cl.enqueue_copy(queue, h_array_output, d_array_output)

print("Input array:")
print(h_array_input)
print("Squared values:")
print(h_array_output)