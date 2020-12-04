import math
import numpy as np
import pyopencl as cl


np.random.seed(2)
n = 100000                    # Vector size.
m = 100                       # Group size.
ch = 100                      # Chunk size.
g = math.ceil(n / (m * ch))   # Number of groups.
T = 10000                     # Number of iterations.

# Generamos 2 n números aleatorios entre -2 y 2.
c1_h = np.random.uniform(-2, 2, n).astype(np.float32)
c2_h = np.random.uniform(-2, 2, n).astype(np.float32)

partial_results_h = np.zeros(g).astype(np.float32)
final_h = np.zeros(1).astype(np.float32)

platforms = cl.get_platforms()
gpu_devices = platforms[0].get_devices(cl.device_type.GPU)

context = cl.Context(gpu_devices)
queue = cl.CommandQueue(context)

c1_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=c1_h)
c2_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=c2_h)

partial_results_d = cl.Buffer(context, cl.mem_flags.READ_WRITE, partial_results_h.nbytes)
final_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, final_h.nbytes)

program = cl.Program(context,
"""
__kernel void in_set(
    __global float *c1_d,
    __global float *c2_d,
    __global float *partial_results_d,
    unsigned ch_d,
    unsigned T,
    __local float *local_results_d)
{
    int group_size = get_local_size(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int global_id = get_global_id(0);

    int start_id_group = group_id * group_size * ch_d;
    int start_id_thread = start_id_group + local_id * ch_d;
    local_results_d[local_id] = 0;

    float real;
    float im;
    float new_r;
    float new_i;

    int belongs;

    for (int i = 0; i < ch_d; i++)
    {
        real = 0;
        im = 0;
        new_r = 0;
        new_i = 0;

        belongs = 1;

        for (int t = 0; t < T; t++)
        {
            new_r = (real + c1_d[global_id]);
            new_i = (im + c2_d[global_id]);

            if (pow(new_r, 2) + pow(new_i, 2) > 4)
            {
                belongs = 0;
                break;
            }

            real = pow(new_r, 2) - pow(new_i, 2);
            im = 2 * new_r * new_i;
        }

        local_results_d[local_id] += belongs;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0)
    {
        int sum = 0;
        for (int i = 0; i < group_size; i++)
        {
            sum += local_results_d[i];
        }
        partial_results_d[group_id] = sum;
    }
}

__kernel void reduce(
    __global float *partial_results_d,
    __global float *final_d,
    float n)
{
  int global_id = get_global_id(0);
  int global_size = get_global_size(0);

  if (global_id == 0) {
    float sum = 0;
    for (int i = 0; i < global_size; i++)
    {
      sum += partial_results_d[i];
    }
    final_d[0] = (sum / n) * 16;
  }
}
"""
).build()

program.in_set(queue, (n,), (m,), c1_d, c2_d, partial_results_d, np.uint32(ch), np.uint32(T), cl.LocalMemory(4 * m))
program.reduce(queue, (g,), (m,), partial_results_d, final_d, np.float32(n))
cl.enqueue_copy(queue, final_h, final_d)

correcto = 1.50659177
resultado = final_h[0]
error = abs(correcto - resultado)

print("Resultado:", resultado)
print("Error:", error)
print("Error (%):", error * 100 / correcto)
