import time as tm
import numpy as np
import pyopencl as cl

n = 100
m = 10
a_h = np.random.rand(n).astype(np.float32)
b_h = np.random.rand(n).astype(np.float32)
#a_h = np.ones(n).astype(np.float32)
#b_h = np.ones(n).astype(np.float32)
d_h = np.zeros(m).astype(np.float32)

plataforms = cl.get_platforms()
print("Plataformas: ", plataforms)

gpu_devices = plataforms[0].get_devices(cl.device_type.GPU)
print("Devices: ", gpu_devices)

context = cl.Context(gpu_devices)
queue = cl.CommandQueue(context)

a_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_h)
b_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_h)

c_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a_h.nbytes)
d_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, 4*m)
# Los Flags pueden ser READ_WRITE, READ_ONLY o WRITE_ONLY

program = cl.Program(context, 
"""
__kernel void dot_product(
    __global float *a_d, __global float *b_d, __global float *c_d, __global float *d_d)
{
  int global_id = get_global_id(0);
  int local_id = get_local_id(0);
  int group_id = get_group_id(0); 
  int local_size = get_local_size(0);
  c_d[global_id] = a_d[global_id] * b_d[global_id];

  if (local_id == 0)
  {
    for (int i=0;i<local_size;++i)
    {
      d_d[group_id] += c_d[global_id + i];
    }
  }

}
"""
).build()

t0_sequencial = tm.time()
producto_punto = 0
for i in range(len(a_h)):
  producto_punto += a_h[i] * b_h[i]
t1_sequencial = tm.time()
print("\nValor Sequencial: {} Tiempo de ejecucion: {}".format(producto_punto, t1_sequencial-t0_sequencial))

t0_GPU = tm.time()
program.dot_product(queue, (n,), (m,), a_d, b_d, c_d, d_d)
cl.enqueue_copy(queue, d_h, d_d)
producto_punto = np.sum(d_h)
t1_GPU = tm.time()
print("\nValor GPU: {} Tiempo de ejecucion: {}".format(producto_punto, t1_GPU-t0_GPU))

t0_np = tm.time()
producto_punto = np.dot(a_h, b_h)
t1_np = tm.time()
print("\nValor Numpy: {} Tiempo de ejecucion: {}".format(producto_punto, t1_np-t0_np))
