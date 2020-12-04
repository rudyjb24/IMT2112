import time as tm
import numpy as np
import pyopencl as cl


n = 10000
m = 1000



r_h = np.random.rand(n).astype(np.float32)*4-2 ##ponderamos los valores entre -2 y 2
c_h = np.random.rand(n).astype(np.float32)*4-2



plataforms = cl.get_platforms()
print("Plataformas: ", plataforms)

gpu_devices = plataforms[0].get_devices(cl.device_type.GPU)
print("Devices: ", gpu_devices)

context = cl.Context(gpu_devices)
queue = cl.CommandQueue(context)

r_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = r_h)
c_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = c_h)


xr_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, r_h.nbytes)
xc_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, c_h.nbytes)
sup_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, c_h.nbytes)

resultados = np.zeros(n).astype(np.float32)


program = cl.Program(context, 

"""
__kernel void Mandelbrot(
    __global float *r_d, __global float *c_d, __global float *xc_d, __global float *xr_d,__global float *sup_d)
{
  int global_id = get_global_id(0);
  int local_id = get_local_id(0);
  int group_id = get_group_id(0); 
  int local_size = get_local_size(0);
  int i = 0;
  while (i< 1000)
  {
    if (i == 0)
    {   
        xc_d[global_id] = c_d[global_id];
        xr_d[global_id] = (r_d[global_id] * r_d[global_id]);
    }
    float temp = 2*(xr_d[global_id] * xr_d[global_id]) +  c_d[global_id];
    
    xr_d[global_id] = (xr_d[global_id]*xr_d[global_id]) + (xc_d[global_id]*xc_d[global_id]);
    xc_d[global_id] = temp; 
    
    if ((xc_d[global_id]*xc_d[global_id] +  xr_d[global_id]*xr_d[global_id]) >= 4)
    {
        sup_d[global_id] = 0;
        break;
    }
    sup_d[global_id] = 1;
    i += 1;
  }

}
"""
).build()

t0_GPU = tm.time()
program.Mandelbrot(queue, (n,), (m,), r_d, c_d, xc_d, xr_d,sup_d)
cl.enqueue_copy(queue, resultados, sup_d)
superficie = 16*np.sum(resultados)/n 
#ponderamos el valor de los resultados ya que estamos viendo si un valor pertenece a un cuadrado de lado 4 (area 16)
t1_GPU = tm.time()
print("\nValor GPU: {} Tiempo de ejecucion: {}".format(superficie, t1_GPU-t0_GPU))
