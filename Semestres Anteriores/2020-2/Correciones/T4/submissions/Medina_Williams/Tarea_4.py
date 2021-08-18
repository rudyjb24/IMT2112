import numpy as np
import pyopencl as cl
import time as tm

n = 100 #Cantidad de c_k a genenrar
m = 10 #Tamaño de cada grupo
max_iteraciones = 10000 #Cantidad máxima de iteraciones para verificar convergencia

np.random.seed(16)
#Complejos de la forma c_k = a + bi
a = np.random.uniform(low=-2, high=2, size=(n,)).astype(np.float32)
b = np.random.uniform(low=-2, high=2, size=(n,)).astype(np.float32)
d_h = np.zeros(m).astype(np.float32)


plataforms = cl.get_platforms() #Lista de plataformas
print("Plataformas: ", plataforms)

gpu_devices = plataforms[0].get_devices(cl.device_type.GPU) #Vemos que gpu tenemos
print("Devices: ", gpu_devices)

context = cl.Context(gpu_devices)
queue = cl.CommandQueue(context)


#Creamos los buffers
#Guardamos en gpu a
a_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)

#Guardamos en gpu b
b_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
i_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a.nbytes)

#Vector con valor 1 si módulo > 2, 0 e.o.c
d_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, 4*m)

program = cl.Program(context,
"""
__kernel void pixel_counting(
    __global float *a_d, __global float *b_d, __global float *d_d,__global float *i_d, int max_iteraciones)
{
  int global_id = get_global_id(0);
  int local_id = get_local_id(0);
  int group_id = get_group_id(0);
  int local_size = get_local_size(0);
  float a_z = 0;
  float b_z = 0;
  float sig_a = 0;
  float sig_b = 0;
  float z_n = 0;

  for (int i=0;i < max_iteraciones ;++i)
  {
    sig_a = pow(a_z,2) - pow(b_z,2) + a_d[global_id];

    sig_b = 2 * a_z * b_z + b_d[global_id];

    z_n  = pow(a_z,2) + pow(b_z,2);

    a_z = sig_a;
    b_z = sig_b;

    if (z_n > 4)
    {
        i_d[global_id] = 1;
        break;
    }
    else if (i == max_iteraciones-1)
    {
        i_d[global_id] = 0;
    }
  }
    //Acá sumamos todos los 1 = converge y los guardamos en un vector


  if (local_id == 0)
  {
    for (int i=0;i < local_size; ++i)
    {
       d_d[group_id] += i_d[global_id + i];
    }

  }

}
"""
).build()



#resultado = np.array([0]).astype(np.float32)
program_2 = cl.Program(context,
"""
__kernel void reduction(
    __global float *d_h, __global float *resultado, int n)
{
  int global_id = get_global_id(0);
  int local_id = get_local_id(0);
  int group_id = get_group_id(0);
  int local_size = get_local_size(0);
  float valor = 0;

  if (local_id == 0)
  {
    for (int i=0;i<local_size;++i)
    {
      valor += d_h[i];
    }
    resultado[0] = 16-16*(valor/n);
  }

}
"""
).build()

t0_GPU = tm.time()
program.pixel_counting(queue, (n,), (m,), a_d, b_d, d_d,i_d, np.int32(max_iteraciones))
cl.enqueue_copy(queue, d_h, d_d)


vector_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=d_h)

#proporcion = 16*(np.sum(d_h))/n


resultado = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, 4*m)
prob =  np.zeros(m).astype(np.float32)

program_2.reduction(queue, (n,), (m,), vector_d, resultado, np.int32(n))
cl.enqueue_copy(queue, prob, resultado)
t1_GPU = tm.time()
print("\nTiempo de ejecucion: {}".format(t1_GPU-t0_GPU))
print("El área estimada fue {} con un error de aproximación de {} en {} iteraciones.".format(prob[0],1.50659177- prob[0], max_iteraciones))

t0_sequencial = tm.time()
fallos  = 0
for i in range(n):
    a_z = 0
    b_z = 0
    z_n = 0
    for j in range(max_iteraciones):
        sig_a = a_z**2 - b_z**2 + a[i]
        sig_b = 2*a_z*b_z + b[i]
        z_n = a_z**2 + b_z**2
        a_z = sig_a
        b_z = sig_b
        if z_n > 4:
            fallos +=1
            break

area = 16*(n-fallos)/n
t1_sequencial = tm.time()
print("\nTiempo de ejecucion: {}".format(t1_sequencial-t0_sequencial))
print("El método secuencial estima un área de {} con un error de {} en {} iteraciones".format(area,1.50659177- area, max_iteraciones))
