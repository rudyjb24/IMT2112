# instalar PIL con 
# python3 -m pip install --upgrade Pillow
from PIL import Image, ImageDraw
import numpy as np
import pyopencl as cl
import time as tm

# iteraciones, cantidad de puntos, cantidad de grupos y tamaño de grupos
iteracion = 1000
n = 10000
cant_grupos = 100
m = int(n/cant_grupos)
# 0.0178792 
WIDTH = 600
HEIGHT = 600

im = Image.new('RGB', (WIDTH, HEIGHT), (255, 255, 255))
draw = ImageDraw.Draw(im)

# generacion de vectores, como imaginario del tipo a + bi
a_h = np.random.uniform(-2, 1, n).astype(np.float32)
b_h = np.random.uniform(-1.5, 1.5, n).astype(np.float32)
afinal_h = np.zeros(n).astype(np.float32)
bfinal_h = np.zeros(n).astype(np.float32)
temporal_h = np.zeros(n).astype(np.float32)
norma_h = np.zeros(n).astype(np.float32)
cuenta_h = np.zeros(n).astype(np.float32)
suma_h = np.zeros(cant_grupos).astype(np.float32)

plataforms = cl.get_platforms()
print("Plataformas: ", plataforms)

gpu_devices = plataforms[0].get_devices(cl.device_type.GPU)
print("Devices: ", gpu_devices)

context = cl.Context(gpu_devices)
queue = cl.CommandQueue(context)

a_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_h)
b_d = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_h)

afinal_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, a_h.nbytes)
bfinal_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b_h.nbytes)
cuenta_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, cuenta_h.nbytes)
norma_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, norma_h.nbytes)
suma_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, suma_h.nbytes)
temporal_d = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, temporal_h.nbytes)

program1 = cl.Program(context, 
"""
__kernel void iter_mandelbrot(
    __global float *a_d, __global float *b_d, __global float *afinal_d,
    __global float *bfinal_d, __global float *norma_d, __global float *cuenta_d,
    __global float *suma_d, __global float *temporal_d, int iteracion)
{
  int global_id = get_global_id(0);
  int local_id = get_local_id(0);
  int group_id = get_group_id(0); 
  int local_size = get_local_size(0);
  
  for (int i=0;i<iteracion;++i)
  {
    temporal_d[global_id] = afinal_d[global_id];

    afinal_d[global_id] = afinal_d[global_id]*afinal_d[global_id] - bfinal_d[global_id]*bfinal_d[global_id]
                          + a_d[global_id];
    
    bfinal_d[global_id] = 2*temporal_d[global_id]*bfinal_d[global_id] + b_d[global_id];
  }

  norma_d[global_id] = afinal_d[global_id]*afinal_d[global_id] + bfinal_d[global_id]*bfinal_d[global_id];

  bool is = isnan( norma_d[global_id] );

  if(norma_d[global_id] <= 4)
  {
    cuenta_d[global_id] = 1;
  }
}
"""
).build()

program2 = cl.Program(context, 
"""
__kernel void dot_product(
    __global float *cuenta_d, __global float *suma_d)
{
  int global_id = get_global_id(0);
  int local_id = get_local_id(0);
  int group_id = get_group_id(0); 
  size_t local_size = get_local_size(0);

  if (local_id == 0)
  {
    for (int i=0;i<local_size;++i)
    {
      suma_d[group_id] += cuenta_d[global_id + i];
    }
  }

}
"""
).build()

t0_GPU = tm.time()
program1.iter_mandelbrot(queue, (n,), (m,), a_d, b_d, afinal_d, bfinal_d, norma_d, 
                         cuenta_d, suma_d, temporal_d, np.int32(iteracion))
cl.enqueue_copy(queue, cuenta_h, cuenta_d)

program2.dot_product(queue, (n,), (m,), cuenta_d, suma_d)
cl.enqueue_copy(queue, suma_h, suma_d)
t1_GPU = tm.time()
print("\nValor GPU: Tiempo de ejecucion: {}".format( t1_GPU-t0_GPU))

pixel_in = np.sum(suma_h)
p_i = np.sum(cuenta_h)
print("Pixeles dentro:", pixel_in)
print("cuenta", p_i)

area_total = 3 * 3
proporcion = pixel_in / n

area_mandelbrot = area_total * proporcion
print("Area Mandelbrot calculada:", area_mandelbrot)

error = abs(area_mandelbrot -1.50659177)
print("Error:", error)
'''
for i in range(n):
  if cuenta_h[i] == 1:
    draw.point([a_h[i]*200+400, b_h[i]*200+300], (0, 0, 0))

im.save('output.png', 'PNG')

'''