#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h> // Para Windows/Linux
//#include <OpenCL/opencl.h> // Para MAC

/* Encuentra GPU o CPU */
cl_device_id create_device() 
{

   	cl_platform_id platform;
   	cl_device_id dev;
  	cl_int err;

	/* Identifica la plataforma */
		err = clGetPlatformIDs(1, &platform, NULL);
		if(err != 0) 
		{
			printf("No pude indentificar plataforma");
			exit(1);
		}
	/* Identifica la plataforma */


	/* Accede al device */
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
		if(err == CL_DEVICE_NOT_FOUND) 
		{	
			printf("No pude acceder a la GPU\n");
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
		}
		if(err != 0) 
		{
			printf("No pude acceder a un device\n");
			exit(1);
		}
	/* Accede al device */

	return dev;
}

/* Crea el programa a partir de file y lo compila */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* El archivo que contiene el programa y lo guarda en un buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) 
   {
      printf("No encontre el archivo\n");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);
   /* El archivo que contiene el programa y lo guarda en un buffer */


   /* Creamos el programa */
   program = clCreateProgramWithSource(ctx, 1,
      (const char**)&program_buffer, &program_size, &err);
   if(err != 0) 
   {
      printf("No pude crear el programa");
      exit(1);
   }
   free(program_buffer);
   /* Creamos el programa */


   /* Construimos el programa */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err != 0) 
   {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }
   /* Construimos el programa */

   return program;
}


int main() 
{

	/* Creamos Data */
		const unsigned int n = 16;
		int *h_a = (int *)malloc(sizeof(int) * n);
		int *h_b = (int *)malloc(sizeof(int) * n);
		int *h_c = (int *)malloc(sizeof(int) * n);
		int *h_d = (int *)malloc(sizeof(int) * n);
		for (int i = 0; i < n; i++) 
		{
			h_a[i] = 0;
			h_b[i] = 0;
			h_c[i] = 0;
			h_d[i] = 0;
		}
	/* Creamos Data */


	/* Variables OpenCL clasicas */
		cl_device_id device;
		cl_context context;
		cl_program program;
		cl_kernel kernel;
		cl_command_queue queue;
		cl_int err;
	/* Variables OpenCL clasicas */


	/* Creamos Device y Context */
		device = create_device();
		context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   		if(err != 0) 
		{
      		printf("No pude crear el contexto\n");
      		exit(1);
   		}
	/* Creamos Device y Context*/


	/* Crea el programa */
		program = build_program(context, device, "opencl-id-rudy.cl");
	/* Crea el programa */


	/* Crea el queue */
   		queue = clCreateCommandQueue(context, device, 0, &err);
		if(err != 0) 
		{
			printf("No pude crear el queue\n");
			exit(1);
		};
	/* Crea el queue */


	/* Crea el kernel */
		kernel = clCreateKernel(program, "getmyid", &err);
   		if(err != 0) 
		{
      		printf("No pude crear el kernel\n");
      		exit(1);
   		};
	/* Crea el kernel */


	/* Reservar memoria en el device */
		cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &err);
		cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &err);
		cl_mem d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &err);
		cl_mem d_d = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(int), NULL, &err);
	/* Reservar memoria en el device */


	/* Le damos los argumentos al kernel */
		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_a);
		err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_b);
		err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_c);
		err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_d);
		err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&n);
	/* Le damos los argumentos al kernel */


	/* Ejecutamos el kernel en chuncks de localsize */
		size_t localSize = 8;
		size_t globalSize = 16;
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
   	/* Ejecutamos el kernel en chuncks de localsize */


	/* Leemos el output */
		err = clEnqueueReadBuffer(queue, d_a, CL_TRUE, 0, n*sizeof(int), (void*)h_a, 0, NULL, NULL);
		err = clEnqueueReadBuffer(queue, d_b, CL_TRUE, 0, n*sizeof(int), (void*)h_b, 0, NULL, NULL);
		err = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, n*sizeof(int), (void*)h_c, 0, NULL, NULL);
		err = clEnqueueReadBuffer(queue, d_d, CL_TRUE, 0, n*sizeof(int), (void*)h_d, 0, NULL, NULL);
	/* Leemos el output */



	/* Prints */
		printf("Global thread IDs:\n");
		for (int i = 0; i < n; i++) {
			printf("%i  ", h_a[i]);
		}
		printf("\n");
		printf("Local thread IDs:\n");
		for (int i = 0; i < n; i++) {
			printf("%i  ", h_b[i]);
		}
		printf("\n");
		printf("Block thread IDs:\n");
		for (int i = 0; i < n; i++) {
			printf("%i  ", h_c[i]);
		}
		printf("\n");
		printf("Local sizes:\n");
		for (int i = 0; i < n; i++) {
			printf("%i  ", h_d[i]);
		}
		printf("\n");
	/* Prints */
	
    
	/* Limpieza */
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseMemObject(d_a);
		clReleaseMemObject(d_b);
		clReleaseMemObject(d_c);
		clReleaseMemObject(d_d);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		clReleaseDevice(device);
		free(h_a);
		free(h_b);
		free(h_c);
		free(h_d);
	/* Limpieza */


	return 0;
}

