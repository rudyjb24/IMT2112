#include <mpi.h>
#include <stdio.h>
#include <math.h> 

float alpha(float x,float y)
{
    float z = x * y * (x-1) * (y-1) + 1;
    return z;
}

void print_vector(float *vector, int ny)
{
  printf("\n");
  for (int i = 0; i < ny; i++) 
  {
      printf("%f\n", vector[i]);
  }
}

void print_matriz(float** matrix, int nx, int ny)
{
  printf("\n");
  for (int i = 0; i < nx; i++) 
  {
      for (int j = 0; j < ny; j++)
      {
         printf("%f ", matrix[i][j]); 
      }
    printf("\n");
  }
}

void free_matriz(float **matrix, int nx)
{
  for (int i = 0; i < nx; i++)
  {
    free(matrix[i]);
  }
  free(matrix);
}


float** Fmatrix(int temp, int ny, int nx,  int world_rank, int len_nx)
{
    float** matrix = (float**) calloc(len_nx+ temp , sizeof(float*));

    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        matrix[i] = (float *) calloc(ny, sizeof(float));
        matrix[i][len_nx] = 10; 
    }

    return matrix;
}


float** Omatrix(int temp, int ny, int nx,  int world_rank, int len_nx, float hx,float paso_x, float paso_y)
{
    float** matrix = (float**) calloc(len_nx+ temp , sizeof(float*));

    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        matrix[i] = (float *) calloc(ny, sizeof(float));
    }

    for (int i = 0; i < len_nx+ temp ; i++)
    {
            for (int j = 1; j < ny; j++)
            {
                matrix[i][j] = -alpha(paso_x *(i+len_nx*world_rank -0.5) , paso_y * (j)) / hx; 
            }
    }
    return matrix;
}

float** Ematrix(int temp, int ny, int nx,  int world_rank, int len_nx, float hx,float paso_x, float paso_y)
{
    float** matrix = (float**) calloc(len_nx+ temp , sizeof(float*));

    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        matrix[i] = (float *) calloc(ny, sizeof(float));
    }

    for (int i = 0; i < len_nx+ temp ; i++)
    {
            for (int j = 0; j < ny-1; j++)
            {
                matrix[i][j] = -alpha(paso_x *(i+len_nx*world_rank + 0.5) , paso_y * (j)) / hx; 
            }
    }
    return matrix;
}

float** Nmatrix(int temp, int ny, int nx,  int world_rank, int len_nx, float hy, int is_0,float paso_x, float paso_y)
{
    float** matrix = (float**) calloc(len_nx+ temp , sizeof(float*));

    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        matrix[i] = (float *) calloc(ny, sizeof(float));
    }

    for (int i = is_0; i < len_nx+ temp ; i++)
    {
            for (int j = 0; j < ny; j++)
            {
                matrix[i][j] = -alpha(paso_x *(i+len_nx*world_rank) , paso_y * (j + 0.5)) / hy; 
            }
    }
    return matrix;
}

float** Smatrix(int temp, int ny, int nx,  int world_rank, int len_nx, float hy, int is_terminal,float paso_x, float paso_y)
{
    float** matrix = (float**) calloc(len_nx+ temp , sizeof(float*));

    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        matrix[i] = (float *) calloc(ny, sizeof(float));
    }

    for (int i = 0; i < len_nx+ temp- is_terminal ; i++)
    {
            for (int j = 0; j < ny; j++)
            {
                matrix[i][j] = -alpha(paso_x *(i+len_nx*world_rank) , paso_y * (j - 0.5)) / hy; 
            }
    }
    return matrix;
}

float** Cmatrix(int temp, int ny, int nx,  int world_rank, int len_nx, float hy,float hx, int is_0, int is_terminal,float paso_x, float paso_y)
{
    float** matrix = (float**) calloc(len_nx+ temp , sizeof(float*));

    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        matrix[i] = (float *) calloc(ny, sizeof(float));
    }

    for (int i = 0; i < len_nx+ temp ; i++)
    {
            for (int j = 0; j < ny; j++)
            {
                matrix[i][j] = (alpha(paso_x *(i+len_nx*world_rank) , paso_y * (j - 0.5)) + alpha(paso_x *(i+len_nx*world_rank) , paso_y * (j + 0.5)))/ hy + (alpha(paso_x *(i+len_nx*world_rank + 0.5) , paso_y * (j)) + alpha(paso_x *(i+len_nx*world_rank - 0.5) , paso_y * (j)) )/ hx  +1; 
            }
    }
    return matrix;
}


float** onesmatrix(int temp, int ny, int nx, int len_nx, int is_0, int is_terminal)
{
    float** matrix = (float**) calloc(len_nx+ temp , sizeof(float*));

    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        matrix[i] = (float *) calloc(ny, sizeof(float));
    }

    for (int i = 0; i < len_nx+ temp ; i++)
    {
            for (int j = 0; j < ny; j++)
            {
                matrix[i][j] = 1; 
            }
    }
    return matrix;
}

float dot_product(float** vect1,float** vect2, int len_nx, int ny, int temp,  int is_0, int is_terminal,int world_rank,int world_size)
{
    float parcial_sum = 0;
    float err;
    float receiveBuffer;

    // Calculo local de productos punto en una suma parcial
    for (int i = 0; i < len_nx + temp  ; i++)
    {   
        for (int j = 0; j < ny; j++)
        {
            parcial_sum += vect1[i][j] * vect2[i][j]  ; 
        }
    }

    
    if (world_rank == 0)
    {
        // El procesador 0 recibe todas los resultados parciales y los suma.
        for (int p=1; p<world_size; p++)
        {
            err = MPI_Recv(&receiveBuffer, 1, MPI_FLOAT, p, p, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            parcial_sum += receiveBuffer;
        }

        // El procesador 0 envia el resultado a todos los procesadores.
        for (int p=1; p<world_size; p++)
        {
            err = MPI_Send(&parcial_sum, 1, MPI_FLOAT, p, p, MPI_COMM_WORLD);
        }
    }
    else 
    {
        // El procesador p envia sus resultados parciales a p0
        err = MPI_Send(&parcial_sum, 1, MPI_FLOAT, 0, world_rank, MPI_COMM_WORLD);

        // Se recibe el resultado total desde p0
        err = MPI_Recv(&receiveBuffer, 1, MPI_FLOAT, 0,world_rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        parcial_sum = receiveBuffer;
    }
    


    return parcial_sum;
}

float** ponderacion(float** matrix, float alpha,int temp, int ny, int len_nx, int is_0, int is_terminal)
{
    float** ponderado = (float**) calloc(len_nx+ temp , sizeof(float*));

    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        ponderado[i] = (float *) calloc(ny, sizeof(float));
    }

    for (int i = 0; i < len_nx+ temp ; i++)
    {
            for (int j = 0; j < ny; j++)
            {
                ponderado[i][j] += alpha*matrix[i][j];
                }
    }
    return ponderado;
}

float** suma(float** matrix1, float** matrix2,int temp, int ny, int len_nx, int is_0, int is_terminal)
{
    float** resultado = (float**) calloc(len_nx+ temp , sizeof(float*));

    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        resultado[i] = (float *) calloc(ny, sizeof(float));
    }

    for (int i = 0; i < len_nx+ temp ; i++)
    {
            for (int j = 0; j < ny; j++)
            {
                resultado[i][j] = matrix1[i][j] + matrix2[i][j] ;
                }
    }
    return resultado;
}

float** resta(float** matrix1, float** matrix2,int temp, int ny, int len_nx, int is_0, int is_terminal)
{
    float** resultado = (float**) calloc(len_nx+ temp , sizeof(float*));

    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        resultado[i] = (float *) calloc(ny, sizeof(float));
    }

    for (int i = 0; i < len_nx+ temp ; i++)
    {
            for (int j = 0; j < ny; j++)
            {
                resultado[i][j] = matrix1[i][j] - matrix2[i][j] ;
            }
    }
    return resultado;
}

float** Ax_stencil(float** N,float** O,float** S,float** E,float** C, float** x, float* x_sup, float* x_inf, int temp, int ny, int len_nx, int is_0, int is_terminal)
{
    // Creamos la matriz del tamaño necesario
    float** resultado = (float**) calloc(len_nx+ temp , sizeof(float*));

    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        resultado[i] = (float *) calloc(ny, sizeof(float));
    }

    // Suma en Norte
    // Si el procesador es el 0, no se considera x_sup
    if (is_0 == 0)
    {
        for (int j = 0; j < ny; j++)
                {
                    resultado[0][j] +=  N[0][j]*x_sup[j] ;

                }
        
    }

    for (int i = 1; i < len_nx+ temp  ; i++)
        {
            for (int j = 0; j < ny ; j++)
                {
                    resultado[i][j] +=   N[i][j]*x[i-1][j] ;
                }
        }

    // Suma Sur
    // Si es el ultimo procesador, en la ultima fila no se considera
    if (is_terminal == 0)
    {
        for (int j = 0; j < ny; j++)
                {
                    resultado[len_nx+temp-1][j] += S[len_nx+temp-1][j] * x_inf[j];
                }
        
    }

    for (int i = 0; i < len_nx+ temp-1 ; i++)
        {
            for (int j = 0; j < ny; j++)
                {
                    resultado[i][j] += S[i][j] * x[i+1][j] ;
                }
        }

    //Suma Oeste
    // No se considera el primer nodo pues x[-1][j] = 0
    for (int i = 0; i < len_nx+ temp ; i++)
        {
            for (int j = 1; j < ny; j++)
                {
                    resultado[i][j] +=  O[i][j] * x[i][j-1] ;
                }
        }

    //Suma Este
    // No se considera el primer nodo pues x[maximo+1][j] = 0
    for (int i = 0; i < len_nx+ temp ; i++)
        {
            for (int j = 0; j < ny-1; j++)
                {
                    resultado[i][j] +=  E[i][j] * x[i][j+1] ;
                }
        }

    //Suma Central
    for (int i = 0; i < len_nx+ temp ; i++)
        {
            for (int j = 0; j < ny; j++)
                {
                    resultado[i][j] +=  C[i][j] * x[i][j] ;
                }
        }
    return resultado;
}

float* envios_inf(int world_rank,int world_size,float** x,  int temp, int ny, int len_nx, int is_0, int is_terminal) //envios(world_rank, world_size,N,  temp, ny, len_nx, is_0, is_terminal)
{
    // envios de X_sup
    float* receiveBuffer= (float *) calloc(ny, sizeof(float));
    float* x_inf = (float *) calloc(ny, sizeof(float));
    for (int i = 0; i<ny;i++)
    {   
        x_inf[i] = x[len_nx+ temp-1][i];
    }
    //Primero se envía desde procesadores  pares
    if (world_rank % 2 == 0) 
    {
        // El procesador p envia sus resultados parciales a p0
        if (world_rank < world_size-1)
        {
            MPI_Send(x_inf, ny, MPI_FLOAT, world_rank+1, world_rank, MPI_COMM_WORLD);
        }
        // Se recibe el resultado total desde p0
        if (world_rank >0)
        {
            MPI_Recv(receiveBuffer, ny, MPI_FLOAT, world_rank-1,world_rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        if (world_rank >0)
        {
            MPI_Recv(receiveBuffer, ny, MPI_FLOAT, world_rank-1,world_rank-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (world_rank < world_size-1)
        {
            MPI_Send(x_inf, ny, MPI_FLOAT, world_rank+1, world_rank, MPI_COMM_WORLD);
        }
    }
    return x_inf;
}

float* envios_sup(int world_rank,int world_size,float** x,  int temp, int ny, int len_nx, int is_0, int is_terminal)
{
    // envios de X_inf

    float* receiveBuffer= (float *) calloc(ny, sizeof(float));
    float* x_sup = (float *) calloc(ny, sizeof(float));
    for (int i = 0; i<ny;i++)
    {   
        x_sup[i] = x[0][i];
    }
    
    //Primero se envía desde procesadores  pares
    if (world_rank % 2 == 0) 
    {
        // El procesador p envia sus resultados parciales a p0
        if (world_rank > 0)
        {
            MPI_Send(x_sup, ny, MPI_FLOAT, world_rank-1, world_rank, MPI_COMM_WORLD);
        }
        // Se recibe el resultado total desde p0
        if (world_rank < world_size-1)
        {
            MPI_Recv(receiveBuffer, ny, MPI_FLOAT, world_rank+1,world_rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    else
    {
        if (world_rank < world_size-1)
        {
            MPI_Recv(receiveBuffer, ny, MPI_FLOAT, world_rank+1,world_rank+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (world_rank > 0)
        {
            MPI_Send(x_sup, ny, MPI_FLOAT, world_rank-1, world_rank, MPI_COMM_WORLD);
        }
    }
    return x_sup;
}

float mandar_todos(int world_rank,int world_size, float mensaje)
{
    float r0 = mensaje;
    if (world_rank == 0)
    {
        for (int p = 1; p<world_size;p++)
            {
            MPI_Send(&r0, 1, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
            }
    }
    else
    {
        MPI_Recv(&r0, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return mensaje;
}


int main(int argc, char** argv)
{


    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    int world_size;
    int world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int nx = 10; // Numero particiones en x
    int ny = 10; // Número particiones en y
    int len_nx = ny / world_size;

    float paso_x = 1 / float(nx);
    float paso_y = 1 / float(ny);
    float hx = paso_x * paso_x;
    float hy = paso_y * paso_y;

    // Código especial para primer y último procesador, is_0 y is_terminal dicen si el procesador es el primero o el último (pueden ser ambos)
    // temp, en el caso de que el procesador sea el último asigna la cantidad extra de a generar por ejemplo, si tenemos que generar una matriz
    // de 10 x 10, cada procesador genera 3 filas y el ultimo genera 3 filas + temp (temp = 1).
    int is_0 = 0;
    int is_terminal = 0;
    int temp = 0;
    
    if (world_rank == world_size -1)
    {   
    temp = ny % world_size;
    is_terminal = 1;
    }
    if (world_rank == 0)
    {
    is_0 = 1;
    }
    
    // Generando matrices
    float** N = Nmatrix(temp,ny,nx,world_rank,len_nx,hx,is_0,paso_x,paso_y); 
    float** O = Omatrix(temp,ny,nx,world_rank,len_nx,hy,paso_x,paso_y);
    float** S = Smatrix(temp,ny,nx,world_rank,len_nx,hy,is_terminal,paso_x,paso_y); 
    float** E = Ematrix(temp,ny,nx,world_rank,len_nx,hy,paso_x,paso_y);
    float** C = Cmatrix(temp,ny,nx,world_rank,len_nx,hy,hx,is_0,is_terminal,paso_x,paso_y); 
    //float** f = Fmatrix(temp, ny, nx,  world_rank, len_nx);
    
    float** x = onesmatrix(temp, ny, nx, len_nx, is_0, is_terminal);
    float* x_sup = envios_inf(world_rank, world_size,x,  temp, ny, len_nx, is_0, is_terminal);
    float* x_inf = envios_sup(world_rank, world_size,x,  temp, ny, len_nx, is_0, is_terminal);
   int iteraciones = 100;
    float** r = (float**) calloc(len_nx+ temp , sizeof(float*));
    float** p = (float**) calloc(len_nx+ temp , sizeof(float*));
    float** t = (float**) calloc(len_nx+ temp , sizeof(float*));
    
    for (int i = 0 ;  i < len_nx + temp ; i++)
    {
        p[i] = (float *) calloc(ny, sizeof(float));
        r[i] = (float *) calloc(ny, sizeof(float));
        t[i] = (float *) calloc(ny, sizeof(float));
    }
   float* r_norma = (float *) calloc(iteraciones, sizeof(float));
    float* a = (float *) calloc(iteraciones, sizeof(float));
    float* t_norma = (float *) calloc(iteraciones, sizeof(float));
    float* beta = (float *) calloc(iteraciones, sizeof(float));
     
    r = resta(f,Ax_stencil( N,O,S,E,C, x,  x_sup, x_inf,  temp,  ny,  len_nx,  is_0,  is_terminal ), temp,  ny,  len_nx,  is_0,  is_terminal);
    p = resta(f,Ax_stencil( N,O,S,E,C, x,  x_sup, x_inf,  temp,  ny,  len_nx,  is_0,  is_terminal ), temp,  ny,  len_nx,  is_0,  is_terminal);
    
    r_norma[0] = dot_product(r,r, len_nx, ny,  temp,   is_0,  is_terminal, world_rank, world_size);
    for (int j = 0; j<iteraciones-1; j++)
    {
        float* p_sup = envios_inf(world_rank, world_size,p,  temp, ny, len_nx, is_0, is_terminal);
        float* p_inf = envios_sup(world_rank, world_size,p,  temp, ny, len_nx, is_0, is_terminal);
    
        t = Ax_stencil( N,O,S,E,C, p,  p_sup, p_inf,  temp,  ny,  len_nx,  is_0,  is_terminal );
        t_norma[j] = pow(dot_product(t,p, len_nx, ny,  temp,   is_0,  is_terminal, world_rank, world_size),0.5);
        a[j] = r_norma[j] / t_norma[j];
        x = suma(x,ponderacion( p,  a[j], temp,  ny,  len_nx,  is_0,  is_terminal), temp,  ny,  len_nx,  is_0,  is_terminal);
        float* x_sup = envios_inf(world_rank, world_size,x,  temp, ny, len_nx, is_0, is_terminal);
        float* x_inf = envios_sup(world_rank, world_size,x,  temp, ny, len_nx, is_0, is_terminal);
        r = resta(r,ponderacion( t,  a[j], temp,  ny,  len_nx,  is_0,  is_terminal), temp,  ny,  len_nx,  is_0,  is_terminal);
        r_norma[j+1] = dot_product(r,r, len_nx, ny,  temp,   is_0,  is_terminal, world_rank, world_size);
        beta[j+1] = r_norma[j+1]/r_norma[j];
        p = suma(r,ponderacion( p,  beta[j+1], temp,  ny,  len_nx,  is_0,  is_terminal), temp,  ny,  len_nx,  is_0,  is_terminal);
        if (world_rank ==0)
        {
            print_matriz(p,len_nx+temp,ny);
                printf("%f\n",r_norma[j]);
        }    
    }
    



    free_matriz(N,len_nx+ temp);
    free_matriz(O,len_nx+ temp);
    free_matriz(S,len_nx+ temp);
    free_matriz(E,len_nx+ temp);
    free_matriz(C,len_nx+ temp ;
    free_matriz(x,len_nx+ temp);

    free_matriz(f,len_nx+ temp);
    free_matriz(t,len_nx+ temp);
    free_matriz(r,len_nx+ temp ;
    free_matriz(p,len_nx+ temp);
       
    
    free(x_inf);
    free(x_sup);
    free(r_norma);
    free(t_norma);
    free(a); 
    free(beta);
    
    MPI_Finalize();
}