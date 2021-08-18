// Compilation: mpic++ main.cpp
// Execution: mpirun -np 3 a.out

#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

//genera los pedazos de vectores que necesitamos.
float** matrix_generator(int Nx, int height)
{
	//se retorna una matriz de Nx x altura.
  float** matrix = (float**) calloc(height, sizeof(float*));

  for (int i = 0; i < height; i++)
  {
    matrix[i] = (float *)calloc(Nx, sizeof(float));

  }
  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < Nx; j++)
    {
      matrix[i][j] = 0;
    }
  }
  return matrix;
}

//calcular alpha(i,j) = i*(i-1)*j*(j-1) + 1
float alpha(float i, float j){
	return i*(i-1)*j*(j-1) + 1;
}

//producto punto local
float local_dot_product(float** arr1, float** arr2, int Nx, int height){
  float dot = 0;
  for (int n=0; n<height; n++){
  for (int m=0; m<Nx; m++){
		dot +=  arr1[n][m]*arr2[n][m];
	 }
  }
  return dot;
}

//producto punto total (junta todos los productos puntos locales).
float parallel_dot_product(float** arr1, float** arr2, int Nx, int height){
  int err;
  float rho_final;
  float rho = local_dot_product(arr1, arr2, Nx, height); //calculamos el rho local
  err = MPI_Allreduce(&rho, &rho_final, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  return rho_final;
}

//vector * escalar
float** reescalate_vector(float** vec, float scale, int Nx, int height){
  for (int n=0; n<height; n++){
  for (int m=0; m<Nx; m++){
		vec[n][m]*= scale;
	 }
  }
  return vec;
}

//suma de vectores
float** sum_of_vectors(float** arr1, float** arr2, int Nx, int height){
  float** sum = matrix_generator(Nx, height);
  for (int n=0; n<height; n++){
  for (int m=0; m<Nx; m++){
		sum[n][m] =  arr1[n][m] + arr2[n][m];
	 }
  }
  return sum;
}

//envia la fila del sur a los procesadores k+1 desde el k, cuando k<= world_size.
float* comunicar_sur(float** p, int Nx, int world_rank, int world_size, int localheight){
  float* sur = (float *)calloc(Nx, sizeof(float));
  if (world_rank == 0){
    //el procesador 0 la fila de más arriba de p al procesador 1:
    MPI_Send(p[localheight-1],Nx,MPI_FLOAT,world_rank+1,world_rank+1,MPI_COMM_WORLD);
  }else if(world_rank == world_size -1){
    //el utlimo recibe pero no le manda a nadie
    MPI_Recv(sur,Nx,MPI_FLOAT,world_rank-1,world_rank,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }else{
    MPI_Recv(sur,Nx,MPI_FLOAT,world_rank-1,world_rank,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Send(p[localheight-1],Nx,MPI_FLOAT,world_rank+1,world_rank+1,MPI_COMM_WORLD);
  }
  return sur;
}

//envia la fila del norte a los procesadores k-1 desde el k, cuando k>1.
float* comunicar_norte(float** p, int Nx, int world_rank, int world_size){
  float* norte = (float *)calloc(Nx, sizeof(float));
  if (world_rank == world_size -1){
    //el ultimo procesador envia pero no recibe
    MPI_Send(p[0],Nx,MPI_FLOAT,world_rank-1,world_rank-1,MPI_COMM_WORLD);
  }else if(world_rank == 0){
    //el primero recibe pero no le manda a nadie
    MPI_Recv(norte,Nx,MPI_FLOAT,world_rank+1,world_rank,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }else{
    //los demas reciben y envian
    MPI_Recv(norte,Nx,MPI_FLOAT,world_rank+1,world_rank,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    MPI_Send(p[0],Nx,MPI_FLOAT,world_rank-1,world_rank-1,MPI_COMM_WORLD);
  }
  return norte;
}

//retorna q = Ap, donde A se representa con los stencils.
float** matvec(float** Norte, float** Sur, float** Este, float** Oeste,
              float** Centro, float** p, int Nx, int localheight,
              int world_rank, int world_size){

  //tenemos que comunicar
  float* sur = comunicar_sur(p, Nx, world_rank, world_size, localheight);
  float* norte = comunicar_norte(p, Nx, world_rank, world_size);

  float** q = matrix_generator(Nx, localheight);
  for (int n=0; n<localheight; n++){
    for (int m=0; m<Nx; m++){
      //siempre se suma el centro
      q[n][m] += Centro[n][m]*p[n][m];
      //si estamos a la izquiera, no hay oeste
      if (m == 0){
        q[n][m] += Este[n][m]*p[n][m+1];
      }
      //si estamos a la derecha, no hay este
      else if (m == Nx - 1){
        q[n][m] += Oeste[n][m]*p[n][m-1];
      }
      //en los otros casos, tenemos a ambos
      else{
        q[n][m] += Oeste[n][m]*p[n][m-1] + Este[n][m]*p[n][m+1];
      }

      //para sumar los norte y sur, depende del procesador en que estemos
      if (world_rank == 0){
        //si estamos abajo, entonces no hay sur.
        if (n == 0){
          q[n][m] += Norte[n][m]*p[n+1][m];
        }
        //si estamos arriba, el norte se recibio desde p1 en la variable norte
        else if (n == localheight - 1){
          q[n][m] += Sur[n][m]*p[n-1][m] + Norte[n][m]*norte[m];
        }
        //en otro caso, el sur y el norte ya estaban en el procesador
        else{
          q[n][m] += Sur[n][m]*p[n-1][m] + Norte[n][m]*p[n+1][m];
        }
      }
      else if (world_rank == world_size -1){
        //en el ultimo procesador, no hay norte si estamos arriba:
        if (n == localheight - 1){
          q[n][m] += Sur[n][m]*p[n-1][m];
        }
        //si estamos abajo, el sur se recibio en la variable sur.
        else if (n == 0){
          q[n][m] += Sur[n][m]*sur[m] + Norte[n][m]*p[n+1][m];
        }
        // en otro caso, el norte y el sur ya estaban presentes
        else{
          q[n][m] += Sur[n][m]*p[n-1][m] + Norte[n][m]*p[n+1][m];
        }
      }
      else{
        //en otro caso, siempre hay sur y norte
        // si estamos arriba, se recibio el norte
        if (n == localheight - 1){
          q[n][m] += Sur[n][m]*p[n-1][m] + Norte[n][m]*norte[m];
        }
        //si estamos abajo se recibio el sur
        else if (n == 0){
          q[n][m] += Sur[n][m]*sur[m] + Norte[n][m]*p[n+1][m];
        }
        else{
          q[n][m] += Sur[n][m]*p[n-1][m] + Norte[n][m]*p[n+1][m];
        }
      }
    }
  }
  return q;
}

int main(){

  //inicializamos los parámetros
	int Nx = 39;
	int Ny = 74;
	float hx = 1/float (Nx-1);
	float hy = 1/float (Ny-1);
  //definimos las variables que usaremos
  float beta;
  float delta;
  float rho_final;
  float rho_inicial;
  float** p;
  float** x;
  float** r;
  float** q;
  float** Norte;
  float** Sur;
  float** Este;
  float** Oeste;
  float** Centro;

	MPI_Init(NULL,NULL);

	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // sacamos el rango de altura de la matriz que tendrá cada procesador.
	int firstIndex, localheight;
  localheight = Ny / world_size;
  firstIndex = world_rank * (Ny / world_size);
  if (world_rank == world_size-1){
      localheight += Ny % world_size;
  }
	cout << "Rank: " << world_rank << ", first index: " << firstIndex << ", local height: " << localheight << endl;

  // creamos el stencil local, son 5 pedazos de arrays Nx x altura.

  Norte = matrix_generator(Nx, localheight);
  for (int n=0; n<localheight; n++){
  for (int m=0; m<Nx; m++){
      Norte[n][m] = - alpha(((float) m)*hx, ((float) (n + firstIndex))*hy + 0.5)/(hy*hy);
    }
  }
  Sur = matrix_generator(Nx, localheight);
  for (int n=0; n<localheight; n++){
  for (int m=0; m<Nx; m++){
      Sur[n][m] = - alpha(((float) m)*hx, ((float) n + firstIndex)*hy - 0.5)/(hy*hy);
    }
  }
  Este = matrix_generator(Nx, localheight);
  for (int n=0; n<localheight; n++){
  for (int m=0; m<Nx; m++){
      Este[n][m] = - alpha(((float) m)*hx + 0.5, ((float) n + firstIndex)*hy)/(hx*hx);
    }
  }
  Oeste = matrix_generator(Nx, localheight);
  for (int n=0; n<localheight; n++){
  for (int m=0; m<Nx; m++){
      Oeste[n][m] = - alpha(((float) m)*hx - 0.5, ((float) n + firstIndex)*hy)/(hx*hx);
    }
  }
  Centro = matrix_generator(Nx, localheight);
  for (int n=0; n<localheight; n++){
  for (int m=0; m<Nx; m++){
      Centro[n][m] = -(Norte[n][m] + Sur[n][m] + Este[n][m] + Oeste[n][m]) + 1;
    }
  }


  // creamos el vector x local como array Nx x altura. Parte como puros ceros
  x = matrix_generator(Nx, localheight);

  // creamos el vector residuo local como array Nx x altura.
  r = matrix_generator(Nx, localheight);

  //el residuo parte como r0 = Ax0 - b = -b.
  //Pero b = f, por la ecuacion. Y podemos asumir que f es 1 sólo en el punto (0,0) del dominio, y fuera es 0.
  //luego, r0 = -b, con -b valiendo -1 en (0,0) y 0 en el resto de la malla:
  if (world_rank == 0){
    for (int n=0; n<localheight; n++){
    for (int m=0; m<Nx; m++){
      if (n == 0 & m == 0){
        r[n][m] = -1;
      }else {
        r[n][m] = 0;
      }
     }
    }
  }
  else {
    for (int n=0; n<localheight; n++){
    for (int m=0; m<Nx; m++){
      r[n][m] = 0;
    }
    }
  }


  //loop
  for (int i=1; i<50; i++){

    rho_final = parallel_dot_product(r, r, Nx, localheight);

    if (i == 1){
      p = matrix_generator(Nx, localheight);
      for (int n=0; n<localheight; n++){
        for (int m=0; m<Nx; m++){
          p[n][m] = r[n][m];
        }
      }
    }
    else{
      beta = rho_final/rho_inicial;
      p = sum_of_vectors(r, reescalate_vector(p, beta, Nx, localheight), Nx, localheight);
    }

    q = matvec(Norte,Sur,Este,Oeste,Centro,p,Nx,localheight,world_rank,world_size);
    delta = rho_final/parallel_dot_product(p, q, Nx, localheight);
    x = sum_of_vectors(x, reescalate_vector(p, -delta, Nx, localheight),Nx, localheight);
    r = sum_of_vectors(r, reescalate_vector(q, -delta, Nx, localheight),Nx, localheight);

    if (world_rank == 0){
      cout << "Iteracion: " << i << " , Norma: " << sqrt(rho_final) << endl;
    }

    rho_inicial = rho_final;
  }

	MPI_Finalize();

}
