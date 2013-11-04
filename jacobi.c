/* Implementation of a Jacobi iteration 

   Sequential version

   Copyright (C) 2013 Université Pierre et Marie Curie Paris 6.
   
   Contributor: Christoph Lauter

   all rights reserved.

*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

/* Some constants */
#define DEFAULT_PROBLEM_SIZE  1024
#define JACOBI_EPS            1e-13
#define JACOBI_MAX_ITER       100

#define JACOBI_MAX_RESIDUAL   1e-7
#define DISPLAY_FIRST_ENTRIES 10

#define MASTER 0

/* utility macros */

#define MAX(a,b) (((a)>(b))?(a):(b))

/* Tags */

#define TAG_ABS_RES 1<<0
#define TAG_SUBVECTOR 1<<1
#define TAG_FULLVECTOR 1<<2

// Global vars
int my_rank;
int world_size;

/* Compute time differences in seconds */
double computeTimeDifferenceInSeconds(struct timeval *before, struct timeval *after) {
  double secs, usecs;

  secs = ((double) after->tv_sec) - ((double) before->tv_sec);
  usecs = ((double) after->tv_usec) - ((double) before->tv_usec);
  
  return secs + (usecs / 1000000.0);
}

/* Generate a random double precision number between -absMax and
   absMax 
*/
double randomDoubleInDomain(double absMax) {
  double res;

  res = absMax * ((double) random()) / ((double) RAND_MAX);

  if (random() & 1) res = -res;

  return res;
}

/* Generate a matrix of size n * n that is certainly diagonally
   dominant 
   -modif- : 
   - ajouté h et hM aux paramètres
   - i = 0 à h - 1
   - A[i * n + i] => A[i * n + i + hM + h * (my_rank - 1)]
*/
void generateRandomDiagonallyDominantMatrix(double *A, int n, int h, int hM) {
  int i, j;
  double absMax;

  absMax = 0.5 / ((double) n);

  for (i=0;i<h;i++) {
    for (j=0;j<n;j++) {
      A[i * n + j] = randomDoubleInDomain(absMax);
    }
    if (my_rank == MASTER)
      A[i * n + i] = 2.0 + randomDoubleInDomain(1.0);
    else
      // pas sûr de l'indice, à revoir
      A[i * n + i + hM + h * (my_rank-1) ] = 2.0 + randomDoubleInDomain(1.0);
  }
}

/* Generate a random vector of size n
   -modif : ajouté h aux paramètres-
*/
void generateRandomVector(double *v, int n) {
  int i;

  for (i=0;i<n;i++) {
    v[i] = randomDoubleInDomain(1024.0);
  }
}

/* Compute the residual b - A * x for a n * n matrix A, a n sized
   right-hand side b and a n sized solution x, using no extra
   precision. 

   -modif- add h param
*/
void computeResidual(double *r, double *A, double *x, double *b, int n, int h) {
  int i, j;
  double c;
  
  for (i=0;i<h;i++) {
    c = b[i];
    for (j=0;j<n;j++) {
      c -= A[i * n + j] * x[j];
    }
    r[i] = c;
  }
}

/* Get the maximum magnitude of the entries in an n sized vector */
double maxAbsVector(double *v, int n) {
  int i;
  double c, res;

  res = 0.0;
  for (i=0;i<n;i++) {
    c = fabs(v[i]);
    if (c > res) res = c;
  }

  return res;
}

/* Perform Jacobi iteration on a n * n sized, diagonally dominant
   matrix A and a n sized right-hand size b.

   Stop the iteration when the difference between two consecutive solutions 
   becomes less than eps in magnitude in each line. 

   Stop in any case after maxIter iterations.

   Put the n sized solution in the vector x, which is supposed to be
   allocated.

   Do not change the system matrix A nor the right-hand side b.

   Uses a scratch vector xp of the same size as x.

   Returns one of the pointers x or xp to indicate in which one of 
   both the final result has been stored.

   h = hauteur des vecteurs et matrice traités
   hM = hauteur du maitre
*/
double *jacobiIteration(double *x, double *xp, double *A, double *b, double eps, int n, int maxIter, int h, int hM) { 
  
  // x = vecteur plein, mis à jour à chaque itération
  // xp = sous-vecteur, que l'on doit envoyer au maitre
  
  int i, j, convergence, iter; 
  double c, d, delta;
  double *xNew, *xPrev, *xt;

  /* Init du vecteur x : commun à tous les processus */
  for (i=0;i<n;i++) {
    x[i] = 1.0;
  }

  do {
    iter++;
    delta = 0.0;
    
    /* faire le calcul sur sa partie de la matrice */
    for (i=0;i< h; i++) {
      c = b[i];
      for (j=0;j<n;j++) {
	// pas sûr de la valeur de i: à vérif
	if (i + (my_rank==MASTER?0:hM) + my_rank * h != j) {
	  c -= A[i * n + j] * x[j];
	}
      }
      c /= A[i * n + i];
      // pas sûr de la valeur de i: à vérif
      d = fabs(x[i + (my_rank==MASTER?0:hM) + my_rank * h] - c);
      if (d > delta) delta = d;
      xp[i] = c;
    }
    /*  
	xt = xPrev;
	xPrev = xNew; 
	xNew = xt;
    */
    convergence = (delta < eps);

    if (my_rank == MASTER) {
      MPI_Status status;
      int tmp_conv;
      for (i = 1; i < world_size; i++){
	MPI_Probe(MPI_ANY_SOURCE , TAG_SUBVECTOR, MPI_COMM_WORLD, &status);
	int id_source = status.MPI_SOURCE;
	/* matrice + décalage du bout du maitre + rank * taille bout esclave */
	MPI_Recv(x + hM + (id_source - 1) * h,
		 n * h, MPI_DOUBLE, 
		 id_source, TAG_SUBVECTOR, MPI_COMM_WORLD, &status);
      }
    }
    else {
      MPI_Send(xp, h, MPI_DOUBLE, MASTER, TAG_SUBVECTOR, MPI_COMM_WORLD);
    }
    
    /* Met à jour le vecteur x */
    MPI_Bcast(x, n, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    
    /* Applique un "et logique" aux convergences de tous les process et stocke
       le résultat dans convergence */
    MPI_Allreduce (&convergence, &convergence, 1, MPI_INT,
                   MPI_LAND, MPI_COMM_WORLD);

  } while ((!convergence) && (iter < maxIter));

  return x;
}

/* A small testing main program */
int main(int argc, char *argv[]) {
  int i, n;
  double *A, *b, *x, *r; //, *xA, *xB;
  double *x_vect, *x_vect_modif;
  double maxAbsRes;
  struct timeval before, after;

  MPI_Status status;
  
  /* Get the argument that indicates the problem size */
  if(argc > 1) {
    n = atoi(argv[1]);
  } else {
    n = DEFAULT_PROBLEM_SIZE;
    fprintf(stderr, "Using default problem size n = %d\n",n);
  }
  
  /* Initialize the random seed */
  srandom((unsigned int) time(NULL));
  
  // INIT
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  
  // h = partie entière (n / world_size)

  /* h : hauteur de la matrice  */
  int xN = n / world_size, h;
  double xR = (double)n / (double)world_size;
  if (xR > xN)
    h = xN+1;
  else
    h = xN;
  
  /* hM : hauteur pour le maitre */
  int hM = n - (world_size-1) * h;

  
  /* Allocation des blocs de la matrice pour chaque processus (taille : h * n)  
     -modifié-
   */
  if ((A = (double *) calloc(h * n, sizeof(double))) == NULL) {
    fprintf(stderr, "Not enough memory.\n");
    MPI_Finalize();
    return 1;
  }

  
  /* sous-vecteur résultat pour chaque processus 
     -modifié-
   */
  if ((b = (double *) calloc(h, sizeof(double))) == NULL) {
    free(A);
    fprintf(stderr, "Not enough memory.\n");
    MPI_Finalize();
    return 1;
  }
  
  /* Un vecteur de taille n qui sera mis à jour à chaque itération
     Un vecteur de taille h qui servira à stocker les résultats intermédiaires
     -modifié-
  */
  if ((x_vect = (double *) calloc(n, sizeof(double))) == NULL) {
    free(A);
    free(b);
    fprintf(stderr, "Not enough memory.\n");
    MPI_Finalize();
    return 1;
  }
  

  if ((x_vect_modif = (double *) calloc(h, sizeof(double))) == NULL) {
    free(A);
    free(b);
    free(x_vect);
    fprintf(stderr, "Not enough memory.\n");
    MPI_Finalize();
    return 1;
  }
  
  /* sous-vecteur résidu 
     -modif-
  */
  if ((r = (double *) calloc(h, sizeof(double))) == NULL) {
    free(A);
    free(b);
    free(x_vect);
    free(x_vect_modif);
    fprintf(stderr, "Not enough memory.\n");
    MPI_Finalize();
    return 1;
  }

  /* Generate a random diagonally dominant matrix A and a random
     right-hand side b 
  */
  generateRandomDiagonallyDominantMatrix(A, n, my_rank==MASTER?hM:h, hM);
  generateRandomVector(b, my_rank==MASTER?hM:h);

  /* FIN MODIF */

  /* Perform Jacobi iteration 
     Time this (interesting) part of the code.
  */
  gettimeofday(&before, NULL);
  x = jacobiIteration(x_vect, x_vect_modif, A, b, JACOBI_EPS, n, JACOBI_MAX_ITER, h, hM);
  gettimeofday(&after, NULL);

  /* Compute the residual */
  computeResidual(r, A, x, b, n, my_rank==MASTER?hM:h);

  /* Compute the maximum absolute value of the residual */
  maxAbsRes = maxAbsVector(r, my_rank==MASTER?hM:h);

  double tmp_maxAbsRes;
  MPI_Reduce(&maxAbsRes, &tmp_maxAbsRes, 1, MPI_DOUBLE, 
	     MPI_MAX, MASTER, MPI_COMM_WORLD);

  if (my_rank == MASTER){
    maxAbsRes = MAX(maxAbsRes, tmp_maxAbsRes);
  }
  /* Envoi par les esclaves des max résidus et TERMINAISON des
     esclaves */
  else {
    goto free_zone;
  }
  
  /* Display maximum absolute value of residual and a couple of
     entries of the solution vector and corresponding residual
  */
  printf("Maximum absolute value of residual: %1.8e\n", maxAbsRes);
  printf("\n");
  for (i=0;i<DISPLAY_FIRST_ENTRIES;i++) {
    printf("%1.8e\t%1.8e\n", x[i], r[i]);
  }
  printf("\n");

  /* Decide if residual is small enough */
  if (maxAbsRes <= JACOBI_MAX_RESIDUAL) {
    printf("Solution OKAY\n");
  } else {
    printf("Solution NOT okay\n");
  }

  /* Display time for the Jacobi iteration */
  printf("Computing the solution with Jacobi iteration took %12.6fs\n", computeTimeDifferenceInSeconds(&before, &after));

  // Terminaisons des processus
 free_zone:
  printf("Fin du processus : %d\n", my_rank);
  
  /* Free the memory */
  free(A);
  free(b);
  free(x_vect);
  free(x_vect_modif);
  /*  free(xA);
      free(xB); */
  free(r);
  MPI_Finalize();
  /* Return success */
  return 0;
}
