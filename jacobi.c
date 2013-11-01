/* Implementation of a Jacobi iteration 

   Sequential version

   Copyright (C) 2013 Université Pierre et Marie Curie Paris 6.
   
   Contributor: Christoph Lauter

   All rights reserved.

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
   -modif : ajouté h aux paramètres-
*/
void generateRandomDiagonallyDominantMatrix(double *A, int n, int h) {
  int i, j;
  double absMax;

  absMax = 0.5 / ((double) n);

  for (i=0;i<h;i++) {
    for (j=0;j<n;j++) {
      A[i * n + j] = randomDoubleInDomain(absMax);
    }
    A[i * n + i] = 2.0 + randomDoubleInDomain(1.0);
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
*/
void computeResidual(double *r, double *A, double *x, double *b, int n) {
  int i, j;
  double c;

  for (i=0;i<n;i++) {
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

*/
double *jacobiIteration(double *x, double *xp, double *A, double *b, double eps, int n, int maxIter) { 
  int i, j, convergence, iter; 
  double c, d, delta;
  double *xNew, *xPrev, *xt;

  for (i=0;i<n;i++) {
    x[i] = 1.0;
  }

  xPrev = x;
  xNew = xp;
  iter = 0;
  do {
    iter++;
    delta = 0.0;
    for (i=0;i<n;i++) {
      c = b[i];
      for (j=0;j<n;j++) {
	if (i != j) {
	  c -= A[i * n + j] * xPrev[j];
	}
      }
      c /= A[i * n + i];
      d = fabs(xPrev[i] - c);
      if (d > delta) delta = d;
      xNew[i] = c;
    }
    xt = xPrev;
    xPrev = xNew; 
    xNew = xt;
    convergence = (delta < eps);
  } while ((!convergence) && (iter < maxIter));

  return xPrev;
}

/* A small testing main program */
int main(int argc, char *argv[]) {
  int i, n;
  double *A, *b, *x, *r; //, *xA, *xB;
  double *x_vect, *x_vect_modif;
  double maxAbsRes;
  struct timeval before, after;

  int my_rank;
  int world_size;

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
  
  /* hM : taille de la matrice effective */
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
  
  /* sous-vecteur résultat pour chaque processus 

     Un vecteur de taille n qui sera mis à jour à chaque itération
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
  generateRandomDiagonallyDominantMatrix(A, n, h);
  generateRandomVector(b, h);
  

  /* FIN MODIF */


  /* Perform Jacobi iteration 
     Time this (interesting) part of the code.
  */
  gettimeofday(&before, NULL);
  x = jacobiIteration(x_vect, x_vect_modif, A, b, JACOBI_EPS, n, JACOBI_MAX_ITER);
  gettimeofday(&after, NULL);

  /* Compute the residual */
  computeResidual(r, A, x, b, n);

  /* Compute the maximum absolute value of the residual */
  maxAbsRes = maxAbsVector(r, n);
  
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

  /* Free the memory */
  free(A);
  free(b);
  free(x_vect);
  free(x_vect_modif);
  /*  free(xA);
      free(xB); */
  free(r);

  /* Return success */
  return 0;
}
