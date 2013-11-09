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
#include <string.h>

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
MPI_Status status;

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
    A[i * n + i + (my_rank == MASTER?0:hM + h * (my_rank-1)) ] = 2.0 + randomDoubleInDomain(1.0);
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

  iter = 0;
  do {
    iter++;
    delta = 0.0;
    
    
    /* faire le calcul sur sa partie de la matrice */
    for (i=0;i< (my_rank==MASTER?hM:h); i++) {
      c = b[i];
      
      int i2 = 
	(my_rank==MASTER?0:hM) + //décalage de HMaitre 
	(my_rank==MASTER?0:(my_rank - 1) * h) + 
	i;
      
      for (j=0;j<n;j++) {
	// pas sûr de la valeur de i: à vérif
	if (i2 != j) {
	  c -= A[i * n + j] * x[j];
	}
      }

      c /= A[i * n + i2]; // colonne i2
      
      // pas sûr de la valeur de i: à vérif
      d = fabs(x[i2] - c);
      if (d > delta) delta = d;
      xp[i] = c;
    }

    convergence = (delta < eps);

    if (world_size == 1) continue;

    if (my_rank == MASTER) {
      MPI_Status status;
      for (i = 1; i < world_size; i++){
	MPI_Probe(MPI_ANY_SOURCE , TAG_SUBVECTOR, MPI_COMM_WORLD, &status);
	int id_source = status.MPI_SOURCE;
	/* vecteur + décalage du bout du maitre + (rank - 1 (puisqu'on
	   a inclut le maitre)) * taille bout esclave */
	MPI_Recv(x + hM + (id_source - 1) * h,
		 h, MPI_DOUBLE, 
		 id_source, TAG_SUBVECTOR, MPI_COMM_WORLD, &status);
      }
      
      // recopie du xp local au master
      memcpy(x, xp, sizeof(double)*hM);
    }
    else {
      MPI_Send(xp, h, MPI_DOUBLE, MASTER, TAG_SUBVECTOR, MPI_COMM_WORLD);
    }
    
    /* màj du vecteur x */
    MPI_Bcast(x, n, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    
    /* Applique un "et logique" aux convergences de tous les process et stocke
       le résultat dans convergence */
    double tmp = 0.0;
    MPI_Allreduce (&convergence, &tmp, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    convergence = tmp;

  } while ((!convergence) && (iter < maxIter));

  return x;
}

/**
   Jacobi iteration sans communications globales
 **/

double *jacobiIterationV2(double *x, double *xp, double *A, double *b, double eps, int n, int maxIter, int h, int hM) { 
  
  // x = vecteur plein, mis à jour à chaque itération
  // xp = sous-vecteur, que l'on doit envoyer au maitre
  
  int i, j, convergence, iter; 
  double c, d, delta;
  double *xNew, *xPrev, *xt;

  /* Init du vecteur x : commun à tous les processus */
  for (i=0;i<n;i++) {
    x[i] = 1.0;
  }

  iter = 0;
  do {
    iter++;
    delta = 0.0;
    
    
    /* faire le calcul sur sa partie de la matrice */
    for (i=0;i< (my_rank==MASTER?hM:h); i++) {
      c = b[i];
      
      int i2 = 
	(my_rank==MASTER?0:hM) + //décalage de HMaitre 
	(my_rank==MASTER?0:(my_rank - 1) * h) + 
	i;
      
      for (j=0;j<n;j++) {
	// pas sûr de la valeur de i: à vérif
	if (i2 != j) {
	  c -= A[i * n + j] * x[j];
	}
      }

      c /= A[i * n + i2]; // colonne i2
      
      // pas sûr de la valeur de i: à vérif
      d = fabs(x[i2] - c);
      if (d > delta) delta = d;
      xp[i] = c;
    }

    convergence = (delta < eps);

    // On copie notre partie du nouveau vecteur x dans le vecteur global
    memcpy(x + (my_rank == MASTER? 0 : hM + (my_rank - 1) * h), xp,
	   sizeof(double) * (my_rank == MASTER?hM:h));

    if (world_size == 1) continue;
    
    // Cheat mais pas grave
    double tmp = 0.0;
    MPI_Allreduce (&convergence, &tmp, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    convergence = tmp;
 
    // Premier tour de l'anneau
    if (my_rank == MASTER) {
      // On envoie notre vecteur màj.
      MPI_Send(x, hM, MPI_DOUBLE, my_rank+1, TAG_SUBVECTOR, MPI_COMM_WORLD);
      // On recoit le nouveau vecteur du dernier processus (world_size - 1) ayant fait le tour de l'anneau 
      MPI_Recv(x + hM, h * (world_size - 1), MPI_DOUBLE, (world_size - 1), TAG_SUBVECTOR, MPI_COMM_WORLD, &status);

      // Deuxième tour de l'anneau 
      // Si il y a plus de deux processus alors
      if (world_size > 2) {
	// On envoie la partie du vecteur que le voisin n'avait pas reçu. i.e : (my_rank + 2) à (world_size - 1)
	MPI_Send(x + hM + h, h * (world_size - 2) , MPI_DOUBLE, my_rank + 1, TAG_SUBVECTOR, MPI_COMM_WORLD);
      }
    }
    // Si on est le dernier processus de l'anneau : n - 1
    else if (my_rank == world_size - 1){
      // On recoit les vecteurs de 0 à n - 2 que l'on stocke dans x
      MPI_Recv(x, hM + (world_size - 2) * h, MPI_DOUBLE, my_rank - 1, TAG_SUBVECTOR, MPI_COMM_WORLD, &status);
      // On envoie au maitre le reste du vecteur de : 1 à n - 1 (il a déjà la première partie)
      MPI_Send(x + hM, (world_size - 1) * h, MPI_DOUBLE, MASTER, TAG_SUBVECTOR, MPI_COMM_WORLD);
      // Le dernier proc. a fini et peut recommencer ses calculs
    }
    /** Sinon on est un processus lambda (pas le premier, ni le dernier) qui doit recevoir et envoyer
	deux fois, sauf pour l'avant-dernier qui n'effectuera pas le dernier 
	envoi */
    else {
      // On commence par recevoir de son prédecesseur les données qui ont circulées
      MPI_Recv(x, hM + (my_rank - 1) * h, MPI_DOUBLE, my_rank - 1, TAG_SUBVECTOR, MPI_COMM_WORLD, &status);
      // On envoie à son successeur les données reçues précédemment en y ajoutant sa partie
      MPI_Send(x, hM + my_rank * h, MPI_DOUBLE, my_rank + 1, TAG_SUBVECTOR, MPI_COMM_WORLD);
      
      // Deuxième tour de l'anneau

      // On recoit le reste du vecteur dans le deuxième tour de l'anneau. i.e : la partie (my_rank + 1) à (world_size - 1)
      MPI_Recv(x + hM + my_rank * h, (world_size - (my_rank + 1)) * h, MPI_DOUBLE, my_rank - 1, TAG_SUBVECTOR, MPI_COMM_WORLD, &status);

      // Si nous ne sommes pas l'avant-dernier
      if (my_rank < world_size - 2){
	// Alors on envoie la partie qui manque au successeur. i.e : (my_rank + 2) à (world_size - 1)
	MPI_Send(x + hM + (my_rank + 1) * h, world_size - (my_rank + 2), MPI_DOUBLE, my_rank + 1, TAG_SUBVECTOR, MPI_COMM_WORLD);
      }
    }

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

  /* Perform Jacobi iteration 
     Time this (interesting) part of the code.
  */
  gettimeofday(&before, NULL);
  //x = jacobiIteration(x_vect, x_vect_modif, A, b, JACOBI_EPS, n, JACOBI_MAX_ITER, h, hM);
  x = jacobiIterationV2(x_vect, x_vect_modif, A, b, JACOBI_EPS, n, JACOBI_MAX_ITER, h, hM);
  gettimeofday(&after, NULL);
  
  /* Compute the residual */
  computeResidual(r, A, x, b, n, my_rank==MASTER?hM:h);

  /* Compute the maximum absolute value of the residual */
  maxAbsRes = maxAbsVector(r, my_rank==MASTER?hM:h);

  double tmp_maxAbsRes = 0.0;
  MPI_Reduce(&maxAbsRes, &tmp_maxAbsRes, 1, MPI_DOUBLE, 
	     MPI_MAX, MASTER, MPI_COMM_WORLD);
  
  if (my_rank == MASTER){
    maxAbsRes = tmp_maxAbsRes;
  }
  /* TERMINAISON des esclaves */
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
