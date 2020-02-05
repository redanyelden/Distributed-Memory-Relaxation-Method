#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

//returns a square matrix of size d, populated by random numbers between 0 and 1
double **createMatrix(int d)
{
	//allocate memory for matrix
	double *values = malloc((unsigned)d * (unsigned)d * sizeof(double));
	double **matrix = malloc((unsigned)d * sizeof(double *));
	int i;
	for (i = 0; i < d; i++)
	{
		matrix[i] = values + i * d;
	}

	//populate with random numbers
	srand(0);
	int j;
	for (i = 0; i < d; i++)
	{
		for (j = 0; j < d; j++)
		{
			matrix[i][j] = ((rand() / (double)RAND_MAX));
		}
	}
	return matrix;
}

void freeMatrices(double **matrix, double **resultsMatrix)
{
	//free necessary memory
	free(*matrix);
	free(matrix);

	free(*resultsMatrix);
	free(resultsMatrix);
}

//carry out relaxation technique on given rows
int avgRows(double **matrix, double **resultsMatrix, int d, double precision, int start, int end)
{
	int done = 0;

	int count = 0;
	int i, j;

	for (i = start; i < end; i++)
	{
		for (j = 1; j < d - 1; j++)
		{
			//averaging of four neighbours
			resultsMatrix[i][j] = (matrix[i][j - 1] + matrix[i][j + 1] + matrix[i - 1][j] + matrix[i + 1][j]) / 4;
			if (fabs(resultsMatrix[i][j] - matrix[i][j]) < precision)
			{
				count++;
			}
		}
	}

	//checking if number of changes that are within the precision is equal to all cells processed
	if (count == ((d - 2) * ((end - start))))
	{
		done = 1;
	}

	return done;
}

//check if all processors are done
int allDone(int *a, int n)
{
	int i;
	for (i = 0; i < n; i++)
	{
		if (a[i] == 0)
		{
			return 0;
		}
	}
	return 1;
}

//sequential implementation
void sequential(double **matrix, double **copyMatrix, int d, double precision)
{
	double **tmp;
	bool done = false;

	//while the difference between the old and new values for all elements of the matrix are more than the precision
	while (done == false)
	{
		int count = 0;

		//calculate average, and check whether the difference is less than the average. If it is, record it as an increment on count.
		int i, j;
		for (i = 1; i < d - 1; i++)
		{
			for (j = 1; j < d - 1; j++)
			{
				copyMatrix[i][j] = (matrix[i][j - 1] + matrix[i][j + 1] + matrix[i - 1][j] + matrix[i + 1][j]) / 4;
				if (fabs(copyMatrix[i][j] - matrix[i][j]) < precision)
				{
					count++;
				}
			}
		}

		//swap pointers to arrays
		tmp = matrix;
		matrix = copyMatrix;
		copyMatrix = tmp;

		//if the number of elements that have a difference less than the precision is equal to the total number of elements being processed, we're done
		if (count == ((d * d) - (4 * d) + 4))
		{
			done = true;
		}
	}
}

int main(int argc, char **argv)
{
	int dimension = 2000;
	double precision = 0.01;

	int rc = MPI_Init(&argc, &argv);

	if (rc != MPI_SUCCESS)
	{
		printf("Error starting MPI program\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}

	int processorID, numProcessors;

	//storing each processors ID and the total number of processors
	MPI_Comm_rank(MPI_COMM_WORLD, &processorID);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcessors);

	//each processor gets creates an array
	double **matrix = createMatrix(dimension);
	double **resultsMatrix = createMatrix(dimension);

	//MPI_Bcast(&matrix[0][0], dimension * dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	//working out section to work on
	int remainder = (dimension - 2) % numProcessors;
	int nRowsEach = (dimension - remainder - 2) / numProcessors;

	int start = processorID * nRowsEach;
	if (processorID < remainder)
	{
		start = start + processorID;
		nRowsEach = nRowsEach + 1;
	}
	else
	{
		start = start + remainder;
	}
	start = start + 1;
	int end = start + nRowsEach;

	//printf("Processor %d: start = %d, end = %d\n", processorID, start, end);

	int everyoneDone = 0;
	int relaxed = 0;
	double *send = malloc((unsigned)dimension * sizeof(double));
	double *receiveLower = malloc((unsigned)dimension * sizeof(double));
	double *receiveUpper = malloc((unsigned)dimension * sizeof(double));
	double **tmp = matrix;
	int ifDone[numProcessors];
	int i, j;

	//while not all changes are within the precision
	while (!everyoneDone)
	{
		//average given rows
		relaxed = avgRows(matrix, resultsMatrix, dimension, precision, start, end);

		//swap pointers
		tmp = matrix;
		matrix = resultsMatrix;
		resultsMatrix = tmp;

		//sending whether processor is finished then checking if all processors are done
		if (processorID == 0)
		{
			ifDone[0] = relaxed;
			for (i = 1; i < numProcessors; i++)
			{
				MPI_Recv(&ifDone[i], 1, MPI_INT, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			everyoneDone = allDone(ifDone, numProcessors);
		}
		else
		{
			MPI_Send(&relaxed, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
		//broadcasting to all processes whether to continue or not
		MPI_Bcast(&everyoneDone, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		//if the processor needs to continue
		if (!everyoneDone)
		{
			//swapping of upper and lower lines
			//everyone sends upper line to the right (+ 1), lower line to the left (- 1)
			if (processorID == 0)
			{
				send = matrix[end - 1];
				MPI_Send(send, dimension, MPI_DOUBLE, processorID + 1, 0, MPI_COMM_WORLD);
			}
			else if (processorID == numProcessors - 1)
			{
				send = matrix[start];
				MPI_Send(send, dimension, MPI_DOUBLE, processorID - 1, 0, MPI_COMM_WORLD);
			}
			else
			{
				send = matrix[end - 1];
				MPI_Send(send, dimension, MPI_DOUBLE, processorID + 1, 0, MPI_COMM_WORLD);

				send = matrix[start];
				MPI_Send(send, dimension, MPI_DOUBLE, processorID - 1, 0, MPI_COMM_WORLD);
			}

			if (processorID == 0)
			{
				MPI_Recv(receiveUpper, dimension, MPI_DOUBLE, processorID + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for (i = 0; i < dimension; i++)
				{
					matrix[end][i] = receiveUpper[i];
				}
			}
			else if (processorID == numProcessors - 1)
			{
				MPI_Recv(receiveLower, dimension, MPI_DOUBLE, processorID - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for (i = 0; i < dimension; i++)
				{
					matrix[start - 1][i] = receiveLower[i];
				}
			}
			else
			{
				MPI_Recv(receiveUpper, dimension, MPI_DOUBLE, processorID + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for (i = 0; i < dimension; i++)
				{
					matrix[end][i] = receiveUpper[i];
				}

				MPI_Recv(receiveLower, dimension, MPI_DOUBLE, processorID - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for (i = 0; i < dimension; i++)
				{
					matrix[start - 1][i] = receiveLower[i];
				}
			}
		}
	}

	//swap pointers
	tmp = matrix;
	matrix = resultsMatrix;
	resultsMatrix = tmp;

	//free(send);
	//free(receiveLower);
	//free(receiveUpper);

	//join for final matrix

	//each processor sends their matrix to the root processor
	if (processorID != 0)
	{
		MPI_Send(&matrix[0][0], dimension * dimension, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}
	//the root processor inputs the relevant rows into a final matrix
	else
	{
		double **finalMatrix = createMatrix(dimension);
		int k;
		for (i = 0; i < dimension; i++)
		{
			finalMatrix[0][i] = matrix[0][i];
		}
		for (i = 0; i < dimension; i++)
		{
			finalMatrix[dimension - 1][i] = matrix[dimension - 1][i];
		}
		for (i = 1; i < numProcessors; i++)
		{
			MPI_Recv(&matrix[0][0], dimension * dimension, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			//calculating which section was worked on
			remainder = (dimension - 2) % numProcessors;
			nRowsEach = (dimension - remainder - 2) / numProcessors;

			start = i * nRowsEach;
			if (i < remainder)
			{
				start = start + i;
				nRowsEach = nRowsEach + 1;
			}
			else
			{
				start = start + remainder;
			}
			start = start + 1;
			end = start + nRowsEach;

			for (j = start; j < end; j++)
			{
				for (k = 0; k < dimension; k++)
				{
					finalMatrix[j][k] = matrix[j][k];
				}
			}
		}
	}

	MPI_Finalize();

	return 0;
}