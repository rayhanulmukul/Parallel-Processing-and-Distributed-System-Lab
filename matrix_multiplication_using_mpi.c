/* How to run: 
compile: $mpicc matrix_multiplication.c -o matrix
run    : $mpirun -np 2 matrix
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to print a matrix (for debugging or displaying result)
void display(int rows, int cols, int matrix[rows][cols]) {
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank (ID) of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    // Matrix dimensions and total number of matrix pairs
    int K = 100, M = 50, N = 50, P = 50;
    // if(rank == 0) {
    //     printf("Enter Number of Matrices: ");
    //     scanf("%d", &K);
    //     printf("Enter Number of Rows in Matrix A: ");
    //     scanf("%d", &M);
    //     printf("Enter Number of Columns in Matrix A: ");
    //     scanf("%d", &N);
    //     printf("Enter Number of Columns in Matrix B: ");
    //     scanf("%d", &P);
    // }

    // Broadcast the matrix configuration to all processes
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Ensure work can be evenly divided among processes
    if(K % size != 0) {
        if(rank == 0)
            printf("Number of matrices must be divisible by the number of processes.\n");
        MPI_Finalize();
        return 1;
    }

    // Declare global matrices (only used in root process)
    int A[K][M][N], B[K][N][P], R[K][M][P];

    // Initialize matrices A and B with random values in the root process
    if(rank == 0) {
        for(int k = 0; k < K; k++) {
            for(int i = 0; i < M; i++) {
                for(int j = 0; j < N; j++) {
                    A[k][i][j] = rand() % 100;
                }
            }
            for(int i = 0; i < N; i++) {
                for(int j = 0; j < P; j++) {
                    B[k][i][j] = rand() % 100;
                }
            }
        }
    }

    // Local buffers to store the portion of data each process will handle
    int localA[K / size][M][N], localB[K / size][N][P], localR[K / size][M][P];

    // Distribute matrix A and B among all processes
    MPI_Scatter(A, (K / size) * M * N, MPI_INT, localA, (K / size) * M * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, (K / size) * N * P, MPI_INT, localB, (K / size) * N * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Start timer for performance measurement
    double startTime = MPI_Wtime();

    // Perform matrix multiplication for assigned blocks
    for(int k = 0; k < (K / size); k++) {
        for(int i = 0; i < M; i++) {
            for(int j = 0; j < P; j++) {
                localR[k][i][j] = 0;
                for(int l = 0; l < N; l++) {
                    // Multiply A and B element-wise and add to sum
                    localR[k][i][j] += (localA[k][i][l] * localB[k][l][j]) % 100;
                }
                localR[k][i][j] %= 100;
            }
        }
    }

    // Stop timer after computation
    double endTime = MPI_Wtime();

    // Gather computed results from all processes into final result matrix R in root
    MPI_Gather(localR, (K / size) * M * P, MPI_INT, R, (K / size) * M * P, MPI_INT, 0, MPI_COMM_WORLD);

    // Optional: Print all result matrices from root process
    /*
    if(rank == 0) {
        for(int k = 0; k < K; k++) {
            printf("Result Matrix R%d\n", k);
            display(M, P, R[k]);
        }
    }
    */

    // Ensure all processes finish before printing time
    MPI_Barrier(MPI_COMM_WORLD);

    // Print time taken by each process
    printf("Process %d: Time taken = %f seconds\n", rank, endTime - startTime);

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
