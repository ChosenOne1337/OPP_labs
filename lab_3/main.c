#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <libgen.h>
#include <errno.h>
#include <limits.h>
#include <mpich/mpi.h>
#include "matrix.h"
#include "utils.h"
#include "errhandle.h"

void matrix_destroy_wrapper(void *mat) {
    matrix_destroy((Matrix*)mat);
}

void assert_with_resources(int condition, char *errorMsg) {
    if (!(condition)) {
        fprintf(stderr, "%s\n", errorMsg);
        free_resources();
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void print_usage(char *progName) {
    fprintf(stderr, "Usage: %s N1 N2 N3 P1 P2\n", basename(progName));
}

typedef enum Args {
    N1_arg = 1,
    N2_arg,
    N3_arg,
    P1_arg,
    P2_arg,
    TotalArgs
} Args;

void calculate(int N1, int N2, int N3, int P1, int P2);

int main(int argc, char *argv[]) {
    if (argc != TotalArgs) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }
    int returnCode = 0;
    long N1, N2, N3, P1, P2;
    returnCode |= parse_long(&N1, argv[N1_arg]);
    returnCode |= parse_long(&N2, argv[N2_arg]);
    returnCode |= parse_long(&N3, argv[N3_arg]);
    returnCode |= parse_long(&P1, argv[P1_arg]);
    returnCode |= parse_long(&P2, argv[P2_arg]);
    if (returnCode == FAILURE_CODE) {
        fprintf(stderr, "Failed to parse given arguments\n");
        return EXIT_FAILURE;
    }
    if (N1 <= 0 || N2 <= 0 || N3 <= 0 || P1 <= 0 || P2 <= 0) {
        fprintf(stderr, "Parameters can't be <= 0\n");
        return EXIT_FAILURE;
    }
    if (N1 > INT_MAX || N2 > INT_MAX || N3 > INT_MAX || P1 > INT_MAX || P2 > INT_MAX) {
        fprintf(stderr, "Parameters can't be > %d\n", INT_MAX);
    }
    if (N1 < P1 || N2 < P2) {
        fprintf(stderr, "Invalid sizes of matrices parts");
        return EXIT_FAILURE;
    }
    calculate((int)N1, (int)N2, (int)N3, (int)P1, (int)P2);

    return EXIT_SUCCESS;
}

#define NDIMS (2)
#define TRUE (1)
#define FALSE (0)
#define ROWS_DIM (0)
#define COLS_DIM (1)

void calculate(int N1, int N2, int N3, int P1, int P2) {
    MPI_Init(NULL, NULL);
    int rank = 0, procNum = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    if (procNum != P1 * P2) {
        fprintf(stderr, "P1 * P2 has to be equal to the number of processes\n");
        return;
    }
    // create 2D grid communicator
    MPI_Comm gridComm;
    int coords[NDIMS] = {0, 0};
    int dims[NDIMS] = {P1, P2}, periods[NDIMS] = {FALSE, FALSE};
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, TRUE, &gridComm);
    // get 2d coordinates of the process in the grid
    MPI_Cart_get(gridComm, NDIMS, dims, periods, coords);
    // get process rank in the grid
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // get neighbors ranks
    int displacement = 1;
    int prevX, prevY, nextX, nextY;
    MPI_Cart_shift(gridComm, ROWS_DIM, displacement, &prevY, &nextY);
    MPI_Cart_shift(gridComm, COLS_DIM, displacement, &prevX, &nextX);


    MPI_Comm_free(&gridComm);
    MPI_Finalize();
}
