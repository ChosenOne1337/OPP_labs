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

void mpi_comm_free_wrapper(void *comm) {
    MPI_Comm_free((MPI_Comm*)comm);
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

typedef enum Coord {
    CoordX = 0,
    CoordY = 1
} Coord;

void print_mat(Matrix *mat) {
    for (int r = 0; r < mat->rows; ++r) {
        for (int c = 0; c < mat->cols; ++c) {
            printf("%+.3f\t", elem_at(mat, r, c));
        }
        printf("\n");
    }
    printf("\n");
}

void calculate(int matArows, int matAcols, int matBcols, int gridRows, int gridCols) {
    MPI_Init(NULL, NULL);
    int procNum = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    if (procNum != gridRows * gridCols) {
        fprintf(stderr, "gridRows * gridCols has to be equal to the number of processes\n");
        return;
    }

    // create 2D grid communicator
    MPI_Comm gridComm;
    int dims[NDIMS] = {gridRows, gridCols}, periods[NDIMS] = {FALSE, FALSE}, reorder = TRUE;
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &gridComm);
    // get 2d coordinates of the process in the grid
    int coords[NDIMS] = {0, 0};
    MPI_Cart_get(gridComm, NDIMS, dims, periods, coords);
    // get process rank in the grid
    int gridRank = 0;
    int rootGridRank = 0;
    MPI_Comm_rank(gridComm, &gridRank);
    // split grid into rows and cols
    MPI_Comm rowComm, colComm;
    MPI_Comm_split(gridComm, coords[CoordX], coords[CoordY], &rowComm);
    MPI_Comm_split(gridComm, coords[CoordY], coords[CoordX], &colComm);
    int rootRowRank = 0, rootColRank = 0;
    // add communicators to resources
    add_resource(&gridComm, mpi_comm_free_wrapper);
    add_resource(&rowComm, mpi_comm_free_wrapper);
    add_resource(&colComm, mpi_comm_free_wrapper);

    // create matrices A, B, C in the main process
    Matrix *matA = NULL, *matB = NULL, *matC = NULL;
    if (gridRank == rootGridRank) {
        matA = matrix_create(matArows, matAcols);
        matB = matrix_create(matAcols, matBcols);
        matC = matrix_create(matArows, matBcols);
        add_resource(matA, matrix_destroy_wrapper);
        add_resource(matB, matrix_destroy_wrapper);
        add_resource(matC, matrix_destroy_wrapper);
        assert_with_resources(matA && matB && matC, "Failed to allocate the required amount of memory");

        matrix_randomize(matA, 1.0);
        matrix_randomize(matB, 1.0);
        print_mat(matA);
        print_mat(matB);
    }
    // create submatrices of A, B, C in all processes
    int submatAcols = matAcols;
    int submatArows = get_chunk_size(coords[CoordX], gridRows, matArows);
    int submatBcols = get_chunk_size(coords[CoordY], gridCols, matBcols);
    Matrix *submatA = matrix_create(submatArows, submatAcols);
    Matrix *submatB = matrix_create(submatAcols, submatBcols);
    Matrix *submatC = matrix_create(submatArows, submatBcols);
    add_resource(submatA, matrix_destroy_wrapper);
    add_resource(submatB, matrix_destroy_wrapper);
    add_resource(submatC, matrix_destroy_wrapper);
    assert_with_resources(submatA && submatB && submatC, "Failed to allocate the required amount of memory");

    // distribute matrices A, B between the processes
    int *matAchunkSizes = (int*)calloc((size_t)gridRows, sizeof(int));
    int *matAchunkOffsets = (int*)calloc((size_t)gridRows, sizeof(int));
    int *matBchunkSizes = (int*)calloc((size_t)gridCols, sizeof(int));
    int *matBchunkOffsets = (int*)calloc((size_t)gridCols, sizeof(int));
    int *matCchunkSizes = (int*)calloc((size_t)(gridRows * gridCols), sizeof(int));
    int *matCchunkOffsets = (int*)calloc((size_t)(gridRows * gridCols), sizeof(int));
    add_resource(matAchunkSizes, free);
    add_resource(matAchunkOffsets, free);
    add_resource(matBchunkSizes, free);
    add_resource(matBchunkOffsets, free);
    add_resource(matCchunkSizes, free);
    add_resource(matCchunkOffsets, free);
    assert_with_resources(matAchunkSizes && matAchunkOffsets && matBchunkSizes && matBchunkOffsets && matCchunkSizes && matCchunkOffsets,
                          "Failed to allocate the required amount of memory");
    for (int row = 0, offset = 0; row < gridRows; ++row) {
        matAchunkSizes[row] = get_chunk_size(row, gridRows, matArows) * matAcols;
        matAchunkOffsets[row] = offset;
        offset += matAchunkSizes[row];
    }
    for (int col = 0, offset = 0; col < gridCols; ++col) {
        matBchunkSizes[col] = get_chunk_size(col, gridCols, matBcols) * matAcols;
        matBchunkOffsets[col] = offset;
        offset += matBchunkSizes[col];
    }
    for (int row = 0; row < gridRows; ++row) {
        for (int col = 0; col < gridCols; ++col) {
            matCchunkSizes[row * gridCols + col] = get_chunk_size(col, gridCols, matBcols);
            matCchunkOffsets[row * gridCols + col] = get_chunk_offset(col, gridCols, matBcols)
                    + get_chunk_offset(row, gridRows, matArows) * matBcols;
        }
    }
    // scatter matrix A within the first column
    if (coords[CoordY] == 0) {
        MPI_Barrier(colComm);
        MPI_Scatterv(gridRank == rootGridRank ? matA->data : NULL, matAchunkSizes, matAchunkOffsets, MPI_DOUBLE,
                     submatA->data, submatA->rows * submatA->cols, MPI_DOUBLE, rootColRank, colComm);
    }
    // scatter matrix B within the first row
    if (coords[CoordX] == 0) {
        if (gridRank == rootGridRank) {
            matrix_transpose(matB);
        }
        swap_ints(&submatB->rows, &submatB->cols);
        MPI_Scatterv(gridRank == rootGridRank ? matB->data : NULL, matBchunkSizes, matBchunkOffsets, MPI_DOUBLE,
                     submatB->data, submatB->rows * submatB->cols, MPI_DOUBLE, rootRowRank, rowComm);
        // submatrices of B are transposed, transposition is required
        matrix_transpose(submatB);
    }
    // broadcast submatrix of A to other columns in the grid
    MPI_Bcast(submatA->data, submatA->rows * submatA->cols, MPI_DOUBLE, rootRowRank, rowComm);
    // broadcast submatrix of B to other rows in the grid
    MPI_Bcast(submatB->data, submatB->rows * submatB->cols, MPI_DOUBLE, rootColRank, colComm);
    // multiplicate submatrices
    matrix_mult(submatA, submatB, submatC);
    // assemble result in the main process
    int iterNum = get_chunk_size(0, gridRows, matArows);
    for (int iter = 0; iter < iterNum; ++iter) {
        MPI_Gatherv(submatC->data + iter * submatC->cols, submatC->cols, MPI_DOUBLE,
                    (gridRank == rootGridRank) ? matC->data + iter * matC->cols: NULL,
                    matCchunkSizes, matCchunkOffsets, MPI_DOUBLE, rootGridRank, gridComm);
    }

    if (gridRank == rootGridRank) {
        print_mat(matC);
    }

    free_resources();
    MPI_Finalize();
}
