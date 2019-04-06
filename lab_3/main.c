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

#define NDIMS (2)
#define TRUE (1)
#define FALSE (0)
#define ROWS_DIM (0)
#define COLS_DIM (1)

typedef enum Dimension {
    RowDim = 0,
    ColDim = 1
} Coord;

typedef enum Args {
    N1_arg = 1,
    N2_arg,
    N3_arg,
    P1_arg,
    P2_arg,
    TotalArgs
} Args;

typedef struct TaskParams {
    int matArows;
    int matAcols;
    int matBrows;
    int matBcols;
    int gridRows;
    int gridCols;
} TaskParams;

static int gridRank;
static int rootRank = 0;
static int coords[NDIMS];
static MPI_Datatype submatType;
static MPI_Datatype stripType;
static MPI_Comm gridComm, rowComm, colComm;

int create_matrices(Matrix **matA, Matrix **matB, Matrix **matC,
                        Matrix **submatA, Matrix **submatB, Matrix **submatC,
                            TaskParams *params) {
    if (gridRank == rootRank) {
        *matA = matrix_create(params->matArows, params->matAcols);
        *matB = matrix_create(params->matBrows, params->matBcols);
        *matC = matrix_create(params->matArows, params->matBcols);
        if (!*matA || !*matB || !*matC) {
            matrix_destroy(*matA);
            matrix_destroy(*matB);
            matrix_destroy(*matC);
            return FAILURE_CODE;
        }
    }

    int submatAcols = params->matAcols;
    int submatArows = get_chunk_size(coords[RowDim], params->gridRows, params->matArows);
    int submatBrows = params->matBrows;
    int submatBcols = get_chunk_size(coords[ColDim], params->gridCols, params->matBcols);
    *submatA = matrix_create(submatArows, submatAcols);
    *submatB = matrix_create(submatBrows, submatBcols);
    *submatC = matrix_create(submatArows, submatBcols);

    if (*submatA && *submatB && *submatC) {
        return SUCCESS_CODE;
    }
    matrix_destroy(*submatA);
    matrix_destroy(*submatB);
    matrix_destroy(*submatC);

    return FAILURE_CODE;
}

void free_matrices(Matrix *matA, Matrix *matB, Matrix *matC,
                    Matrix *submatA, Matrix *submatB, Matrix *submatC) {
    matrix_destroy(matA);
    matrix_destroy(matB);
    matrix_destroy(matC);
    matrix_destroy(submatA);
    matrix_destroy(submatB);
    matrix_destroy(submatC);
}

void initialize_matrices(Matrix *matA, Matrix *matB) {
    if (gridRank == rootRank) {
        double amp = 1.0;
        matrix_randomize(matA, amp);
        matrix_randomize(matB, amp);
    }
}

void create_grid_communicators(TaskParams *params) {
    int dims[NDIMS] = {params->gridRows, params->gridCols};
    int periods[NDIMS] = {0, 0};
    int reorder = TRUE;
    // create process grid
    MPI_Cart_create(MPI_COMM_WORLD, NDIMS, dims, periods, reorder, &gridComm);
    // get rank of the process in the grid communicator
    MPI_Comm_rank(gridComm, &gridRank);
    // get process coordinates
    MPI_Cart_coords(gridComm, gridRank, NDIMS, coords);
    // get rows and cols communicators in the grid
    int remainDims[NDIMS];
    remainDims[RowDim] = FALSE; remainDims[ColDim] = TRUE;
    MPI_Cart_sub(gridComm, remainDims, &rowComm);
    remainDims[RowDim] = TRUE; remainDims[ColDim] = FALSE;
    MPI_Cart_sub(gridComm, remainDims, &colComm);
}

void free_grid_communicators(void) {
    MPI_Comm_free(&gridComm);
    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
}

void create_mpi_datatypes(TaskParams *params) {
    MPI_Datatype newType;
    MPI_Aint lb;
    MPI_Aint doubleExtent;
    MPI_Type_extent(MPI_DOUBLE, &doubleExtent);

    // vertical strips of matrix B
    int numberOfBlocks = params->matBrows;
    int blockLength = params->matBcols / params->gridCols;
    int stride = params->matBcols;
    MPI_Type_vector(numberOfBlocks, blockLength, stride, MPI_DOUBLE, &newType);
    MPI_Type_lb(newType, &lb);
    MPI_Type_create_resized(newType, lb, doubleExtent * blockLength, &stripType);
    MPI_Type_commit(&stripType);

    // submatrices of matrix C
    numberOfBlocks = params->matArows / params->gridRows;
    blockLength = params->matBcols / params->gridCols;
    stride = params->matBcols;
    MPI_Type_vector(numberOfBlocks, blockLength, stride, MPI_DOUBLE, &newType);
    MPI_Type_lb(newType, &lb);
    MPI_Type_create_resized(newType, lb, doubleExtent * blockLength, &submatType);
    MPI_Type_commit(&submatType);
}

void free_mpi_datatypes(void) {
    MPI_Type_free(&stripType);
    MPI_Type_free(&submatType);
}

void distribute_matrices(Matrix *matA, Matrix *matB, Matrix *submatA, Matrix *submatB, TaskParams *params) {
    // scatter matrix A
    if (coords[ColDim] == 0) {
        MPI_Scatter(gridRank == rootRank ? matA->data : NULL, submatA->rows * submatA->cols, MPI_DOUBLE,
                    submatA->data, submatA->rows * submatA->cols, MPI_DOUBLE, rootRank, colComm);
    }

    // scatter matrix B
    if (coords[RowDim] == 0) {
        int sendCount = 1;
        MPI_Scatter(gridRank == rootRank ? matB->data : NULL, sendCount, stripType,
                    submatB->data, submatB->rows * submatB->cols, MPI_DOUBLE, rootRank, rowComm);
    }

    // broadcast matrices A and B
    MPI_Bcast(submatA->data, submatA->rows * submatA->cols, MPI_DOUBLE, rootRank, rowComm);
    MPI_Bcast(submatB->data, submatB->rows * submatB->cols, MPI_DOUBLE, rootRank, colComm);
}

void assemble_result(Matrix *matC, Matrix *submatC, TaskParams *params) {
    int procNum = params->gridRows * params->gridCols;
    int *recvCounts = NULL, *recvDispls = NULL;

    if (gridRank == rootRank) {
        recvCounts = (int*)calloc((size_t)procNum, sizeof(int));
        recvDispls = (int*)calloc((size_t)procNum, sizeof(int));
        for (int row = 0; row < params->gridRows; ++row) {
            for (int col = 0; col < params->gridCols; ++col) {
                recvCounts[row * params->gridCols + col] = 1;
                recvDispls[row * params->gridCols + col] = row * submatC->rows * params->gridCols + col;
            }
        }
    }

    MPI_Gatherv(submatC->data, submatC->rows * submatC->cols, MPI_DOUBLE,
                gridRank == rootRank ? matC->data : NULL, recvCounts, recvDispls, submatType, rootRank, gridComm);

    free(recvCounts);
    free(recvDispls);
}

int check_result(Matrix *matA, Matrix *matB, Matrix *matC) {
    Matrix *matT = matrix_create(matC->rows, matC->cols);
    if (matT == NULL) {
        fprintf(stderr, "check_result() error: failed to allocate memory");
        return FALSE;
    }
    matrix_mult(matA, matB, matT);
    for (int r = 0; r < matC->rows; ++r) {
        for (int c = 0; c < matC->cols; ++c) {
            if (elem_at(matC, r, c) != elem_at(matT, r, c)) {
                return FALSE;
            }
        }
    }
    matrix_destroy(matT);
    return TRUE;
}

void print_mat(Matrix *mat) {
    for (int r = 0; r < mat->rows; ++r) {
        for (int c = 0; c < mat->cols; ++c) {
            printf("%+.3f\t", elem_at(mat, r, c));
        }
        printf("\n");
    }
    printf("\n");
}

void calculate(TaskParams *params) {
    Matrix *matA = NULL, *matB = NULL, *matC = NULL;
    Matrix *submatA = NULL, *submatB = NULL, *submatC = NULL;

    create_grid_communicators(params);
    create_mpi_datatypes(params);
    int returnCode = create_matrices(&matA, &matB, &matC, &submatA, &submatB, &submatC, params);
    if (returnCode == FAILURE_CODE) {
        fprintf(stderr, "Process (%d, %d): insufficient memory\n", coords[RowDim], coords[ColDim]);
        MPI_Abort(MPI_COMM_WORLD, FAILURE_CODE);
    }
    initialize_matrices(matA, matB);

    double beg = MPI_Wtime();
    distribute_matrices(matA, matB, submatA, submatB, params);
    matrix_mult(submatA, submatB, submatC);
    assemble_result(matC, submatC, params);
    double end = MPI_Wtime();

    if (gridRank == rootRank) {
        int resultIsValid = check_result(matA, matB, matC);
        printf("Result is %s!\n"
               "Elapsed time: %g\n",
               resultIsValid ? "valid" : "invalid", end - beg);
    }

    free_mpi_datatypes();
    free_grid_communicators();
    free_matrices(matA, matB, matC, submatA, submatB, submatC);
}

void print_usage(char *progName) {
    fprintf(stderr, "Usage: %s N1 N2 N3 P1 P2\n", basename(progName));
}

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
        return EXIT_FAILURE;
    }
    if (N1 < P1 || N3 < P2) {
        fprintf(stderr, "Invalid sizes of matrices parts\n");
        return EXIT_FAILURE;
    }

    int procNum = 0, procRank = 0;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    if (procRank == 0) {
        if (procNum != P1 * P2) {
            fprintf(stderr, "Number of nodes in the grid has to be equal to the number of MPI processes\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (N1 % P1 != 0) {
            fprintf(stderr, "Number of rows in the matrix A has to be divisible by the number of rows in the grid\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (N3 % P2 != 0) {
            fprintf(stderr, "Number of cols in the matrix B has to be divisible by the number of cols in the grid\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    TaskParams params = {.matArows = (int)N1, .matAcols = (int)N2,
                         .matBrows = (int)N2, .matBcols = (int)N3,
                         .gridRows = (int)P1, .gridCols = (int)P2};
    calculate(&params);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
