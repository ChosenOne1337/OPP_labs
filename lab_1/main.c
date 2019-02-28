#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpich/mpi.h>
#include "matrix.h"
#include "errhandle.h"

#define EPS (1.e-9)

void assert_with_resources(int condition, char *errorMsg) {
    if (!(condition)) {
        fprintf(stderr, "%s\n", errorMsg);
        free_resources();
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
}

void matrix_destroy_wrapper(void *mat) {
    Matrix *pmat = (Matrix*)mat;
    matrix_destroy((Matrix*)mat);
}

void init_matrix_A(Matrix *matA) {
    matrix_fill_with(matA, 1.0);
    for (int ix = 0; ix < matA->rows; ++ix) {
        elem_at(matA, ix, ix) = 2.0;
    }
}

void init_vector_B(Matrix *matA, Matrix *vecB, Matrix *vecU) {
    for (int i = 0; i < vecU->rows; ++i) {
        elem_at(vecU, i, 1) = sin(M_PI * ((double) rand() / RAND_MAX - 0.5));
    }
    matrix_mult(matA, vecU, vecB);
}

void print_matrix(Matrix *mat) {
    for (int row = 0; row < mat->rows; ++row) {
        for (int col = 0; col < mat->cols; ++col) {
            printf("%+.5f ", elem_at(mat, row, col));
        }
        printf("\n");
    }
    printf("\n\n");
}

double get_norm_2(Matrix *mat) {
    double normSquared;
    matrix_inner_product(mat, mat, &normSquared);
    return sqrt(normSquared);
}

double count_tau(Matrix *matA, Matrix *vecY, Matrix *tempVec) {
    matrix_mult(matA, vecY, tempVec);

    double numerator, denominator;
    matrix_inner_product(vecY, tempVec, &numerator);
    matrix_inner_product(tempVec, tempVec, &denominator);

    return numerator / denominator;
}

double count_eps(Matrix *matA, Matrix *vecX, Matrix *vecB, Matrix *tempVec) {
    matrix_mult(matA, vecX, tempVec);
    matrix_subtract(tempVec, vecB);
    return get_norm_2(tempVec) / get_norm_2(vecB);
}

int get_chunk_size(int chunkIndex, int chunksNumber, int sequenceSize) {
    // the remainder is uniformly distributed in chunks
    int quotient = sequenceSize / chunksNumber;
    int remainder = sequenceSize % chunksNumber;
    return quotient + (chunkIndex < remainder ? 1 : 0);
}

int get_chunk_offset(int chunkIndex, int chunksNumber, int sequenceSize) {
    int quotient = sequenceSize / chunksNumber;
    int remainder = sequenceSize % chunksNumber;
    return chunkIndex * quotient + (chunkIndex < remainder ? chunkIndex : remainder);
}

void ring_multiplication(Matrix *matChunk, Matrix *vecChunk, Matrix *vecResChunk, Matrix *vecTempChunk,
                         int *vecChunkOffsets, int *vecChunkSizes, int procRank, int procNum) {
    int biggestChunkSize = vecChunkSizes[0];
    // set vecResChunk to zero
    memset(vecResChunk->data, 0, vecResChunk->rows * sizeof(double));
    // calculate parts of A * vec
    for (int i = 0, chunkIx = procRank; i < procNum; ++i) {
        MPI_Sendrecv_replace(vecChunk->data, biggestChunkSize, MPI_DOUBLE, (procRank + 1) % procNum, 1,
                             (procRank + procNum - 1) % procNum, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        chunkIx = (chunkIx + procNum - 1) % procNum;
        vecChunk->rows = vecChunkSizes[chunkIx];
        matrix_partial_mult(matChunk, vecChunk, vecTempChunk,
                            matChunk->rows, vecChunk->rows, 0, vecChunkOffsets[chunkIx]);
        matrix_add(vecResChunk, vecTempChunk);
    }
}

void calc_variant_zero(int argc, char *argv[]) {
    int N = (argc == 1) ? 10000 : atoi(argv[1]);

    Matrix *matA = matrix_create(N, N);
    Matrix *vecB = matrix_create(N, 1);
    Matrix *vecX = matrix_create(N, 1);
    Matrix *vecY = matrix_create(N, 1);
    Matrix *vecU = matrix_create(N, 1);
    Matrix *tempVec = matrix_create(N, 1);

    double tau = 0.0;
    double eps = 0.0;

    add_resource(matA, matrix_destroy_wrapper);
    add_resource(vecB, matrix_destroy_wrapper);
    add_resource(vecX, matrix_destroy_wrapper);
    add_resource(vecY, matrix_destroy_wrapper);
    add_resource(vecU, matrix_destroy_wrapper);
    add_resource(tempVec, matrix_destroy_wrapper);

    if (!(matA && vecB && vecX && vecY && vecU && tempVec)) {
        fprintf(stderr, "Failed to allocate necessary resources\n");
        free_resources();
        return;
    }

    // init A
    srand(time(NULL));
    init_matrix_A(matA);
    // init B
    init_vector_B(matA, vecB, vecU);

    time_t beg = clock();

    do {
        // y = A * x - b
        matrix_mult(matA, vecX, vecY);
        matrix_subtract(vecY, vecB);
        // tau = (y, A * y) / (A * y, A * y)
        tau = count_tau(matA, vecY, tempVec);
        // x = x - tau * y
        matrix_mult_by(vecY, tau);
        matrix_subtract(vecX, vecY);
        // eps = ||Ax - b|| / ||b||
        eps = count_eps(matA, vecX, vecB, tempVec);
    } while (eps >= EPS);

    time_t end = clock();

    matrix_subtract(vecX, vecU);
    double vecDist = get_norm_2(vecX);
    printf("||x - u|| == %.g\n", vecDist);
    printf("Elapsed time: %.3fs\n", (double)(end - beg) / CLOCKS_PER_SEC);

    free_resources();
}

void calc_variant_first(int argc, char *argv[]) {
    int N = (argc == 1) ? 10000 : atoi(argv[1]);

    int procRank = 0, procNum = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    assert_with_resources(procNum <= N, "Number of processes can't be greater than size of the matrix A\n");

    int localVecChunkSize = get_chunk_size(procRank, procNum, N);
    int localMatChunkSize = N * localVecChunkSize;
    int *vecChunkSizes = (int*)calloc(procNum, sizeof(int));
    int *vecChunkOffsets = (int*)calloc(procNum, sizeof(int));
    int *matChunkSizes = (int*)calloc(procNum, sizeof(int));
    int *matChunkOffsets = (int*)calloc(procNum, sizeof(int));

    Matrix *matAchunk = matrix_create(localVecChunkSize, N);
    Matrix *vecBchunk = matrix_create(localVecChunkSize, 1);
    Matrix *vecXchunk = matrix_create(localVecChunkSize, 1);
    Matrix *vecYchunk = matrix_create(localVecChunkSize, 1);
    Matrix *vecTchunk = matrix_create(localVecChunkSize, 1);

    Matrix *matA = NULL;
    Matrix *vecB = NULL;
    Matrix *vecU = NULL;
    Matrix *vecX = matrix_create(N, 1);
    Matrix *vecY = matrix_create(N, 1);

    double vecBnorm = 0.0;
    double eps = 0.0, localEpsNumerator = 0.0, epsNumerator = 0.0;
    double tau = 0.0, localTauParts[2], tauParts[2];

    // add resources for deallocation and check for equality to NULL

    add_resource(vecChunkSizes, free);
    add_resource(vecChunkOffsets, free);
    add_resource(matChunkSizes, free);
    add_resource(matChunkOffsets, free);
    add_resource(matAchunk, matrix_destroy_wrapper);
    add_resource(vecBchunk, matrix_destroy_wrapper);
    add_resource(vecXchunk, matrix_destroy_wrapper);
    add_resource(vecYchunk, matrix_destroy_wrapper);
    add_resource(vecTchunk, matrix_destroy_wrapper);
    add_resource(matA, matrix_destroy_wrapper);
    add_resource(vecB, matrix_destroy_wrapper);
    add_resource(vecU, matrix_destroy_wrapper);
    add_resource(vecX, matrix_destroy_wrapper);
    add_resource(vecY, matrix_destroy_wrapper);

    assert_with_resources(vecChunkSizes && vecChunkOffsets && matChunkSizes && matChunkOffsets &&
                        matAchunk && vecBchunk && vecXchunk && vecYchunk && vecTchunk && vecX && vecY,
                        "Failed to allocate necessary resources");

    /* initialization work */

    // fill chunk sizes & offsets
    for (int procIndex = 0; procIndex < procNum; ++procIndex) {
        vecChunkSizes[procIndex] = get_chunk_size(procIndex, procNum, N);
        vecChunkOffsets[procIndex] = get_chunk_offset(procIndex, procNum, N);
        matChunkSizes[procIndex] = N * vecChunkSizes[procIndex];
        matChunkOffsets[procIndex] = N * vecChunkOffsets[procIndex];
    }

    double beg = MPI_Wtime();

    if (procRank == 0) {
        // create matrix A, vector B
        matA = matrix_create(N, N);
        vecB = matrix_create(N, 1);
        vecU = matrix_create(N, 1);
        assert_with_resources(matA && vecB && vecU, "Failed to allocate necessary resources");
        // initialize vector B, U
        srand(time(NULL));
        init_matrix_A(matA);
        init_vector_B(matA, vecB, vecU);
        // calculate ||b||
        vecBnorm = get_norm_2(vecB);
    }

    // send ||b|| to all processes
    MPI_Bcast(&vecBnorm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // split matrix A into pieces and send to other processes
    MPI_Scatterv(procRank == 0 ? matA->data : NULL, matChunkSizes, matChunkOffsets, MPI_DOUBLE,
                 matAchunk->data, localMatChunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // send parts of the vector B to other processes
    MPI_Scatterv(procRank == 0 ? vecB->data : NULL, vecChunkSizes, vecChunkOffsets, MPI_DOUBLE,
                 vecBchunk->data, localVecChunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* calculations */

    // calculate parts of y_n' = A * x_n
    matrix_mult(matAchunk, vecX, vecYchunk);
    // calculate parts of y_n = y_n' - b
    matrix_subtract(vecYchunk, vecBchunk);

    do {
                /* y_n = A * x_n - b */
        /*
         * assemble the vector Y in all processes using chunks calculated in the end
         * of the previous iteration or before the do-while loop)
         */
        MPI_Allgatherv(vecYchunk->data, localVecChunkSize, MPI_DOUBLE,
                       vecY->data, vecChunkSizes, vecChunkOffsets, MPI_DOUBLE, MPI_COMM_WORLD);

                /* tau_n = (y_n, A * y_n) / (A * y_n, A * y_n) */

        // calculate parts of t = A * y_n
        matrix_mult(matAchunk, vecY, vecTchunk);
        // get numerator and denominator
        matrix_inner_product(vecYchunk, vecTchunk, &localTauParts[0]);
        matrix_inner_product(vecTchunk, vecTchunk, &localTauParts[1]);
        MPI_Allreduce(&localTauParts, &tauParts, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // calculate tau_n
        tau = tauParts[0] / tauParts[1];

                /* x_n+1 = x_n - tau_n * y_n */

        // calculate parts of tau_n * y_n
        matrix_mult_by(vecYchunk, tau);
        // calculate parts of x_n+1
        matrix_subtract(vecXchunk, vecYchunk);
        // assemble the vector X in all processes
        MPI_Allgatherv(vecXchunk->data, localVecChunkSize, MPI_DOUBLE,
                       vecX->data, vecChunkSizes, vecChunkOffsets, MPI_DOUBLE, MPI_COMM_WORLD);

               /* eps = ||A * x_n - b|| / ||b|| */

        // calculate parts of t = A * x_n
        matrix_mult(matAchunk, vecX, vecYchunk);
        // subtract parts of b from parts of t
        matrix_subtract(vecYchunk, vecBchunk);
        // calculate parts of ||A * x_n - b|| ^ 2
        matrix_inner_product(vecYchunk, vecYchunk, &localEpsNumerator);
        // assemble ||A * x_n - b|| ^2
        MPI_Allreduce(&localEpsNumerator, &epsNumerator, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        // calculate eps
        eps = sqrt(epsNumerator) / vecBnorm;
    } while (eps >= EPS);

    double end = MPI_Wtime();

    if (procRank == 0) {
        // compare
        matrix_subtract(vecX, vecU);
        double vecDist = get_norm_2(vecX);
        printf("||x - u|| == %.g\n", vecDist);
        printf("Elapsed time: %.3fs\n", end - beg);
    }

    /* cleanup */

    free_resources();
    MPI_Finalize();
}

void calc_variant_second(int argc, char *argv[]) {
    int N = (argc == 1) ? 10000 : atoi(argv[1]);

    int procRank = 0, procNum = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    assert_with_resources(procNum <= N, "Number of processes can't be greater than size of the matrix A\n");

    int biggestChunkSize = get_chunk_size(0, procNum, N);
    int localVecChunkSize = get_chunk_size(procRank, procNum, N);
    int localMatChunkSize = N * localVecChunkSize;
    int *vecChunkSizes = (int*)calloc(procNum, sizeof(int));
    int *vecChunkOffsets = (int*)calloc(procNum, sizeof(int));
    int *matChunkSizes = (int*)calloc(procNum, sizeof(int));
    int *matChunkOffsets = (int*)calloc(procNum, sizeof(int));

    Matrix *matAchunk = matrix_create(localVecChunkSize, N);
    Matrix *vecBchunk = matrix_create(localVecChunkSize, 1);
    Matrix *vecXchunk = matrix_create(biggestChunkSize, 1);
    Matrix *vecYchunk = matrix_create(biggestChunkSize, 1);
    Matrix *vecTchunk = matrix_create(localVecChunkSize, 1);
    Matrix *vecVchunk = matrix_create(localVecChunkSize, 1);

    Matrix *matA = NULL;
    Matrix *vecB = NULL;
    Matrix *vecU = NULL;
    Matrix *vecX = NULL;

    double vecBnorm = 0.0;
    double eps = 0.0, localEpsNumerator = 0.0, epsNumerator = 0.0;
    double tau = 0.0, localTauParts[2], tauParts[2];

    // add resources for deallocation and check for equality to NULL

    add_resource(vecChunkSizes, free);
    add_resource(vecChunkOffsets, free);
    add_resource(matChunkSizes, free);
    add_resource(matChunkOffsets, free);
    add_resource(matAchunk, matrix_destroy_wrapper);
    add_resource(vecBchunk, matrix_destroy_wrapper);
    add_resource(vecXchunk, matrix_destroy_wrapper);
    add_resource(vecYchunk, matrix_destroy_wrapper);
    add_resource(vecTchunk, matrix_destroy_wrapper);
    add_resource(vecVchunk, matrix_destroy_wrapper);
    add_resource(matA, matrix_destroy_wrapper);
    add_resource(vecB, matrix_destroy_wrapper);
    add_resource(vecU, matrix_destroy_wrapper);
    add_resource(vecX, matrix_destroy_wrapper);

    assert_with_resources(vecChunkSizes && vecChunkOffsets && matChunkSizes && matChunkOffsets &&
                        matAchunk && vecBchunk && vecXchunk && vecYchunk && vecTchunk,
                        "Failed to allocate necessary resources");

    /* initialization work */

    // fill chunk sizes & offsets
    for (int procIndex = 0; procIndex < procNum; ++procIndex) {
        vecChunkSizes[procIndex] = get_chunk_size(procIndex, procNum, N);
        vecChunkOffsets[procIndex] = get_chunk_offset(procIndex, procNum, N);
        matChunkSizes[procIndex] = N * vecChunkSizes[procIndex];
        matChunkOffsets[procIndex] = N * vecChunkOffsets[procIndex];
    }

    double beg = MPI_Wtime();

    if (procRank == 0) {
        // create matrix A, vector B
        matA = matrix_create(N, N);
        vecB = matrix_create(N, 1);
        vecU = matrix_create(N, 1);
        // create vector X for result validation
        vecX = matrix_create(N, 1);
        assert_with_resources(matA && vecB && vecU, "Failed to allocate necessary resources");
        // initialize vector B, U
        srand(time(NULL));
        init_matrix_A(matA);
        init_vector_B(matA, vecB, vecU);
        // calculate ||b||
        vecBnorm = get_norm_2(vecB);
    }

    // send ||b|| to all processes
    MPI_Bcast(&vecBnorm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // split matrix A into pieces and send to other processes
    MPI_Scatterv(procRank == 0 ? matA->data : NULL, matChunkSizes, matChunkOffsets, MPI_DOUBLE,
                 matAchunk->data, localMatChunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // send parts of the vector B to other processes
    MPI_Scatterv(procRank == 0 ? vecB->data : NULL, vecChunkSizes, vecChunkOffsets, MPI_DOUBLE,
                 vecBchunk->data, localVecChunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* calculations */

    // in the beginning, x_n == 0 => y_n = -b
    vecYchunk->rows = vecChunkSizes[procRank];
    matrix_subtract(vecYchunk, vecBchunk);
    do {
                /* tau_n = (y_n, A * y_n) / (A * y_n, A * y_n) */

        // get parts of v = A * y_n
        ring_multiplication(matAchunk, vecYchunk, vecVchunk, vecTchunk,
                            vecChunkOffsets, vecChunkSizes, procRank, procNum);
        // get numerator and denominator
        matrix_inner_product(vecYchunk, vecTchunk, &localTauParts[0]);
        matrix_inner_product(vecTchunk, vecTchunk, &localTauParts[1]);
        MPI_Allreduce(&localTauParts, &tauParts, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        // calculate tau_n
        tau = tauParts[0] / tauParts[1];

                /* x_n+1 = x_n - tau_n * y_n */

        matrix_mult_by(vecYchunk, tau);
        matrix_subtract(vecXchunk, vecYchunk);

               /*
                * eps = ||A * x_n - b|| / ||b||;
                * y_n = A * x_n - b
                */

        // get parts of y' = A * x_n
        ring_multiplication(matAchunk, vecXchunk, vecYchunk, vecTchunk,
                            vecChunkOffsets, vecChunkSizes, procRank, procNum);
        // get y = y' - b <=> A * x_n - b
        matrix_subtract(vecYchunk, vecBchunk);

        // calculate parts of ||A * x_n - b|| ^ 2
        matrix_inner_product(vecYchunk, vecYchunk, &localEpsNumerator);
        // assemble ||A * x_n - b|| ^2
        MPI_Allreduce(&localEpsNumerator, &epsNumerator, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        // calculate eps
        eps = sqrt(epsNumerator) / vecBnorm;

    } while (eps >= EPS);

    double end = MPI_Wtime();

    if (procRank == 0) {
        MPI_Allgatherv(vecXchunk->data, localVecChunkSize, MPI_DOUBLE,
                       vecX->data, vecChunkSizes, vecChunkOffsets, MPI_DOUBLE, MPI_COMM_WORLD);
        // compare
        matrix_subtract(vecX, vecU);
        double vecDist = get_norm_2(vecX);
        printf("||x - u|| == %.g\n", vecDist);
        printf("Elapsed time: %.3fs\n", end - beg);
    }

    /* cleanup */


    free_resources();
    MPI_Finalize();
}

int main(int argc, char *argv[]) {
//    calc_variant_first(argc, argv);
    return EXIT_SUCCESS;
}
