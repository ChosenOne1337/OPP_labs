#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpich/mpi.h>
#include "matrix.h"

#define N (10)
#define EPS (1.e-5)

void init_matrix_A(Matrix *matA) {
    matrix_fill_with(matA, 1.0);
    for (ulong ix = 0; ix < matA->rows; ++ix) {
        elem_at(matA, ix, ix) = 2.0;
    }
}

void init_vector_B(Matrix *matA, Matrix *vecB, Matrix *vecU) {
    for (ulong i = 0; i < N; ++i) {
        elem_at(vecU, i, 1) = sin(M_PI * ((double) rand() / RAND_MAX - 0.5));
    }
    matrix_mult(matA, vecU, vecB);
}

void print_matrix(Matrix *mat) {
    for (ulong row = 0; row < mat->rows; ++row) {
        for (ulong col = 0; col < mat->cols; ++col) {
            printf("%+.3f ", elem_at(mat, row, col));
        }
        printf("\n");
    }
    printf("\n\n");
}

double count_tau(Matrix *matA, Matrix *vecY, Matrix *tempVec) {
    matrix_mult(matA, vecY, tempVec);

    double numerator, denominator;
    matrix_inner_product(vecY, tempVec, &numerator);
    matrix_inner_product(tempVec, tempVec, &denominator);

    return numerator / denominator;
}

double get_norm_2(Matrix *mat) {
    double normSquared;
    matrix_inner_product(mat, mat, &normSquared);
    return sqrt(normSquared);
}

double count_eps(Matrix *matA, Matrix *vecX, Matrix *vecB, Matrix *tempVec) {
    matrix_mult(matA, vecX, tempVec);
    matrix_subtract(tempVec, vecB);
    return get_norm_2(tempVec) / get_norm_2(vecB);
}

void calc_variant_zero(void);
void calc_variant_first(void);
void calc_variant_second(void);

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

int main(int argc, char *argv[]) {
    int procRank = 0, procNum = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    if (procNum > N) {
        MPI_Finalize();
        fprintf(stderr, "Number of processes can't be greater than size of the matrix A\n");
        exit(-1);
    }

    int localVecChunkSize = get_chunk_size(procRank, procNum, N);
    int localMatChunkSize = N * localVecChunkSize;
    int *vecChunkSizes = (int*)calloc(procNum, sizeof(int));
    int *vecChunkOffsets = (int*)calloc(procNum, sizeof(int));
    int *matChunkSizes = (int*)calloc(procNum, sizeof(int));
    int *matChunkOffsets = (int*)calloc(procNum, sizeof(int));

    // fill chunk sizes & offsets
    for (int procIndex = 0; procIndex < procNum; ++procIndex) {
        vecChunkSizes[procIndex] = get_chunk_size(procIndex, procNum, N);
        vecChunkOffsets[procIndex] = get_chunk_offset(procIndex, procNum, N);
        matChunkSizes[procIndex] = N * vecChunkSizes[procIndex];
        matChunkOffsets[procIndex] = N * vecChunkOffsets[procIndex];
    }

    Matrix *matAchunk = matrix_create(localVecChunkSize, N);
    Matrix *vecBchunk = matrix_create(localVecChunkSize, 1);
    Matrix *vecXchunk = matrix_create(localVecChunkSize, 1);
    Matrix *vecYchunk = matrix_create(localVecChunkSize, 1);
    Matrix *vecTchunk = matrix_create(localVecChunkSize, 1);

    Matrix *matA = NULL;
    Matrix *vecB = NULL;
    Matrix *vecX = matrix_create(N, 1);
    Matrix *vecY = matrix_create(N, 1);

    double vecBnorm = 0.0;

    /* initialization work */

    if (procRank == 0) {
        // create matrix A, vector B
        Matrix *vecU = matrix_create(N, 1);
        matA = matrix_create(N, N);
        vecB = matrix_create(N, 1);
        // initialize vector B, U
        srand(time(NULL));
        init_matrix_A(matA);
        init_vector_B(matA, vecB, vecU);
        // calculate ||b||
        vecBnorm = get_norm_2(vecB);
        // do cleanup
        print_matrix(vecU);
        matrix_destroy(vecU);
    }
    // split matrix A into pieces and send to other processes
    MPI_Scatterv(matA->data, matChunkSizes, matChunkOffsets, MPI_DOUBLE,
                 matAchunk->data, localMatChunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // send parts of the vector B to other processes
    MPI_Scatterv(vecB->data, vecChunkSizes, vecChunkOffsets, MPI_DOUBLE,
                 vecBchunk->data, localVecChunkSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* calculations */

    double eps = 0.0, tau = 0.0;
    double localTauParts[2], tauParts[2];
    do {
                /* y_n = A * x_n - b */

        // calculate parts of y' = A * x_n
        matrix_mult(matAchunk, vecX, vecYchunk);
        // calculate parts of y = y' - b
        matrix_subtract(vecYchunk, vecBchunk);
        // assemble the vector Y in all processes
        MPI_Allgatherv(vecYchunk->data, localVecChunkSize, MPI_DOUBLE,
                       vecY->data, vecChunkSizes, vecChunkOffsets, MPI_DOUBLE, MPI_COMM_WORLD);

                /* tau_n = (y_n, A * y_n) / (A * y_n, A * y_n) */

        // calculate parts of t = A * y_n
        matrix_mult(matAchunk, vecY, vecTchunk);
        // get numerator and denominator
        matrix_inner_product(vecYchunk, vecTchunk, &localTauParts[0]);
        matrix_inner_product(vecTchunk, vecTchunk, &localTauParts[1]);
        MPI_Reduce(&localTauParts, &tauParts, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
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
        matrix_mult(matAchunk, vecX, vecTchunk);
        // subtract parts of b from parts of t
        matrix_subtract(vecTchunk, vecBchunk);

    } while (eps >= EPS);

    /* cleanup */

    matrix_destroy(matAchunk);
    matrix_destroy(vecBchunk);
    matrix_destroy(vecXchunk);
    matrix_destroy(vecYchunk);
    matrix_destroy(vecTchunk);

    matrix_destroy(vecX);
    matrix_destroy(vecY);
    matrix_destroy(matA);
    matrix_destroy(vecB);

    free(vecChunkSizes);
    free(vecChunkOffsets);
    free(matChunkSizes);
    free(matChunkOffsets);

    MPI_Finalize();
}



//void calc_variant_zero(void) {
//    srand(time(NULL));
//    Matrix *matA = matrix_create(N, N);
//    Matrix *vecB = matrix_create(N, 1);
//    Matrix *vecX = matrix_create(N, 1);
//    Matrix *vecY = matrix_create(N, 1);
//    Matrix *tempVec = matrix_create(N, 1);
//    double tau = 0.0;
//    double eps = 0.0;

//    // init A
//    init_matrix_A(matA);
//    // init B
//    init_vector_B(matA, vecB);

//    do {
//        // y = A * x - b
//        matrix_mult(matA, vecX, vecY);
//        matrix_subtract(vecY, vecB);
//        // tau = (y, A * y) / (A * y, A * y)
//        tau = count_tau(matA, vecY, tempVec);
//        // x = x - tau * y
//        matrix_mult_by(vecY, tau);
//        matrix_subtract(vecX, vecY);
//        // eps = ||Ax - b|| / ||b||
//        eps = count_eps(matA, vecX, vecB, tempVec);
//    } while (eps >= EPS);

//    printf("X_n, eps == %f:\n", eps);
//    print_matrix(vecX);

//    matrix_destroy(matA);
//    matrix_destroy(vecB);
//    matrix_destroy(vecX);
//    matrix_destroy(vecY);
//    matrix_destroy(tempVec);
//}

void calc_variant_first(void) {

}

void calc_variant_second(void) {

}
