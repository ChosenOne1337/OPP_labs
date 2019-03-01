#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"
#include "errhandle.h"

#define EPS (1.e-13)

void init_matrix_A(Matrix *matA);
void init_vector_B(Matrix *matA, Matrix *vecB, Matrix *vecU);
void matrix_destroy_wrapper(void *mat);

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
