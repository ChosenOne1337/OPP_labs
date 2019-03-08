#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <libgen.h>
#include <errno.h>
#include <limits.h>
#include "matrix.h"
#include "errhandle.h"

#define SUCCESS_CODE (0)
#define FAILURE_CODE (1)

#define EPS (1e-9)
#define TAU (1e-5)

void matrix_destroy_wrapper(void *mat) {
    matrix_destroy((Matrix*)mat);
}

void assert_with_resources(int condition, char *errorMsg) {
    if (!(condition)) {
        fprintf(stderr, "%s\n", errorMsg);
        free_resources();
        exit(EXIT_FAILURE);
    }
}

void init_matrix_A(Matrix *matA) {
    matrix_fill_with(matA, 1.0);
    #pragma omp parallel for
    for (int ix = 0; ix < matA->rows; ++ix) {
        elem_at(matA, ix, ix) = 2.0;
    }
}

void init_vector_B(Matrix *vecB, Matrix *vecU) {
    #pragma omp parallel for
    for (int i = 0; i < vecU->rows; ++i) {
        elem_at(vecU, i, 0) = 1.0;
        elem_at(vecB, i, 0) = vecB->rows + 1;
    }
}

int get_chunk_size(int chunkIndex, int chunksNumber, int sequenceSize) {
    // the remainder is uniformly distributed in chunks
    int chunkSize = sequenceSize / chunksNumber;
    int remainder = sequenceSize % chunksNumber;
    return chunkSize + (chunkIndex < remainder ? 1 : 0);
}

int get_chunk_offset(int chunkIndex, int chunksNumber, int sequenceSize) {
    int chunkSize = sequenceSize / chunksNumber;
    int remainder = sequenceSize % chunksNumber;
    return chunkIndex * chunkSize + (chunkIndex < remainder ? chunkIndex : remainder);
}

void work1(int N, int numThreads) {
    Matrix *matA = matrix_create(N, N);
    Matrix *vecB = matrix_create(N, 1);
    Matrix *vecX = matrix_create(N, 1);
    Matrix *vecY = matrix_create(N, 1);
    Matrix *vecU = matrix_create(N, 1);

    double eps = 0.0;
    double vecBnorm = 0.0;

    add_resource(matA, matrix_destroy_wrapper);
    add_resource(vecB, matrix_destroy_wrapper);
    add_resource(vecX, matrix_destroy_wrapper);
    add_resource(vecY, matrix_destroy_wrapper);
    add_resource(vecU, matrix_destroy_wrapper);

    assert_with_resources(matA && vecB && vecX && vecY && vecU,
                          "Failed to allocate necessary resources\n");
    init_matrix_A(matA);
    init_vector_B(vecB, vecU);

    double beg = omp_get_wtime();

    vecBnorm = get_norm_2(vecB);
    matrix_subtract(vecY, vecB);
    do {
        // x = x - tau * y
        matrix_mult_by(vecY, TAU);
        matrix_subtract(vecX, vecY);
        // eps = ||Ax - b|| / ||b||
        matrix_mult(matA, vecX, vecY);
        matrix_subtract(vecY, vecB);
        eps = get_norm_2(vecY) / vecBnorm;
    } while (eps >= EPS);

    double end = omp_get_wtime();

    matrix_subtract(vecX, vecU);
    double vecDist = get_norm_2(vecX);
    printf("||x - u|| == %.g\n", vecDist);
    printf("Elapsed time: %.3fs\n", end - beg);

    free_resources();
}

void work2(int N, int numThreads) {
    Matrix *matA = matrix_create(N, N);
    Matrix *vecB = matrix_create(N, 1);
    Matrix *vecX = matrix_create(N, 1);
    Matrix *vecY = matrix_create(N, 1);
    Matrix *vecU = matrix_create(N, 1);

    add_resource(matA, matrix_destroy_wrapper);
    add_resource(vecB, matrix_destroy_wrapper);
    add_resource(vecX, matrix_destroy_wrapper);
    add_resource(vecY, matrix_destroy_wrapper);
    add_resource(vecU, matrix_destroy_wrapper);

    assert_with_resources(matA && vecB && vecX && vecY && vecU,
                          "Failed to allocate necessary resources\n");
    init_matrix_A(matA);
    init_vector_B(vecB, vecU);

    double beg = omp_get_wtime();

    #pragma omp parallel
    {
        double eps = 0.0;
        double vecBnorm = parallel_get_norm_2(vecB);
        parallel_matrix_subtract(vecY, vecB);
        do {
            // x = x - tau * y
            parallel_matrix_mult_by(vecY, TAU);
            parallel_matrix_subtract(vecX, vecY);
            // eps = ||Ax - b|| / ||b||
            parallel_matrix_mult(matA, vecX, vecY);
            parallel_matrix_subtract(vecY, vecB);
            eps = parallel_get_norm_2(vecY) / vecBnorm;
        } while (eps >= EPS);
    }

    double end = omp_get_wtime();

    matrix_subtract(vecX, vecU);
    double vecDist = get_norm_2(vecX);
    printf("||x - u|| == %.g\n", vecDist);
    printf("Elapsed time: %.3fs\n", end - beg);

    free_resources();
}

int parse_int(const char *str, int *result, const char *failureMsg) {
    int base = 10;
    char *endptr = NULL;
    int errnoSaved = errno;
    errno = 0;

    long number = strtol(str, &endptr, base);
    if (endptr == str || *endptr != '\0') {
        fprintf(stderr, "%s\n", failureMsg != NULL ? failureMsg :
                    "parse_int(..): failed to parse an integer value\n");
        return FAILURE_CODE;
    }
    if (number < INT_MIN || number > INT_MAX ||
            (errno == ERANGE && (number == LONG_MAX || number == LONG_MIN))) {
        fprintf(stderr, "parse_int(..): value is out of range "
                        "(has to be from %d to %d)\n", INT_MIN, INT_MAX);
        return FAILURE_CODE;
    }
    if (number == 0 && errno != 0) {
        perror("parse_uint(..)");
        return FAILURE_CODE;
    }
    *result = (int)number;

    errno = errnoSaved;
    return SUCCESS_CODE;
}

#define REQUIRED_ARG_NUM (4)
#define VARIANTS_NUM (2)

void print_usage(char *progName) {
    fprintf(stderr, "Usage: %s <variant> <matrix size> "
                    "<number of threads>\n", basename(progName));
}

int main(int argc, char *argv[]) {
    if (argc != REQUIRED_ARG_NUM) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    int returnCode = 0;
    int matrixSize = 0, numThreads = 0, variant = 0;
    returnCode |= parse_int(argv[1], &variant,      "Failed to parse the program variant");
    returnCode |= parse_int(argv[2], &matrixSize,   "Failed to parse the matrix size");
    returnCode |= parse_int(argv[3], &numThreads,   "Failed to parse the number of threads");

    if (returnCode == FAILURE_CODE) {
        return EXIT_FAILURE;
    }
    if (variant <= 0 || variant > VARIANTS_NUM) {
        fprintf(stderr, "Program variant has to be from 1 to %d\n", VARIANTS_NUM);
        return EXIT_FAILURE;
    }
    if (matrixSize <= 0 || numThreads <= 0) {
        fprintf(stderr, "Matrix size and number of threads "
                        "have to be positive non-zero numbers\n");
        return EXIT_FAILURE;
    }
    if (numThreads > matrixSize) {
        fprintf(stderr, "Number of threads must be less than "
                        "or equal to the matrix size\n");
        return EXIT_FAILURE;
    }

    omp_set_num_threads(numThreads);
    if (variant == 1) {
        work1(matrixSize, numThreads);
    } else {
        work2(matrixSize, numThreads);
    }

    return EXIT_SUCCESS;
}
