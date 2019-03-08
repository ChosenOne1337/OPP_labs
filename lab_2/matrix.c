#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "matrix.h"

Matrix* matrix_create(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return NULL;
    }

    Matrix *mat = (Matrix*)malloc(sizeof(Matrix));
    if (mat == NULL) {
        return NULL;
    }

    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double*)calloc(rows * cols, sizeof(double));
    if (mat->data == NULL) {
        free(mat);
        return NULL;
    }

    return mat;
}

void matrix_destroy(Matrix *mat) {
    if (mat != NULL) {
        free(mat->data);
        free(mat);
    }
}

int matrix_mult(Matrix *leftMat, Matrix *rightMat, Matrix *resMat) {
    if (leftMat->cols !=  rightMat->rows ||
        resMat->rows  !=  leftMat->rows  ||
        resMat->cols  !=  rightMat->cols) {
        return -1;
    }

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < resMat->rows; ++row) {
        for (int col = 0; col < resMat->cols; ++col) {
            elem_at(resMat, row, col) = 0.0;
            for (int j = 0; j < leftMat->cols; ++j) {
                elem_at(resMat, row, col) += elem_at(leftMat, row, j) * elem_at(rightMat, j, col);
            }
        }
    }

    return 0;
}

int parallel_matrix_mult(Matrix *leftMat, Matrix *rightMat, Matrix *resMat) {
    if (leftMat->cols !=  rightMat->rows ||
        resMat->rows  !=  leftMat->rows  ||
        resMat->cols  !=  rightMat->cols) {
        return -1;
    }

    #pragma omp for collapse(2)
    for (int row = 0; row < resMat->rows; ++row) {
        for (int col = 0; col < resMat->cols; ++col) {
            elem_at(resMat, row, col) = 0.0;
            for (int j = 0; j < leftMat->cols; ++j) {
                elem_at(resMat, row, col) += elem_at(leftMat, row, j) * elem_at(rightMat, j, col);
            }
        }
    }

    return 0;
}

int matrix_add(Matrix *leftMat, Matrix *rightMat) {
    if (leftMat->cols != rightMat->cols ||
        leftMat->rows != rightMat->rows) {
        return -1;
    }

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < leftMat->rows; ++row) {
        for (int col = 0; col < leftMat->cols; ++col) {
            elem_at(leftMat, row, col) += elem_at(rightMat, row, col);
        }
    }

    return 0;
}

int parallel_matrix_add(Matrix *leftMat, Matrix *rightMat) {
    if (leftMat->cols != rightMat->cols ||
        leftMat->rows != rightMat->rows) {
        return -1;
    }

    #pragma omp for collapse(2)
    for (int row = 0; row < leftMat->rows; ++row) {
        for (int col = 0; col < leftMat->cols; ++col) {
            elem_at(leftMat, row, col) += elem_at(rightMat, row, col);
        }
    }

    return 0;
}


int matrix_subtract(Matrix *leftMat, Matrix *rightMat) {
    if (leftMat->cols != rightMat->cols ||
        leftMat->rows != rightMat->rows) {
        return -1;
    }

    #pragma omp parallel for collapse(2)
    for (int row = 0; row < leftMat->rows; ++row) {
        for (int col = 0; col < leftMat->cols; ++col) {
            elem_at(leftMat, row, col) -= elem_at(rightMat, row, col);
        }
    }

    return 0;
}

int parallel_matrix_subtract(Matrix *leftMat, Matrix *rightMat) {
    if (leftMat->cols != rightMat->cols ||
        leftMat->rows != rightMat->rows) {
        return -1;
    }

    #pragma omp for collapse(2)
    for (int row = 0; row < leftMat->rows; ++row) {
        for (int col = 0; col < leftMat->cols; ++col) {
            elem_at(leftMat, row, col) -= elem_at(rightMat, row, col);
        }
    }

    return 0;
}

int matrix_mult_by(Matrix *mat, double val) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < mat->rows; ++row) {
        for (int col = 0; col < mat->cols; ++col) {
            elem_at(mat, row, col) *= val;
        }
    }
    return 0;
}

int parallel_matrix_mult_by(Matrix *mat, double val) {
    #pragma omp for collapse(2)
    for (int row = 0; row < mat->rows; ++row) {
        for (int col = 0; col < mat->cols; ++col) {
            elem_at(mat, row, col) *= val;
        }
    }
    return 0;
}

int matrix_fill_with(Matrix *mat, double val) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < mat->rows; ++row) {
        for (int col = 0; col < mat->cols; ++col) {
            elem_at(mat, row, col) = val;
        }
    }
    return 0;
}

int parallel_matrix_fill_with(Matrix *mat, double val) {
    #pragma omp for collapse(2)
    for (int row = 0; row < mat->rows; ++row) {
        for (int col = 0; col < mat->cols; ++col) {
            elem_at(mat, row, col) = val;
        }
    }
    return 0;
}


int matrix_inner_product(Matrix *mat1, Matrix *mat2, double *result) {
    if (mat1->rows != mat2->rows ||
        mat1->cols != mat2->cols) {
        return -1;
    }

    double res = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:res)
    for (int row = 0; row < mat1->rows; ++row) {
        for (int col = 0; col < mat1->cols; ++col) {
            res += elem_at(mat1, row, col) * elem_at(mat2, row, col);
        }
    }
    *result = res;

    return 0;
}

int parallel_matrix_inner_product(Matrix *mat1, Matrix *mat2, double *result) {
    if (mat1->rows != mat2->rows ||
        mat1->cols != mat2->cols) {
        return -1;
    }

    static double res = 0.0;
    #pragma omp single
    res = 0.0;
    #pragma omp for collapse(2) reduction(+:res)
    for (int row = 0; row < mat1->rows; ++row) {
        for (int col = 0; col < mat1->cols; ++col) {
            res += elem_at(mat1, row, col) * elem_at(mat2, row, col);
        }
    }
    *result = res;

    return 0;
}

double get_norm_2(Matrix *mat) {
    double normSquared;
    matrix_inner_product(mat, mat, &normSquared);
    return sqrt(normSquared);
}

double parallel_get_norm_2(Matrix *mat) {
    static double result = 0.0;
    double normSquared;
    parallel_matrix_inner_product(mat, mat, &normSquared);
    #pragma omp single
    result = sqrt(normSquared);
    return result;
}
