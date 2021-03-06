#include <stdlib.h>
#include <math.h>
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

int matrix_transpose(Matrix *mat) {
    Matrix *newMat = matrix_create(mat->cols, mat->rows);
    if (newMat == NULL) {
        return -1;
    }
    for (int row = 0; row < mat->rows; ++row) {
        for (int col = 0; col < mat->cols; ++col) {
            elem_at(newMat, col, row) = elem_at(mat, row, col);
        }
    }
    mat->rows = newMat->rows;
    mat->cols = newMat->cols;
    double *temp = newMat->data;
    newMat->data = mat->data;
    mat->data = temp;
    matrix_destroy(newMat);
    return 0;
}

int matrix_partial_mult(Matrix *leftMat, Matrix *rightMat, Matrix *resMat,
                        int subRows, int subCols, int rowOffset, int colOffset) {
    if (rowOffset < 0   || colOffset < 0        ||
        subCols < 0     || subRows < 0          ||
        rowOffset + subRows > leftMat->rows     ||
        colOffset + subCols > leftMat->cols     ||
        rightMat->rows  != subCols              ||
        resMat->rows    != subRows              ||
        resMat->cols    != rightMat->cols)       {
        return -1;
    }

    for (int row = 0; row < resMat->rows; ++row) {
        for (int col = 0; col < resMat->cols; ++col) {
            elem_at(resMat, row, col) = 0.0;
            for (int j = 0; j < subCols; ++j) {
                elem_at(resMat, row, col) +=
                        elem_at(leftMat, rowOffset + row, colOffset + j) * elem_at(rightMat, j, col);
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

    for (int row = 0; row < leftMat->rows; ++row) {
        for (int col = 0; col < leftMat->cols; ++col) {
            elem_at(leftMat, row, col) -= elem_at(rightMat, row, col);
        }
    }

    return 0;
}

int matrix_mult_by(Matrix *mat, double val) {
    for (int row = 0; row < mat->rows; ++row) {
        for (int col = 0; col < mat->cols; ++col) {
            elem_at(mat, row, col) *= val;
        }
    }
    return 0;
}

int matrix_fill_with(Matrix *mat, double val) {
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

    *result = 0.0;
    for (int row = 0; row < mat1->rows; ++row) {
        for (int col = 0; col < mat1->cols; ++col) {
            *result += elem_at(mat1, row, col) * elem_at(mat2, row, col);
        }
    }

    return 0;
}

double get_norm_2(Matrix *mat) {
    double normSquared;
    matrix_inner_product(mat, mat, &normSquared);
    return sqrt(normSquared);
}

void matrix_randomize(Matrix *mat, double amp) {
    for (int row = 0; row < mat->rows; ++row) {
        for (int col = 0; col < mat->cols; ++col) {
            elem_at(mat, row, col) = amp * (2.0 * rand() / RAND_MAX - 1.0);
        }
    }
}
