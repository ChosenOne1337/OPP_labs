#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    int rows;
    int cols;
    double *data;
} Matrix;

#define elem_at(mat, row, col) \
    ((mat)->data[(row) * (mat)->cols + (col)])

Matrix* matrix_create(int rows, int cols);

void matrix_destroy(Matrix *mat);

int matrix_mult(Matrix *leftMat, Matrix *rightMat, Matrix *resMat);

int matrix_transpose(Matrix *mat);

int matrix_subtract(Matrix *leftMat, Matrix *rightMat);

int matrix_add(Matrix *leftMat, Matrix *rightMat);

int matrix_mult_by(Matrix *mat, double val);

int matrix_partial_mult(Matrix *leftMat, Matrix *rightMat, Matrix *resMat,
                        int subRows, int subCols, int rowOffset, int colOffset);

int matrix_fill_with(Matrix *mat, double val);

int matrix_inner_product(Matrix *mat1, Matrix *mat2, double *result);

double get_norm_2(Matrix *mat);

void matrix_randomize(Matrix *mat, double amp);

#endif // MATRIX_H
