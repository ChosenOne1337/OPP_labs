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
int parallel_matrix_mult(Matrix *leftMat, Matrix *rightMat, Matrix *resMat);

int matrix_subtract(Matrix *leftMat, Matrix *rightMat);
int parallel_matrix_subtract(Matrix *leftMat, Matrix *rightMat);

int matrix_add(Matrix *leftMat, Matrix *rightMat);
int parallel_matrix_add(Matrix *leftMat, Matrix *rightMat);

int matrix_mult_by(Matrix *mat, double val);
int parallel_matrix_mult_by(Matrix *mat, double val);

int matrix_fill_with(Matrix *mat, double val);
int parallel_matrix_fill_with(Matrix *mat, double val);

int matrix_inner_product(Matrix *mat1, Matrix *mat2, double *result);
int parallel_matrix_inner_product(Matrix *mat1, Matrix *mat2, double *result);

double get_norm_2(Matrix *mat);
double parallel_get_norm_2(Matrix *mat);

#endif // MATRIX_H
