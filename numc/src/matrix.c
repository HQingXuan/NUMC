#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    int num_cols = mat->cols;
    return mat->data[row * num_cols + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    int num_cols = mat->cols;
    mat->data[row * num_cols + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.

    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    matrix *new_matrix = (matrix *) malloc(sizeof(matrix));
    if (!new_matrix) {
        return -2;
    }
    new_matrix->data = calloc(rows * cols, sizeof(double));
    if (!new_matrix->data) {
        free(new_matrix);
        return -2;
    }
    new_matrix->rows = rows;
    new_matrix->cols = cols;
    new_matrix->parent = NULL;
    new_matrix->ref_cnt = 1;
    *mat = new_matrix;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.

    if (mat == NULL) {
        return;
    } else if (mat->parent == NULL) {
        mat->ref_cnt -= 1;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
        }
    } else {
        deallocate_matrix(mat->parent);
        free(mat);
    }
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.

    if (rows <= 0 || cols <= 0) {
        return -1;
    }
    matrix *new_matrix = (matrix *) malloc(sizeof(matrix));
    if (!new_matrix) {
        return -2;
    }
    new_matrix->data = from->data + offset;
    new_matrix->rows = rows;
    new_matrix->cols = cols;
    new_matrix->parent = from;
    from->ref_cnt += 1;
    *mat = new_matrix;
    return 0;
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO
    // naive
    // int i;
    // for (i = 0; i < mat->rows * mat->cols; i += 4) {
    //     *(mat->data + i) = val;
    // }
    
    // With SIMD, loop unrolling, OpenMP speedup
    int i;
    int size = mat->rows * mat->cols;
    __m256d vector = _mm256_set1_pd(val);
    #pragma omp parallel for
    for (i = 0; i < size - size % 16 ; i += 16) {
        _mm256_storeu_pd(mat->data + i, vector);
        _mm256_storeu_pd(mat->data + i + 4, vector);
        _mm256_storeu_pd(mat->data + i + 8, vector);
        _mm256_storeu_pd(mat->data + i + 12, vector);
    }
    // tail case
    for (i = size - size % 16; i < size; i++) {
        *(mat->data + i) = val;
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    // naive
    // int i;
    // for (i = 0; i < mat->rows * mat->cols; i++) {
    //     double val = *(mat->data + i);
    //     if (val < 0) {
    //         val = -val;
    //     }
    //     *(result->data + i) = val;
    // }
    // return 0;

    // with speedups
    int i;
    int size = mat->rows * mat->cols;
    __m256d neg_ones = _mm256_set1_pd(-1.0);
    #pragma omp parallel for
    for (i = 0; i < size - size % 4; i += 4) {
        __m256d vector = _mm256_loadu_pd(mat->data + i);
        // take the max between x and -x
        vector = _mm256_max_pd(vector, _mm256_mul_pd(vector, neg_ones));
        _mm256_storeu_pd(result->data + i, vector);
    }
    //tail case
    for (i = size - size % 4; i < size; i++) {
        double val = *(mat->data + i);
        if (val < 0) {
            val = -val;
        }
        *(result->data + i) = val;
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    int i;
    for (i = 0; i < mat->rows * mat->cols; i++) {
        *(result->data + i) = *(mat->data + i) * (-1);
    }
    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    // naive
    // int i;
    // for (i = 0; i < mat1->rows * mat1->cols; i++) {
    //     *(result->data + i) = *(mat1->data + i) + *(mat2->data + i);
    // }
    // return 0;

    // with speedups
    int i;
    int size = mat1->rows * mat1->cols;
    #pragma omp parallel for
    for (i = 0; i < size - size % 4; i += 4) {
        __m256d vector1 = _mm256_loadu_pd(mat1->data + i);
        __m256d vector2 = _mm256_loadu_pd(mat2->data + i);
        _mm256_storeu_pd(result->data + i, _mm256_add_pd(vector1, vector2));
    }
    // tail case
    for (i = size - size % 4; i < size; i++) {
        *(result->data + i) = *(mat1->data + i) + *(mat2->data + i);
    }
    return 0;
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int i;
    for (i = 0; i < mat1->rows * mat1->cols; i++) {
        *(result->data + i) = *(mat1->data + i) - *(mat2->data + i);
    }
    return 0;
}

/* 
 * Transpose matrix src.
 */
void transpose(int width, int height, int blocksize, matrix *dst, matrix *src) {
    //naive
    #pragma omp parallel for
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            dst->data[y + x * height] = src->data[x + y * width];
        }
    }
    // use cache blocking
    // #pragma omp parallel for
    // for (int x_block = 0; x_block < width; x_block += blocksize) {
    //     for (int y_block = 0; y_block < height; y_block += blocksize) {
    //         for (int x = x_block; x < x_block + blocksize && x < width; x++) {
    //             for (int y = y_block; y < y_block + blocksize && y < height; y++) {
    //                 dst->data[y + x * height] = src->data[x + y * width];
    //             }
    //         }
    //     }
    // }
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.6 TODO
    // naive
    // int i, j, k;
    // int inner_dim = mat1->cols;
    // for (i = 0; i < mat1->rows; i++) {
    //     for (j = 0; j < mat2->cols; j++) {
    //         double sum = 0.0;
    //         for (k = 0; k < inner_dim; k++) {
    //             sum += *(mat1->data + i * inner_dim + k) * *(mat2->data + k * mat2->cols + j);
    //         }
    //         *(result->data + i * mat2->cols + j) = sum;
    //     }
    // }
    // return 0;


    // with speedups
    int inner_dim = mat1->cols;
    int i, j, k;
    // transpose matrix 2
    int blocksize = 50;
    matrix **temp = (matrix **) malloc(sizeof(matrix *));
    allocate_matrix(temp, mat2->cols, mat2->rows);
    matrix *mat2_T = *temp;
    transpose(mat2->cols, mat2->rows, blocksize, mat2_T, mat2);

    //#pragma omp parallel for
    for (i = 0; i < mat1->rows; i++) {
        #pragma omp parallel for
        for (j = 0; j < mat2->cols - mat2->cols % 4; j += 4) {
            __m256d sum_vector1 = _mm256_set1_pd(0.0);
            __m256d sum_vector2 = _mm256_set1_pd(0.0);
            __m256d sum_vector3 = _mm256_set1_pd(0.0);
            __m256d sum_vector4 = _mm256_set1_pd(0.0);
            __m256d vector1, vector2;
            for (k = 0; k < inner_dim - inner_dim % 4; k += 4) {
                vector1 = _mm256_loadu_pd(mat1->data + i * inner_dim + k);
                vector2 = _mm256_loadu_pd(mat2_T->data + j * inner_dim + k);
                sum_vector1 = _mm256_fmadd_pd(vector1, vector2, sum_vector1);

                vector2 = _mm256_loadu_pd(mat2_T->data + (j + 1) * inner_dim + k);
                sum_vector2 = _mm256_fmadd_pd(vector1, vector2, sum_vector2);

                vector2 = _mm256_loadu_pd(mat2_T->data + (j + 2) * inner_dim + k);
                sum_vector3 = _mm256_fmadd_pd(vector1, vector2, sum_vector3);

                vector2 = _mm256_loadu_pd(mat2_T->data + (j + 3) * inner_dim + k);
                sum_vector4 = _mm256_fmadd_pd(vector1, vector2, sum_vector4);
            }
            double sum1[4], sum2[4], sum3[4], sum4[4];
            _mm256_storeu_pd(sum1, sum_vector1);
            _mm256_storeu_pd(sum2, sum_vector2);
            _mm256_storeu_pd(sum3, sum_vector3);
            _mm256_storeu_pd(sum4, sum_vector4);
            // tail case
            double tail1 = 0.0, tail2 = 0.0, tail3 = 0.0, tail4 = 0.0;
            for (k = inner_dim - inner_dim % 4; k < inner_dim; k++) {
                double mat1_val = *(mat1->data + i * inner_dim + k);
                tail1 += mat1_val * *(mat2_T->data + j * inner_dim + k);
                tail2 += mat1_val * *(mat2_T->data + (j + 1) * inner_dim + k);
                tail3 += mat1_val * *(mat2_T->data + (j + 2) * inner_dim + k);
                tail4 += mat1_val * *(mat2_T->data + (j + 3) * inner_dim + k);
            }

            __m256d mul_result = _mm256_set_pd(sum4[0] + sum4[1] + sum4[2] + sum4[3] + tail4, \
                                                sum3[0] + sum3[1] + sum3[2] + sum3[3] + tail3, \
                                                sum2[0] + sum2[1] + sum2[2] + sum2[3] + tail2, \
                                                sum1[0] + sum1[1] + sum1[2] + sum1[3] + tail1);
            _mm256_storeu_pd(result->data + i * mat2->cols + j, mul_result);
        }

        // tail case for mat2 cols
        for (j = mat2->cols - mat2->cols % 4; j < mat2->cols; j++) {
            __m256d sum_vector = _mm256_set1_pd(0.0);
            __m256d vector1, vector2;
            //#pragma omp parallel for
            for (k = 0; k < inner_dim - inner_dim % 4; k += 4) {
                vector1 = _mm256_loadu_pd(mat1->data + i * inner_dim + k);
                vector2 = _mm256_loadu_pd(mat2_T->data + j * inner_dim + k);
                sum_vector = _mm256_fmadd_pd(vector1, vector2, sum_vector);
            }
            double sum[4];
            _mm256_storeu_pd(sum, sum_vector);
            // tail case
            double tail = 0.0;
            for (k = inner_dim - inner_dim % 4; k < inner_dim; k++) {
                tail += *(mat1->data + i * inner_dim + k) * *(mat2_T->data + j * inner_dim + k);
            }
            *(result->data + i * mat2->cols + j) = sum[0] + sum[1] + sum[2] + sum[3] + tail;
        }
    }

    deallocate_matrix(mat2_T);
    free(temp);
    return 0;

}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    int num_side = mat->rows;
    int matrix_size = mat->rows * mat->cols;
    int i;
    #pragma omp parallel for
    for (i = 0; i < num_side * num_side; i++) {
        if (i % (num_side + 1) == 0) {
            *(result->data + i) = 1;
        } else {
            *(result->data + i) = 0;
        }
    }
    if (pow == 0) {
        return 0;
    }
    if (pow == 1) {
        memcpy(result->data, mat->data, matrix_size * sizeof(double));
        return 0;
    }

    matrix **temp = (matrix **) malloc(sizeof(matrix *));
    allocate_matrix(temp, mat->rows, mat->cols);
    matrix *temp_matrix = *temp;

    matrix **copy = (matrix **) malloc(sizeof(matrix *));
    allocate_matrix(copy, mat->rows, mat->cols);
    matrix *mat_copy = *copy;
    memcpy(mat_copy->data, mat->data, matrix_size * sizeof(double));
    
    while (pow > 1) {
        if (pow % 2 == 0) {
            mul_matrix(temp_matrix, mat_copy, mat_copy);
            matrix *swap = temp_matrix;
            temp_matrix = mat_copy;
            mat_copy = swap;
            //memcpy(mat_copy->data, temp_matrix->data, matrix_size * sizeof(double));
            pow = pow / 2;
        } else {
            mul_matrix(temp_matrix, result, mat_copy); // multiply at odd power
            memcpy(result->data, temp_matrix->data, matrix_size * sizeof(double));

            mul_matrix(temp_matrix, mat_copy, mat_copy);
            matrix *swap = temp_matrix;
            temp_matrix = mat_copy;
            mat_copy = swap;
            //memcpy(mat_copy->data, temp_matrix->data, matrix_size * sizeof(double));
            pow = (pow - 1) / 2;
        }
    }
    mul_matrix(temp_matrix, mat_copy, result);
    memcpy(result->data, temp_matrix->data, matrix_size * sizeof(double));

    deallocate_matrix(temp_matrix);
    free(temp);
    deallocate_matrix(mat_copy);
    free(copy);
    return 0;
    
}
