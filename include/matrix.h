#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include "../include/arena.h"

// Shape/NULL error reporting. Aborts by default so bugs surface immediately.
// Define MAT_NO_ABORT to print the error and return instead.
#ifdef MAT_NO_ABORT
#define MAT_FAIL(...) do {                                 \
    fprintf(stderr, "[MATRIX ERROR] %s: ", __func__);      \
    fprintf(stderr, __VA_ARGS__);                          \
    fprintf(stderr, "\n");                                 \
    return;                                                \
} while (0)
#else
#define MAT_FAIL(...) do {                                 \
    fprintf(stderr, "[MATRIX ERROR] %s: ", __func__);      \
    fprintf(stderr, __VA_ARGS__);                          \
    fprintf(stderr, "\n");                                 \
    abort();                                               \
} while (0)
#endif

#define MAT_CHECK_VALID(m) \
    do { if (!(m) || !(m)->data) MAT_FAIL("NULL or uninitialized matrix: " #m); } while (0)

#define MAT_CHECK_SAME_SHAPE(a, b) \
    do { if ((a)->rows != (b)->rows || (a)->cols != (b)->cols) \
        MAT_FAIL(#a " is (%zu, %zu) but " #b " is (%zu, %zu)", \
                    (a)->rows, (a)->cols, (b)->rows, (b)->cols); } while (0)

typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

// Function Declarations
Matrix *mat_create(Mem_Arena *arena, const size_t rows, const size_t cols);
Matrix *mat_identity(Mem_Arena *arena, const size_t n);
void mat_add(const Matrix *m, const Matrix *n, Matrix *dest);
void mat_sub(const Matrix *m, const Matrix *n, Matrix *dest);
void mat_mul(const Matrix *m, const Matrix *n, Matrix *dest);
void mat_piecewise(const Matrix *m, const Matrix *n, Matrix *dest);
void mat_fill(Matrix *m, float val);
void mat_scale(Matrix *m, float val);
void mat_transpose(const Matrix *src, Matrix *dest);
void mat_add_rows(const Matrix *m, const Matrix *v, Matrix *dest);

// Inplace Operations
void mat_add_inplace(const Matrix *src, Matrix *dest);
void mat_sub_inplace(const Matrix *src, Matrix *dest);
void mat_piecewise_inplace(const Matrix *src, Matrix *dest);
void mat_sum_rows_inplace(const Matrix *src, Matrix *dest);

// Helper
void mat_print(Matrix *m);

#ifdef MATRIX_IMPLEMENTATION

Matrix *mat_create(Mem_Arena *arena, const size_t rows, const size_t cols) {
    if (!rows || !cols) return NULL;

    size_t size = sizeof(Matrix) + (sizeof(float) * rows * cols);
    Matrix *mat = arena_push(arena, size, _Alignof(Matrix));
    if (!mat) return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float *)(mat + 1);

    size_t total = rows * cols;
    for (size_t i = 0; i < total; i++) mat->data[i] = 0.0f;

    return mat;
}

Matrix *mat_identity(Mem_Arena *arena, const size_t n) {
    if (!n) return NULL;

    Matrix *mat = mat_create(arena, n, n);
    if (!mat) return NULL;

    for (size_t i = 0; i < n; i++) {
        mat->data[i * n + i] = 1.0f;
    }

    return mat;
}

void mat_add(const Matrix *m, const Matrix *n, Matrix *dest) {
    MAT_CHECK_VALID(m); MAT_CHECK_VALID(n); MAT_CHECK_VALID(dest);
    MAT_CHECK_SAME_SHAPE(m, n);
    MAT_CHECK_SAME_SHAPE(m, dest);

    size_t size = m->rows * m->cols;
    for (size_t i = 0; i < size; i++) dest->data[i] = m->data[i] + n->data[i];
}

void mat_sub(const Matrix *m, const Matrix *n, Matrix *dest) {
    MAT_CHECK_VALID(m); MAT_CHECK_VALID(n); MAT_CHECK_VALID(dest);
    MAT_CHECK_SAME_SHAPE(m, n);
    MAT_CHECK_SAME_SHAPE(m, dest);

    size_t size = m->rows * m->cols;
    for (size_t i = 0; i < size; i++) dest->data[i] = m->data[i] - n->data[i];
}

void mat_mul(const Matrix *m, const Matrix *n, Matrix *dest) {
    MAT_CHECK_VALID(m); MAT_CHECK_VALID(n); MAT_CHECK_VALID(dest);

    if (m->cols != n->rows)
        MAT_FAIL("inner dimensions differ: (%zu, %zu) x (%zu, %zu)",
                    m->rows, m->cols, n->rows, n->cols);
    if (dest->rows != m->rows || dest->cols != n->cols)
        MAT_FAIL("dest is (%zu, %zu) but result should be (%zu, %zu)",
                    dest->rows, dest->cols, m->rows, n->cols);

    // Zero out destination first
    size_t dest_size = dest->rows * dest->cols;
    for (size_t idx = 0; idx < dest_size; idx++) dest->data[idx] = 0.0f;

    // ikj loop order for cache efficiency
    for (size_t i = 0; i < m->rows; i++) {
        for (size_t k = 0; k < m->cols; k++) {
            float r = m->data[i * m->cols + k];
            for (size_t j = 0; j < n->cols; j++) {
                dest->data[i * dest->cols + j] += r * n->data[k * n->cols + j];
            }
        }
    }
}

void mat_piecewise(const Matrix *m, const Matrix *n, Matrix *dest) {
    MAT_CHECK_VALID(m); MAT_CHECK_VALID(n); MAT_CHECK_VALID(dest);
    MAT_CHECK_SAME_SHAPE(m, n);
    MAT_CHECK_SAME_SHAPE(m, dest);

    size_t size = m->rows * m->cols;
    for (size_t i = 0; i < size; i++) {
        dest->data[i] = m->data[i] * n->data[i];
    }
}

void mat_fill(Matrix *m, float val) {
    MAT_CHECK_VALID(m);

    size_t size = m->rows * m->cols;
    for (size_t i = 0; i < size; i++) m->data[i] = val;
}

void mat_scale(Matrix *m, float val) {
    MAT_CHECK_VALID(m);

    size_t size = m->rows * m->cols;
    for (size_t i = 0; i < size; i++) m->data[i] *= val;
}

void mat_transpose(const Matrix *src, Matrix *dest) {
    MAT_CHECK_VALID(src); MAT_CHECK_VALID(dest);

    if (dest->rows != src->cols || dest->cols != src->rows)
        MAT_FAIL("src is (%zu, %zu) so dest must be (%zu, %zu), got (%zu, %zu)",
                    src->rows, src->cols, src->cols, src->rows, dest->rows, dest->cols);

    for (size_t i = 0; i < src->rows; i++) {
        for (size_t j = 0; j < src->cols; j++) {
            dest->data[j * dest->cols + i] = src->data[i * src->cols + j];
        }
    }
}

void mat_add_rows(const Matrix *m, const Matrix *v, Matrix *dest) {
    MAT_CHECK_VALID(m); MAT_CHECK_VALID(v); MAT_CHECK_VALID(dest);
    MAT_CHECK_SAME_SHAPE(m, dest);

    if (v->rows != 1 || v->cols != m->cols)
        MAT_FAIL("row vector is (%zu, %zu) but must be (1, %zu)", v->rows, v->cols, m->cols);

    for (size_t i = 0; i < m->rows; i++) {
        for (size_t j = 0; j < m->cols; j++) {
            dest->data[i * dest->cols + j] = m->data[i * m->cols + j] + v->data[j];
        }
    }
}

// ===================================
// ============ INPLACE ==============
// ===================================

void mat_add_inplace(const Matrix *src, Matrix *dest) {
    MAT_CHECK_VALID(src); MAT_CHECK_VALID(dest);
    MAT_CHECK_SAME_SHAPE(src, dest);

    size_t size = src->rows * src->cols;
    for (size_t i = 0; i < size; i++) dest->data[i] += src->data[i];
}

void mat_sub_inplace(const Matrix *src, Matrix *dest) {
    MAT_CHECK_VALID(src); MAT_CHECK_VALID(dest);
    MAT_CHECK_SAME_SHAPE(src, dest);

    size_t size = src->rows * src->cols;
    for (size_t i = 0; i < size; i++) dest->data[i] -= src->data[i];
}

void mat_piecewise_inplace(const Matrix *src, Matrix *dest) {
    MAT_CHECK_VALID(src); MAT_CHECK_VALID(dest);
    MAT_CHECK_SAME_SHAPE(src, dest);

    size_t size = src->rows * src->cols;
    for (size_t i = 0; i < size; i++) dest->data[i] *= src->data[i];
}

void mat_sum_rows_inplace(const Matrix *src, Matrix *dest) {
    MAT_CHECK_VALID(src); MAT_CHECK_VALID(dest);

    if (dest->rows != 1 || dest->cols != src->cols)
        MAT_FAIL("dest is (%zu, %zu) but must be (1, %zu)", dest->rows, dest->cols, src->cols);

    for (size_t i = 0; i < src->rows; i++) {
        for (size_t j = 0; j < src->cols; j++) {
            dest->data[j] += src->data[i * src->cols + j];
        }
    }
}

// ===================================
// ============  HELPER  =============
// ===================================

void mat_print(Matrix *m) {
    if (!m || !m->data) {
        printf("[INFOS] Matrix is NULL or uninitialized.\n");
        return;
    }

    printf("=================================================================================================\n");
    printf("[INFOS]\n\t>> shape: (%zu, %zu)\n\t>> Content:\n", m->rows, m->cols);

    for (size_t r = 0; r < m->rows; r++) {
        for (size_t c = 0; c < m->cols; c++) {
            printf("%s%.2f%s%s", (c == 0) ? "\t\t[" : "", m->data[r * m->cols + c], (c < m->cols - 1) ? " " : "", (c == m->cols - 1) ? "]\n" : "");
        }
    }
    printf("=================================================================================================\n");
}

#endif // MATRIX_IMPLEMENTATION
#endif // MATRIX_H