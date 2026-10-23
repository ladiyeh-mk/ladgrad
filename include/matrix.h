#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>

#define ARENA_IMPLEMENTATION
#include "../include/arena.h"

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

// Inplace Operations
void mat_add_inplace(const Matrix *src, Matrix *dest);
void mat_sub_inplace(const Matrix *src, Matrix *dest);
void mat_piecewise_inplace(const Matrix *src, Matrix *dest);

// Helper
void mat_print(Matrix *m);

#ifdef MATRIX_IMPLEMENTATION

Matrix *mat_create(Mem_Arena *arena, const size_t rows, const size_t cols) {
    if (!rows || ! cols) return NULL;

    size_t size = sizeof(Matrix) + (sizeof(float) * rows * cols);
    Matrix *mat = arena_push(arena, size, _Alignof(Matrix));
    if (!mat) return NULL;

    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float *)(mat + 1);

    return mat;
}

Matrix *mat_identity(Mem_Arena *arena, const size_t n) {
    if (!n) return NULL;

    Matrix *mat = mat_create(arena, n, n);
    if (!mat) return NULL;

    size_t total = n * n;
    for (size_t i = 0; i < total; i++) {
        mat->data[i] = 0.0f;
    }

    for (size_t i = 0; i < n; i++) {
        mat->data[i * n + i] = 1.0f;
    }

    return mat;
}

void mat_add(const Matrix *m, const Matrix *n, Matrix *dest) {
    if (!m || !m->data || !n || !n->data || !dest || !dest->data) return;
    if (m->rows != n->rows || m->cols != n->cols) return;
    if (m->rows != dest->rows || m->cols != dest->cols) return;

    size_t size = m->rows * m->cols;
    for (size_t i = 0; i < size; i++) dest->data[i] = m->data[i] + n->data[i];
}

void mat_sub(const Matrix *m, const Matrix *n, Matrix *dest) {
    if (!m || !m->data || !n || !n->data || !dest || !dest->data) return;
    if (m->rows != n->rows || m->cols != n->cols) return;
    if (m->rows != dest->rows || m->cols != dest->cols) return;

    size_t size = m->rows * m->cols;
    for (size_t i = 0; i < size; i++) dest->data[i] = m->data[i] - n->data[i];
}

void mat_mul(const Matrix *m, const Matrix *n, Matrix *dest) {
    if (!m || !m->data || !n || !n->data || !dest || !dest->data) return;
    if (m->cols != n->rows) return;
    if (dest->rows != m->rows || dest->cols != n->cols) return;

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
    if (!m || !m->data || !n || !n->data || !dest || !dest->data) return;
    if (m->rows != n->rows || m->cols != n->cols) return;
    if (m->rows != dest->rows || m->cols != dest->cols) return;

    size_t size = m->rows * m->cols;
    for (size_t i = 0; i < size; i++) {
        dest->data[i] = m->data[i] * n->data[i];
    }
}

void mat_fill(Matrix *m, float val) {
    if (!m || !m->data) return;

    size_t size = m->rows * m->cols;
    for (size_t i = 0; i < size; i++) m->data[i] = val;
}

void mat_scale(Matrix *m, float val) {
    if (m) {
        size_t size = m->rows * m->cols;
        for (size_t i = 0; i < size; i++) m->data[i] *= val;
    }
}

void mat_transpose(const Matrix *src, Matrix *dest) {
    if (!src || !src->data || !dest || !dest->data) return;
    if (dest->rows != src->cols || dest->cols != src->rows) return;

    for (size_t i = 0; i < src->rows; i++) {
        for (size_t j = 0; j < src->cols; j++) {
            dest->data[j * dest->cols + i] = src->data[i * src->cols + j];
        }
    }
}

// ===================================
// ============ INPLACE ==============
// ===================================

void mat_add_inplace(const Matrix *src, Matrix *dest) {
    if (!src || !src->data || !dest || !dest->data) return;
    if (src->rows != dest->rows || src->cols != dest->cols) return;

    size_t size = src->rows * src->cols;
    for (size_t i = 0; i < size; i++) dest->data[i] += src->data[i];
}

void mat_sub_inplace(const Matrix *src, Matrix *dest) {
    if (!src || !src->data || !dest || !dest->data) return;
    if (src->rows != dest->rows || src->cols != dest->cols) return;

    size_t size = src->rows * src->cols;
    for (size_t i = 0; i < size; i++) dest->data[i] -= src->data[i];
}

void mat_piecewise_inplace(const Matrix *src, Matrix *dest) {
    if (!src || !src->data || !dest || !dest->data) return;
    if (src->rows != dest->rows || src->cols != dest->cols) return;

    size_t size = src->rows * src->cols;
    for (size_t i = 0; i < size; i++) dest->data[i] *= src->data[i];
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