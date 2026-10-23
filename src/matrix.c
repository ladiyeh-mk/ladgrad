#include <stdio.h>

#define ARENA_IMPLEMENTATION
#include "../include/arena.h"

typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

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

int main(void) {
    // 1. Create a 4 KiB memory arena
    Mem_Arena *arena = arena_create(KiB(4));
    if (!arena) {
        fprintf(stderr, "Failed to create arena\n");
        return 1;
    }

    // 2. Allocate and initialize matrices A and B (2x2)
    Matrix *a = mat_create(arena, 2, 2);
    Matrix *b = mat_create(arena, 2, 2);
    Matrix *dest = mat_create(arena, 2, 2);

    a->data[0] = 1.0f; a->data[1] = 2.0f;
    a->data[2] = 3.0f; a->data[3] = 4.0f;

    b->data[0] = 5.0f; b->data[1] = 6.0f;
    b->data[2] = 7.0f; b->data[3] = 8.0f;

    mat_print(a, "Matrix A");
    mat_print(b, "Matrix B");

    // 3. Test Addition
    mat_add(a, b, dest);
    mat_print(dest, "A + B");

    // 4. Test Multiplication
    mat_mul(a, b, dest);
    mat_print(dest, "A * B (Cache-Optimized ikj)");

    // 5. Test Identity Matrix & Transpose
    Matrix *id = mat_identity(arena, 2);
    mat_print(id, "Identity Matrix (2x2)");

    Matrix *trans = mat_create(arena, 2, 2);
    mat_transpose(a, trans);
    mat_print(trans, "Transpose of A");

    // 6. Print remaining arena space & clean up everything in one go
    printf("Remaining arena memory: %zu bytes\n", arena_remaining(arena));
    
    arena_free(arena);
    printf("Arena freed cleanly. Zero leaks!\n");

    return 0;
}