#include <stdio.h>

#define ARENA_IMPLEMENTATION
#include "../include/arena.h"

#define MATRIX_IMPLEMENTATION
#include "../include/matrix.h"

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

    mat_print(a);
    mat_print(b);

    // 3. Test Addition
    mat_add(a, b, dest);
    mat_print(dest);

    // 4. Test Multiplication
    mat_mul(a, b, dest);
    mat_print(dest);

    // 5. Test Identity Matrix & Transpose
    Matrix *id = mat_identity(arena, 2);
    mat_print(id);

    Matrix *trans = mat_create(arena, 2, 2);
    mat_transpose(a, trans);
    mat_print(trans);

    // 6. Print remaining arena space & clean up everything in one go
    printf("Remaining arena memory: %zu bytes\n", arena_remaining(arena));
    
    arena_free(arena);
    printf("Arena freed cleanly. Zero leaks!\n");

    return 0;
}
// gcc -o bin/matrix_test tests/matrix_test.c && ./bin/matrix_test