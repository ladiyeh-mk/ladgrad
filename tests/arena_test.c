#define ARENA_IMPLEMENTATION
#include "../include/arena.h"

#include <stdio.h>

int main(void) {
    Mem_Arena *arena = arena_create(KiB(1));

    int size   = 10;
    int *array = arena_push(arena, sizeof(int) * size, _Alignof(int));
    for (int i = 0; i < size; i++) array[i] = i;
    for (int i = 0; i < size; i++) printf("array[%d] = %d\n", i, array[i]);

    size_t remaining = arena_remaining(arena);
    printf("remaining mem: %zu\n", remaining);

    int buffer_size   = 50;
    char *char_buffer = arena_push(arena, sizeof(char) * buffer_size, _Alignof(char));
    fgets(char_buffer, buffer_size, stdin);
    printf("char buffer = %s\n", char_buffer);

    remaining = arena_remaining(arena);
    printf("remaining mem: %zu\n", remaining);

    arena_free(arena);
    return 0;
}

// gcc -o bin/arena_test tests/arena_test.c && ./bin/arena_test 