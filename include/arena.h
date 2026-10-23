#ifndef ARENA_H
#define ARENA_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

typedef uint8_t u8;
typedef uint64_t u64;

#define KiB(n) ((u64) (n) << 10)
#define MiB(n) ((u64) (n) << 20)
#define GiB(n) ((u64) (n) << 30)

typedef struct {
    u8 *data;
    size_t offset;
    size_t capacity;
} Mem_Arena;

// Function Declarations
Mem_Arena *arena_create(const size_t capacity);
void *arena_push(Mem_Arena *arena, const size_t size, const size_t alignment);
void arena_free(Mem_Arena *arena);
void arena_reset(Mem_Arena *arena);
void arena_restore(Mem_Arena *arena, size_t saved_offset);
size_t arena_save(Mem_Arena *arena);
size_t arena_remaining(Mem_Arena *arena);

// Convenience Macros
#define arena_push_type(arena, type) \
    (type *) arena_push((arena), sizeof(type), _Alignof(type))

#define arena_push_array(arena, type, count) \
    (type *) arena_push((arena), sizeof(type) * (count), _Alignof(type))

#ifdef ARENA_IMPLEMENTATION

Mem_Arena *arena_create(const size_t capacity) {
    if (!capacity) return NULL;

    Mem_Arena *arena = (Mem_Arena *) malloc(sizeof(Mem_Arena));
    if (!arena) return NULL;

    arena->offset   = 0;
    arena->capacity = capacity;
    arena->data     = (u8 *) malloc(sizeof(u8) * capacity);

    if (!arena->data) { free(arena); return NULL; }
    return arena;
}

void *arena_push(Mem_Arena *arena, const size_t size, const size_t alignment) {
    if (!arena || !arena->data) return NULL;

    size_t padding = (alignment - (arena->offset & (alignment - 1))) & (alignment - 1);
    if (arena->offset + size + padding >= arena->capacity) return NULL;

    void *ptr = arena->data + arena->offset + padding;
    arena->offset = arena->offset + padding + size;

    return ptr;
}

void arena_free(Mem_Arena *arena) {
    if (arena) {
        if (arena->data) { free(arena->data); }
        free(arena);
    }
}

void arena_reset(Mem_Arena *arena) {
    if (arena) arena->offset = 0;
}

void arena_restore(Mem_Arena *arena, size_t saved_offset) {
    if (arena) arena->offset = saved_offset;
}

size_t arena_save(Mem_Arena *arena) {
    if (arena) return arena->offset;
    return 0;
}

size_t arena_remaining(Mem_Arena *arena) {
    if (arena) return (size_t) arena->capacity - arena->offset;
    return 0;
}

#endif // ARENA_IMPLEMENTATION
#endif // ARENA_H