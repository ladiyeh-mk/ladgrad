#include <string.h>
#include "../include/test_util.h"

#define ARENA_IMPLEMENTATION
#include "../include/arena.h"

int main(void) {
    INFO("starting arena test suite");

    // ------------------------------------------------------------------
    SECTION("arena_create");
    {
        CHECK(arena_create(0) == NULL, "capacity 0 returns NULL");
        CHECK(arena_create((size_t) -1) == NULL, "impossible capacity returns NULL (malloc fails)");

        Mem_Arena *a = arena_create(KiB(1));
        CHECK(a != NULL, "valid capacity returns an arena");
        CHECK(a->data != NULL, "backing buffer is allocated");
        CHECK(a->capacity == KiB(1), "capacity is stored");
        CHECK(arena_save(a) == 0, "offset starts at 0");
        CHECK(arena_remaining(a) == KiB(1), "remaining starts at capacity");
        arena_destroy(a);
    }

    // ------------------------------------------------------------------
    SECTION("arena_push: basics");
    {
        Mem_Arena *a = arena_create(256);

        int *one = arena_push_type(a, int);
        CHECK(one != NULL, "arena_push_type returns a pointer");
        *one = 42;
        CHECK(arena_save(a) == sizeof(int), "offset advanced by sizeof(int)");

        int *arr = arena_push_array(a, int, 10);
        CHECK(arr != NULL, "arena_push_array returns a pointer");
        for (int i = 0; i < 10; i++) arr[i] = i * 2;

        CHECK(*one == 42, "first allocation survives later pushes");
        CHECK(arr[9] == 18, "array contents are readable");
        CHECK((u8 *) arr >= (u8 *) one + sizeof(int), "allocations do not overlap");
        CHECK((u8 *) one >= a->data && (u8 *) (arr + 10) <= a->data + a->capacity,
              "allocations stay inside the buffer");
        CHECK(arena_remaining(a) == 256 - arena_save(a), "remaining = capacity - offset");

        // neighbours stay intact when the other is written
        u8 *x = arena_push(a, 16, 1);
        u8 *y = arena_push(a, 16, 1);
        memset(x, 0xAA, 16);
        memset(y, 0xBB, 16);
        int intact = 1;
        for (int i = 0; i < 16; i++) if (x[i] != 0xAA || y[i] != 0xBB) intact = 0;
        CHECK(intact, "writing one block does not touch its neighbour");

        arena_destroy(a);
    }

    // ------------------------------------------------------------------
    SECTION("arena_push: alignment");
    {
        size_t aligns[] = { 1, 2, 4, 8, 16 };
        for (size_t k = 0; k < sizeof(aligns) / sizeof(aligns[0]); k++) {
            Mem_Arena *a = arena_create(256);
            arena_push(a, 1, 1);                       // misalign the offset on purpose
            void *p = arena_push(a, 8, aligns[k]);
            char name[64];
            snprintf(name, sizeof(name), "pointer is aligned to %zu", aligns[k]);
            CHECK(p != NULL && ((uintptr_t) p % aligns[k]) == 0, name);
            arena_destroy(a);
        }

        Mem_Arena *a = arena_create(128);
        arena_push(a, 1, 1);
        CHECK(arena_save(a) == 1, "1-byte push moves offset to 1");
        arena_push(a, 4, 4);
        CHECK(arena_save(a) == 8, "4-byte push at offset 1 pads 3 bytes (offset 8)");
        arena_push(a, 8, 8);
        CHECK(arena_save(a) == 16, "8-byte push at offset 8 needs no padding (offset 16)");
        arena_destroy(a);
    }

    // ------------------------------------------------------------------
    SECTION("arena_push: capacity limits");
    {
        Mem_Arena *a = arena_create(64);
        CHECK(arena_push(a, 128, 1) == NULL, "push larger than capacity returns NULL");
        CHECK(arena_push(a, 65, 1) == NULL, "push one byte over capacity returns NULL");
        CHECK(arena_save(a) == 0, "failed pushes leave the offset unchanged");
        CHECK(arena_push(a, 64, 1) != NULL, "push that exactly fills the arena succeeds");
        CHECK(arena_remaining(a) == 0, "remaining is 0 when full");
        CHECK(arena_push(a, 1, 1) == NULL, "push into a full arena returns NULL");
        arena_destroy(a);

        a = arena_create(64);
        arena_push(a, 10, 1);
        CHECK(arena_push(a, 100, 1) == NULL, "oversized push after a small one returns NULL");
        CHECK(arena_save(a) == 10, "offset is still 10 after the failed push");
        CHECK(arena_push(a, 10, 1) != NULL, "a push that fits still works afterwards");
        arena_destroy(a);

        a = arena_create(64);
        arena_push(a, 61, 1);
        CHECK(arena_push(a, 2, 4) == NULL, "padding can push a small allocation over capacity");
        CHECK(arena_save(a) == 61, "offset unchanged when padding made the push fail");
        arena_destroy(a);

        a = arena_create(64);
        arena_push(a, 8, 1);
        CHECK(arena_push(a, (size_t) -1, 1) == NULL, "push of SIZE_MAX bytes returns NULL (no size_t overflow)");
        arena_destroy(a);
    }

    // ------------------------------------------------------------------
    SECTION("NULL safety");
    {
        CHECK(arena_push(NULL, 8, 8) == NULL, "arena_push(NULL) returns NULL");
        CHECK(arena_save(NULL) == 0, "arena_save(NULL) returns 0");
        CHECK(arena_remaining(NULL) == 0, "arena_remaining(NULL) returns 0");
        CHECK_SURVIVES("arena_reset(NULL) does not crash", arena_reset(NULL));
        CHECK_SURVIVES("arena_restore(NULL) does not crash", arena_restore(NULL, 0));
        CHECK_SURVIVES("arena_destroy(NULL) does not crash", arena_destroy(NULL));

        Mem_Arena no_buffer = { .data = NULL, .offset = 0, .capacity = 64 };
        CHECK(arena_push(&no_buffer, 8, 8) == NULL, "arena without a buffer returns NULL");
    }

    // ------------------------------------------------------------------
    SECTION("arena_save / arena_restore");
    {
        Mem_Arena *a = arena_create(256);

        int *keep = arena_push_type(a, int);
        *keep = 100;
        size_t checkpoint = arena_save(a);
        size_t remaining_at_checkpoint = arena_remaining(a);

        int *temp = arena_push_array(a, int, 20);
        for (int i = 0; i < 20; i++) temp[i] = 999;
        CHECK(arena_save(a) > checkpoint, "temporary pushes move the offset forward");

        arena_restore(a, checkpoint);
        CHECK(arena_save(a) == checkpoint, "restore puts the offset back");
        CHECK(arena_remaining(a) == remaining_at_checkpoint, "restore gives the memory back");
        CHECK(*keep == 100, "data before the checkpoint is untouched");

        int *reused = arena_push_array(a, int, 20);
        CHECK((void *) reused == (void *) temp, "next push reuses the discarded memory");

        // nested checkpoints
        arena_reset(a);
        arena_push(a, 8, 8);
        size_t outer = arena_save(a);
        arena_push(a, 16, 8);
        size_t inner = arena_save(a);
        arena_push(a, 32, 8);
        arena_restore(a, inner);
        CHECK(arena_save(a) == inner, "inner restore works");
        arena_restore(a, outer);
        CHECK(arena_save(a) == outer, "outer restore works after the inner one");
        arena_restore(a, 0);
        CHECK(arena_save(a) == 0, "restore to 0 empties the arena");

        arena_destroy(a);
    }

    // ------------------------------------------------------------------
    SECTION("arena_reset");
    {
        Mem_Arena *a = arena_create(128);
        void *first = arena_push(a, 100, 1);
        CHECK(arena_remaining(a) == 28, "remaining shrinks after a push");

        arena_reset(a);
        CHECK(arena_save(a) == 0, "reset puts the offset at 0");
        CHECK(arena_remaining(a) == 128, "reset gives all the memory back");
        CHECK(arena_push(a, 1, 1) == first, "first push after reset reuses the start of the buffer");
        arena_destroy(a);
    }

    return test_summary();
}

// not gonna lie but this code came from claude code including test_utils.h too
// gcc -Wall -Wextra -o bin/arena_test tests/arena_test.c && ./bin/arena_test