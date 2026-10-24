#include <string.h>
#include "../include/test_util.h"

#define ARENA_IMPLEMENTATION
#include "../include/arena.h"

#define MATRIX_IMPLEMENTATION
#include "../include/matrix.h"

typedef void (*bin_fn)(const Matrix *, const Matrix *, Matrix *);
typedef void (*inplace_fn)(const Matrix *, Matrix *);

static Matrix *make(Mem_Arena *arena, size_t r, size_t c, const float *vals) {
    Matrix *m = mat_create(arena, r, c);
    for (size_t i = 0; i < r * c; i++) m->data[i] = vals[i];
    return m;
}

static int all_equal(const Matrix *m, float v) {
    for (size_t i = 0; i < m->rows * m->cols; i++) if (m->data[i] != v) return 0;
    return 1;
}

int main(void) {
    Mem_Arena *arena = arena_create(MiB(1));
    size_t base = arena_save(arena);
    char name[160];

    INFO("starting matrix test suite");
    INFO("the [MATRIX ERROR] messages of the abort tests are silenced (they run in child processes)");

    // ------------------------------------------------------------------
    SECTION("mat_create");
    {
        CHECK(mat_create(arena, 0, 3) == NULL, "rows = 0 returns NULL");
        CHECK(mat_create(arena, 3, 0) == NULL, "cols = 0 returns NULL");
        CHECK(mat_create(NULL, 2, 2) == NULL, "NULL arena returns NULL");

        Matrix *m = mat_create(arena, 3, 4);
        CHECK(m != NULL && m->rows == 3 && m->cols == 4, "shape is stored");
        CHECK((void *) m->data == (void *) (m + 1), "data sits right after the header");
        CHECK(((uintptr_t) m->data % _Alignof(float)) == 0, "data is float-aligned");
        CHECK(all_equal(m, 0.0f), "fresh matrix is zeroed");

        arena_restore(arena, base);
        arena_push(arena, 4096, 8);
        memset(arena->data, 0xFF, arena->capacity);       // dirty the memory
        arena_restore(arena, base);
        Matrix *dirty = mat_create(arena, 8, 8);
        CHECK(all_equal(dirty, 0.0f), "zeroed even when the arena memory was dirty");

        Mem_Arena *tiny = arena_create(32);
        CHECK(mat_create(tiny, 4, 4) == NULL, "arena too small returns NULL");
        CHECK(arena_save(tiny) == 0, "failed create does not move the offset");
        arena_destroy(tiny);
    }

    // ------------------------------------------------------------------
    SECTION("mat_identity");
    {
        arena_restore(arena, base);
        CHECK(mat_identity(arena, 0) == NULL, "n = 0 returns NULL");

        float want1[1] = { 1 };
        CHECK_MAT("1x1 identity", mat_identity(arena, 1), want1, 1, 1);

        float want3[9] = { 1,0,0, 0,1,0, 0,0,1 };
        CHECK_MAT("3x3 identity", mat_identity(arena, 3), want3, 3, 3);

        memset(arena->data, 0xFF, arena->capacity);
        arena_restore(arena, base);
        CHECK_MAT("identity on dirty memory", mat_identity(arena, 3), want3, 3, 3);
    }

    // ------------------------------------------------------------------
    SECTION("mat_add / mat_sub / mat_piecewise: values");
    {
        arena_restore(arena, base);
        float av[4] = { 1, 2, 3, 4 }, bv[4] = { 10, 20, 30, 40 };
        float nv[4] = { -1, 0, 1, 2 }, mv[4] = { 1, 0, -1, -2 };
        Matrix *a = make(arena, 2, 2, av), *b = make(arena, 2, 2, bv);
        Matrix *neg = make(arena, 2, 2, nv), *mirror = make(arena, 2, 2, mv);
        Matrix *out = mat_create(arena, 2, 2);

        float add_want[4] = { 11, 22, 33, 44 };
        mat_add(a, b, out);
        CHECK_MAT("mat_add", out, add_want, 2, 2);
        float zeros[4] = { 0, 0, 0, 0 };
        mat_add(neg, mirror, out);
        CHECK_MAT("mat_add with negatives and zeros", out, zeros, 2, 2);
        CHECK_MAT("mat_add leaves its inputs untouched", a, av, 2, 2);

        float sub_want[4] = { -9, -18, -27, -36 };
        mat_sub(a, b, out);
        CHECK_MAT("mat_sub (a - b)", out, sub_want, 2, 2);
        float sub_rev[4] = { 9, 18, 27, 36 };
        mat_sub(b, a, out);
        CHECK_MAT("mat_sub is not commutative (b - a)", out, sub_rev, 2, 2);
        mat_sub(a, a, out);
        CHECK_MAT("mat_sub of a matrix with itself", out, zeros, 2, 2);

        float pw_want[4] = { 10, 40, 90, 160 };
        mat_piecewise(a, b, out);
        CHECK_MAT("mat_piecewise", out, pw_want, 2, 2);
        mat_piecewise(a, mirror, out);
        float pw2[4] = { 1, 0, -3, -8 };
        CHECK_MAT("mat_piecewise with zeros and negatives", out, pw2, 2, 2);

        Matrix *alias = make(arena, 2, 2, av);
        mat_add(alias, b, alias);
        CHECK_MAT("mat_add with dest == first input", alias, add_want, 2, 2);
        Matrix *alias2 = make(arena, 2, 2, bv);
        mat_add(a, alias2, alias2);
        CHECK_MAT("mat_add with dest == second input", alias2, add_want, 2, 2);
        Matrix *alias3 = make(arena, 2, 2, av);
        mat_sub(alias3, b, alias3);
        CHECK_MAT("mat_sub with dest == first input", alias3, sub_want, 2, 2);
        Matrix *alias4 = make(arena, 2, 2, av);
        mat_piecewise(alias4, b, alias4);
        CHECK_MAT("mat_piecewise with dest == first input", alias4, pw_want, 2, 2);
    }

    // ------------------------------------------------------------------
    SECTION("mat_add / mat_sub / mat_piecewise: failures");
    {
        arena_restore(arena, base);
        float v4[4] = { 1, 2, 3, 4 };
        Matrix *a22 = make(arena, 2, 2, v4);
        Matrix *b22 = make(arena, 2, 2, v4);
        Matrix *out22 = mat_create(arena, 2, 2);
        Matrix *m32 = mat_create(arena, 3, 2);
        Matrix *m23 = mat_create(arena, 2, 3);
        Matrix nodata = { .rows = 2, .cols = 2, .data = NULL };

        struct { const char *label; bin_fn fn; } ops[] = {
            { "mat_add", mat_add }, { "mat_sub", mat_sub }, { "mat_piecewise", mat_piecewise },
        };
        for (size_t k = 0; k < sizeof(ops) / sizeof(ops[0]); k++) {
            bin_fn fn = ops[k].fn;
            const char *l = ops[k].label;
            snprintf(name, sizeof(name), "%s: NULL first input aborts", l);   CHECK_ABORTS(name, fn(NULL, b22, out22));
            snprintf(name, sizeof(name), "%s: NULL second input aborts", l);  CHECK_ABORTS(name, fn(a22, NULL, out22));
            snprintf(name, sizeof(name), "%s: NULL dest aborts", l);          CHECK_ABORTS(name, fn(a22, b22, NULL));
            snprintf(name, sizeof(name), "%s: input with NULL data aborts", l); CHECK_ABORTS(name, fn(&nodata, b22, out22));
            snprintf(name, sizeof(name), "%s: row counts differ aborts", l);  CHECK_ABORTS(name, fn(a22, m32, out22));
            snprintf(name, sizeof(name), "%s: column counts differ aborts", l); CHECK_ABORTS(name, fn(a22, m23, out22));
            snprintf(name, sizeof(name), "%s: wrong dest shape aborts", l);   CHECK_ABORTS(name, fn(a22, b22, m23));
        }
    }

    // ------------------------------------------------------------------
    SECTION("mat_mul");
    {
        arena_restore(arena, base);
        float av[6] = { 1, 2, 3, 4, 5, 6 }, bv[6] = { 7, 8, 9, 10, 11, 12 };
        Matrix *a = make(arena, 2, 3, av), *b = make(arena, 3, 2, bv);
        Matrix *out = mat_create(arena, 2, 2);
        float want[4] = { 58, 64, 139, 154 };
        mat_mul(a, b, out);
        CHECK_MAT("2x3 times 3x2", out, want, 2, 2);

        float rv[3] = { 1, 2, 3 }, cv[3] = { 4, 5, 6 };
        Matrix *row = make(arena, 1, 3, rv), *col = make(arena, 3, 1, cv);
        Matrix *dot = mat_create(arena, 1, 1);
        float dot_want[1] = { 32 };
        mat_mul(row, col, dot);
        CHECK_MAT("1x3 times 3x1 (dot product)", dot, dot_want, 1, 1);

        Matrix *outer = mat_create(arena, 3, 3);
        float outer_want[9] = { 4,8,12, 5,10,15, 6,12,18 };
        mat_mul(col, row, outer);
        CHECK_MAT("3x1 times 1x3 (outer product)", outer, outer_want, 3, 3);

        Matrix *id3 = mat_identity(arena, 3), *id2 = mat_identity(arena, 2);
        Matrix *left = mat_create(arena, 2, 3), *right = mat_create(arena, 2, 3);
        mat_mul(a, id3, left);
        CHECK_MAT("matrix times identity", left, av, 2, 3);
        mat_mul(id2, a, right);
        CHECK_MAT("identity times matrix", right, av, 2, 3);

        Matrix *dirty = mat_create(arena, 2, 2);
        mat_fill(dirty, 999.0f);
        mat_mul(a, b, dirty);
        CHECK_MAT("dest is overwritten, not accumulated into", dirty, want, 2, 2);

        float one_a[1] = { 3 }, one_b[1] = { -4 }, one_want[1] = { -12 };
        Matrix *s1 = make(arena, 1, 1, one_a), *s2 = make(arena, 1, 1, one_b), *s3 = mat_create(arena, 1, 1);
        mat_mul(s1, s2, s3);
        CHECK_MAT("1x1 times 1x1", s3, one_want, 1, 1);

        Matrix *zero = mat_create(arena, 3, 2);
        Matrix *zout = mat_create(arena, 2, 2);
        float zwant[4] = { 0, 0, 0, 0 };
        mat_mul(a, zero, zout);
        CHECK_MAT("multiplying by a zero matrix", zout, zwant, 2, 2);

        Matrix *c23 = make(arena, 2, 3, av);
        Matrix *bad_dest_rows = mat_create(arena, 3, 2), *bad_dest_cols = mat_create(arena, 2, 3);
        Matrix nodata = { .rows = 2, .cols = 3, .data = NULL };
        CHECK_ABORTS("inner dimensions differ aborts", mat_mul(a, c23, out));
        CHECK_ABORTS("wrong dest rows aborts", mat_mul(a, b, bad_dest_rows));
        CHECK_ABORTS("wrong dest cols aborts", mat_mul(a, b, bad_dest_cols));
        CHECK_ABORTS("NULL first input aborts", mat_mul(NULL, b, out));
        CHECK_ABORTS("NULL second input aborts", mat_mul(a, NULL, out));
        CHECK_ABORTS("NULL dest aborts", mat_mul(a, b, NULL));
        CHECK_ABORTS("input with NULL data aborts", mat_mul(&nodata, b, out));
    }

    // ------------------------------------------------------------------
    SECTION("mat_fill / mat_scale");
    {
        arena_restore(arena, base);
        Matrix *m = mat_create(arena, 2, 3);
        mat_fill(m, 3.5f);
        CHECK(all_equal(m, 3.5f), "fill with a positive value");
        mat_fill(m, -2.0f);
        CHECK(all_equal(m, -2.0f), "fill with a negative value");
        mat_fill(m, 0.0f);
        CHECK(all_equal(m, 0.0f), "fill with zero");

        float v[4] = { 1, 2, 3, 4 };
        Matrix *s = make(arena, 2, 2, v);
        float w25[4] = { 2.5f, 5, 7.5f, 10 };
        mat_scale(s, 2.5f);
        CHECK_MAT("scale by 2.5", s, w25, 2, 2);
        mat_scale(s, 0.0f);
        CHECK(all_equal(s, 0.0f), "scale by 0");
        Matrix *t = make(arena, 2, 2, v);
        float wneg[4] = { -1, -2, -3, -4 };
        mat_scale(t, -1.0f);
        CHECK_MAT("scale by -1", t, wneg, 2, 2);
        Matrix *u = make(arena, 2, 2, v);
        mat_scale(u, 1.0f);
        CHECK_MAT("scale by 1 changes nothing", u, v, 2, 2);

        Matrix nodata = { .rows = 2, .cols = 2, .data = NULL };
        CHECK_ABORTS("mat_fill(NULL) aborts", mat_fill(NULL, 1.0f));
        CHECK_ABORTS("mat_fill with NULL data aborts", mat_fill(&nodata, 1.0f));
        CHECK_ABORTS("mat_scale(NULL) aborts", mat_scale(NULL, 2.0f));
        CHECK_ABORTS("mat_scale with NULL data aborts", mat_scale(&nodata, 2.0f));
    }

    // ------------------------------------------------------------------
    SECTION("mat_transpose");
    {
        arena_restore(arena, base);
        float v[6] = { 0, 1, 2, 3, 4, 5 };
        Matrix *src = make(arena, 2, 3, v);
        Matrix *dest = mat_create(arena, 3, 2);
        float want[6] = { 0, 3, 1, 4, 2, 5 };
        mat_transpose(src, dest);
        CHECK_MAT("2x3 transposed to 3x2", dest, want, 3, 2);

        Matrix *back = mat_create(arena, 2, 3);
        mat_transpose(dest, back);
        CHECK_MAT("transposing twice gives the original", back, v, 2, 3);

        float rv[4] = { 1, 2, 3, 4 };
        Matrix *row = make(arena, 1, 4, rv), *col = mat_create(arena, 4, 1);
        mat_transpose(row, col);
        CHECK_MAT("1x4 row becomes a 4x1 column", col, rv, 4, 1);

        float sq[9] = { 1,2,3, 4,5,6, 7,8,9 }, sq_want[9] = { 1,4,7, 2,5,8, 3,6,9 };
        Matrix *s = make(arena, 3, 3, sq), *st = mat_create(arena, 3, 3);
        mat_transpose(s, st);
        CHECK_MAT("3x3 square transpose", st, sq_want, 3, 3);

        Matrix *same_shape = mat_create(arena, 2, 3);
        Matrix nodata = { .rows = 3, .cols = 2, .data = NULL };
        CHECK_ABORTS("dest with the same shape as src aborts", mat_transpose(src, same_shape));
        CHECK_ABORTS("NULL src aborts", mat_transpose(NULL, dest));
        CHECK_ABORTS("NULL dest aborts", mat_transpose(src, NULL));
        CHECK_ABORTS("dest with NULL data aborts", mat_transpose(src, &nodata));
    }

    // ------------------------------------------------------------------
    SECTION("in-place operations");
    {
        arena_restore(arena, base);
        float av[4] = { 1, 2, 3, 4 }, bv[4] = { 10, 20, 30, 40 };
        Matrix *src = make(arena, 2, 2, av);

        Matrix *d1 = make(arena, 2, 2, bv);
        float add_want[4] = { 11, 22, 33, 44 };
        mat_add_inplace(src, d1);
        CHECK_MAT("mat_add_inplace (dest += src)", d1, add_want, 2, 2);
        CHECK_MAT("mat_add_inplace leaves src untouched", src, av, 2, 2);

        Matrix *d2 = make(arena, 2, 2, bv);
        float sub_want[4] = { 9, 18, 27, 36 };
        mat_sub_inplace(src, d2);
        CHECK_MAT("mat_sub_inplace (dest -= src)", d2, sub_want, 2, 2);

        Matrix *d3 = make(arena, 2, 2, bv);
        float pw_want[4] = { 10, 40, 90, 160 };
        mat_piecewise_inplace(src, d3);
        CHECK_MAT("mat_piecewise_inplace (dest *= src)", d3, pw_want, 2, 2);

        Matrix *self_add = make(arena, 2, 2, av);
        float doubled[4] = { 2, 4, 6, 8 };
        mat_add_inplace(self_add, self_add);
        CHECK_MAT("mat_add_inplace with src == dest doubles", self_add, doubled, 2, 2);
        Matrix *self_sub = make(arena, 2, 2, av);
        float zeros[4] = { 0, 0, 0, 0 };
        mat_sub_inplace(self_sub, self_sub);
        CHECK_MAT("mat_sub_inplace with src == dest zeroes", self_sub, zeros, 2, 2);
        Matrix *self_pw = make(arena, 2, 2, av);
        float squared[4] = { 1, 4, 9, 16 };
        mat_piecewise_inplace(self_pw, self_pw);
        CHECK_MAT("mat_piecewise_inplace with src == dest squares", self_pw, squared, 2, 2);

        Matrix *m23 = mat_create(arena, 2, 3), *m32 = mat_create(arena, 3, 2);
        Matrix nodata = { .rows = 2, .cols = 2, .data = NULL };
        struct { const char *label; inplace_fn fn; } ops[] = {
            { "mat_add_inplace", mat_add_inplace }, { "mat_sub_inplace", mat_sub_inplace },
            { "mat_piecewise_inplace", mat_piecewise_inplace },
        };
        for (size_t k = 0; k < sizeof(ops) / sizeof(ops[0]); k++) {
            inplace_fn fn = ops[k].fn;
            const char *l = ops[k].label;
            snprintf(name, sizeof(name), "%s: NULL src aborts", l);        CHECK_ABORTS(name, fn(NULL, d1));
            snprintf(name, sizeof(name), "%s: NULL dest aborts", l);       CHECK_ABORTS(name, fn(src, NULL));
            snprintf(name, sizeof(name), "%s: src with NULL data aborts", l); CHECK_ABORTS(name, fn(&nodata, d1));
            snprintf(name, sizeof(name), "%s: rows differ aborts", l);     CHECK_ABORTS(name, fn(m32, d1));
            snprintf(name, sizeof(name), "%s: cols differ aborts", l);     CHECK_ABORTS(name, fn(m23, d1));
        }
    }

    // ------------------------------------------------------------------
    SECTION("mat_add_rows");
    {
        arena_restore(arena, base);
        float mv[12] = { 0,1,2,3, 4,5,6,7, 8,9,10,11 }, vv[4] = { 1, 2, 3, 4 };
        Matrix *m = make(arena, 3, 4, mv), *v = make(arena, 1, 4, vv);
        Matrix *out = mat_create(arena, 3, 4);
        float want[12] = { 1,3,5,7, 5,7,9,11, 9,11,13,15 };
        mat_add_rows(m, v, out);
        CHECK_MAT("3x4 plus a 1x4 row", out, want, 3, 4);
        CHECK_MAT("matrix input is untouched", m, mv, 3, 4);
        CHECK_MAT("row input is untouched", v, vv, 1, 4);

        float z4[4] = { 0, 0, 0, 0 };
        Matrix *zero = make(arena, 1, 4, z4), *same = mat_create(arena, 3, 4);
        mat_add_rows(m, zero, same);
        CHECK_MAT("adding a zero row changes nothing", same, mv, 3, 4);

        float one_row[4] = { 5, 5, 5, 5 }, one_want[4] = { 6, 7, 8, 9 };
        float rv[4] = { 1, 2, 3, 4 };
        Matrix *single = make(arena, 1, 4, one_row), *r = make(arena, 1, 4, rv), *so = mat_create(arena, 1, 4);
        mat_add_rows(single, r, so);
        CHECK_MAT("a single-row matrix works too", so, one_want, 1, 4);

        Matrix *alias = make(arena, 3, 4, mv);
        mat_add_rows(alias, v, alias);
        CHECK_MAT("dest == m works", alias, want, 3, 4);

        Matrix *two_rows = mat_create(arena, 2, 4), *wrong_cols = mat_create(arena, 1, 3);
        Matrix *wrong_dest = mat_create(arena, 2, 4);
        Matrix nodata = { .rows = 1, .cols = 4, .data = NULL };
        CHECK_ABORTS("bias with 2 rows aborts", mat_add_rows(m, two_rows, out));
        CHECK_ABORTS("bias column count mismatch aborts", mat_add_rows(m, wrong_cols, out));
        CHECK_ABORTS("dest shape mismatch aborts", mat_add_rows(m, v, wrong_dest));
        CHECK_ABORTS("NULL matrix aborts", mat_add_rows(NULL, v, out));
        CHECK_ABORTS("NULL bias aborts", mat_add_rows(m, NULL, out));
        CHECK_ABORTS("NULL dest aborts", mat_add_rows(m, v, NULL));
        CHECK_ABORTS("bias with NULL data aborts", mat_add_rows(m, &nodata, out));
    }

    // ------------------------------------------------------------------
    SECTION("mat_sum_rows_inplace");
    {
        arena_restore(arena, base);
        float mv[12] = { 0,1,2,3, 4,5,6,7, 8,9,10,11 };
        Matrix *m = make(arena, 3, 4, mv);
        Matrix *sums = mat_create(arena, 1, 4);
        float w1[4] = { 12, 15, 18, 21 }, w2[4] = { 24, 30, 36, 42 };
        mat_sum_rows_inplace(m, sums);
        CHECK_MAT("column sums of a 3x4", sums, w1, 1, 4);
        mat_sum_rows_inplace(m, sums);
        CHECK_MAT("a second call accumulates", sums, w2, 1, 4);
        CHECK_MAT("src is untouched", m, mv, 3, 4);

        float start[4] = { 100, 100, 100, 100 }, w3[4] = { 112, 115, 118, 121 };
        Matrix *offset = make(arena, 1, 4, start);
        mat_sum_rows_inplace(m, offset);
        CHECK_MAT("adds on top of a non-zero dest", offset, w3, 1, 4);

        float rv[4] = { 1, 2, 3, 4 };
        Matrix *single = make(arena, 1, 4, rv), *d = mat_create(arena, 1, 4);
        mat_sum_rows_inplace(single, d);
        CHECK_MAT("a single-row src is just added", d, rv, 1, 4);

        Matrix *two_rows = mat_create(arena, 2, 4), *wrong_cols = mat_create(arena, 1, 3);
        Matrix nodata = { .rows = 1, .cols = 4, .data = NULL };
        CHECK_ABORTS("dest with 2 rows aborts", mat_sum_rows_inplace(m, two_rows));
        CHECK_ABORTS("dest column count mismatch aborts", mat_sum_rows_inplace(m, wrong_cols));
        CHECK_ABORTS("NULL src aborts", mat_sum_rows_inplace(NULL, sums));
        CHECK_ABORTS("NULL dest aborts", mat_sum_rows_inplace(m, NULL));
        CHECK_ABORTS("dest with NULL data aborts", mat_sum_rows_inplace(m, &nodata));
    }

    // ------------------------------------------------------------------
    SECTION("mat_print");
    {
        arena_restore(arena, base);
        float v[4] = { 1.23f, 4.56f, 7.89f, 0.0f };
        Matrix *m = make(arena, 2, 2, v);
        CHECK_SURVIVES("mat_print(NULL) does not crash", mat_print(NULL));
        CHECK_SURVIVES("mat_print on a valid matrix does not crash", mat_print(m));
        INFO("visual check of mat_print:");
        mat_print(m);
    }

    arena_destroy(arena);
    return test_summary();
}

// this code too came from claude code
// gcc -Wall -Wextra -o bin/matrix_test tests/matrix_test.c && ./bin/matrix_test