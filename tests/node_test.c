#include <string.h>
#include "../include/test_util.h"

#define ARENA_IMPLEMENTATION
#define MATRIX_IMPLEMENTATION
#define NODE_IMPLEMENTATION
#include "../include/node.h"

typedef Node *(*binary_node_fn)(Mem_Arena *, Node *, Node *, Tape *);
typedef Node *(*unary_node_fn)(Mem_Arena *, Node *, Tape *);

typedef struct { Mem_Arena *arena; Tape *tape; } Env;

static Env env_new(size_t arena_bytes, size_t tape_capacity) {
    Env e;
    e.arena = arena_create(arena_bytes);
    e.tape  = tape_create(e.arena, tape_capacity);
    return e;
}

static void env_free(Env *e) { arena_destroy(e->arena); }

static Node *make_node(Mem_Arena *arena, size_t r, size_t c, int grad, const float *vals) {
    Node *n = node_create(arena, r, c, grad);
    for (size_t i = 0; i < r * c; i++) n->main->data[i] = vals[i];
    return n;
}

static int all_zero(const Matrix *m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) if (m->data[i] != 0.0f) return 0;
    return 1;
}

// Uses up the arena so that exactly `room` bytes are left (offset is aligned first).
static void leave_room(Mem_Arena *arena, size_t room) {
    arena_push(arena, 0, 8);
    size_t rem = arena_remaining(arena);
    if (rem > room) arena_push(arena, rem - room, 1);
}

// ----------------------------------------------------------------------
// numerical gradient check helpers
// ----------------------------------------------------------------------
typedef struct {
    Mem_Arena *arena;
    Tape *tape;
    size_t base;
    Node *x, *W, *b, *c, *t;
} Net;

// mode: 0 tanh, 1 relu, 2 sigmoid, 3 sub then tanh
static Node *net_forward(Net *n, int mode) {
    n->tape->count = 0;
    Node *h = node_mul(n->arena, n->x, n->W, n->tape);
    Node *z = node_add_rows(n->arena, h, n->b, n->tape);
    if (mode == 3) z = node_sub(n->arena, z, n->c, n->tape);

    Node *a;
    switch (mode) {
        case 1:  a = node_relu(n->arena, z, n->tape);    break;
        case 2:  a = node_sigmoid(n->arena, z, n->tape); break;
        default: a = node_tanh(n->arena, z, n->tape);    break;
    }
    return node_mse(n->arena, a, n->t, n->tape);
}

static float max_grad_diff(Net *n, int mode, Node *p) {
    const float eps = 1e-3f;
    float worst = 0.0f;
    for (size_t i = 0; i < p->main->rows * p->main->cols; i++) {
        float orig = p->main->data[i];

        p->main->data[i] = orig + eps;
        arena_restore(n->arena, n->base);
        float lp = net_forward(n, mode)->main->data[0];

        p->main->data[i] = orig - eps;
        arena_restore(n->arena, n->base);
        float lm = net_forward(n, mode)->main->data[0];

        p->main->data[i] = orig;
        float diff = fabsf(p->grad->data[i] - (lp - lm) / (2.0f * eps));
        if (diff > worst) worst = diff;
    }
    return worst;
}

int main(void) {
    char name[200];
    INFO("starting node & autograd test suite");
    INFO("the 'Tape capacity exceeded' lines below come from the tape-full tests and are expected");

    // ------------------------------------------------------------------
    SECTION("tape_create");
    {
        Mem_Arena *a = arena_create(KiB(4));
        Tape *t = tape_create(a, 16);
        CHECK(t != NULL, "valid tape is created");
        CHECK(t != NULL && t->count == 0 && t->capacity == 16 && t->nodes != NULL, "count 0, capacity stored, node array allocated");
        arena_destroy(a);

        a = arena_create(8);
        CHECK(tape_create(a, 4) == NULL, "arena too small for the Tape struct returns NULL");
        arena_destroy(a);

        a = arena_create(256);
        CHECK(tape_create(a, 1000) == NULL, "arena too small for the node array returns NULL");
        CHECK(arena_save(a) == 0, "failed tape_create gives its memory back");
        arena_destroy(a);
    }

    // ------------------------------------------------------------------
    SECTION("node_create");
    {
        Mem_Arena *a = arena_create(KiB(4));
        CHECK(node_create(a, 0, 3, 1) == NULL, "rows = 0 returns NULL");
        CHECK(node_create(a, 3, 0, 1) == NULL, "cols = 0 returns NULL");

        Node *g = node_create(a, 2, 3, 1);
        CHECK(g != NULL, "node with grad is created");
        CHECK(g->require_grad == 1, "require_grad is stored");
        CHECK(g->op == OP_NONE, "op is OP_NONE");
        CHECK(g->parents[0] == NULL && g->parents[1] == NULL, "parents are NULL");
        CHECK(g->main->rows == 2 && g->main->cols == 3, "main has the requested shape");
        CHECK(g->grad != NULL && g->grad->rows == 2 && g->grad->cols == 3, "grad has the same shape as main");
        CHECK(all_zero(g->main) && all_zero(g->grad), "main and grad start at zero");

        Node *n = node_create(a, 2, 3, 0);
        CHECK(n != NULL && n->require_grad == 0, "node without grad is created");
        CHECK(n->grad == NULL, "grad is NULL when require_grad is 0");
        arena_destroy(a);

        a = arena_create(16);
        CHECK(node_create(a, 2, 2, 1) == NULL, "arena too small for the Node struct returns NULL");
        arena_destroy(a);

        a = arena_create(64);
        CHECK(node_create(a, 2, 2, 1) == NULL, "arena too small for main returns NULL");
        CHECK(arena_save(a) == 0, "failed create (main) gives its memory back");
        arena_destroy(a);

        a = arena_create(100);
        CHECK(node_create(a, 2, 2, 1) == NULL, "arena too small for grad returns NULL");
        CHECK(arena_save(a) == 0, "failed create (grad) gives its memory back");
        arena_destroy(a);
    }

    // ------------------------------------------------------------------
    SECTION("binary ops: bookkeeping and require_grad propagation");
    {
        struct { const char *label; binary_node_fn fn; Ops op; } ops[] = {
            { "node_add", node_add, OP_ADD }, { "node_sub", node_sub, OP_SUB }, { "node_mul", node_mul, OP_MUL },
        };
        float v[4] = { 1, 2, 3, 4 };
        for (size_t k = 0; k < sizeof(ops) / sizeof(ops[0]); k++) {
            for (int ga = 0; ga <= 1; ga++) {
                for (int gb = 0; gb <= 1; gb++) {
                    Env e = env_new(KiB(64), 8);
                    Node *a = make_node(e.arena, 2, 2, ga, v), *b = make_node(e.arena, 2, 2, gb, v);
                    Node *out = ops[k].fn(e.arena, a, b, e.tape);
                    int want = ga || gb;

                    snprintf(name, sizeof(name), "%s (grad %d, %d): result exists with shape 2x2", ops[k].label, ga, gb);
                    CHECK(out != NULL && out->main->rows == 2 && out->main->cols == 2, name);
                    snprintf(name, sizeof(name), "%s (grad %d, %d): require_grad is %d", ops[k].label, ga, gb, want);
                    CHECK(out->require_grad == want, name);
                    snprintf(name, sizeof(name), "%s (grad %d, %d): grad is %s", ops[k].label, ga, gb, want ? "allocated" : "NULL");
                    CHECK((out->grad != NULL) == (want != 0), name);
                    snprintf(name, sizeof(name), "%s (grad %d, %d): op and parents recorded", ops[k].label, ga, gb);
                    CHECK(out->op == ops[k].op && out->parents[0] == a && out->parents[1] == b, name);
                    snprintf(name, sizeof(name), "%s (grad %d, %d): recorded on the tape", ops[k].label, ga, gb);
                    CHECK(e.tape->count == 1 && e.tape->nodes[0] == out, name);
                    env_free(&e);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    SECTION("unary ops: bookkeeping and require_grad propagation");
    {
        struct { const char *label; unary_node_fn fn; Ops op; } ops[] = {
            { "node_relu", node_relu, OP_RELU }, { "node_tanh", node_tanh, OP_TANH }, { "node_sigmoid", node_sigmoid, OP_SIGMOID },
        };
        float v[4] = { -1, 0, 1, 2 };
        for (size_t k = 0; k < sizeof(ops) / sizeof(ops[0]); k++) {
            for (int g = 0; g <= 1; g++) {
                Env e = env_new(KiB(64), 8);
                Node *a = make_node(e.arena, 2, 2, g, v);
                Node *out = ops[k].fn(e.arena, a, e.tape);

                snprintf(name, sizeof(name), "%s (grad %d): shape is kept", ops[k].label, g);
                CHECK(out != NULL && out->main->rows == 2 && out->main->cols == 2, name);
                snprintf(name, sizeof(name), "%s (grad %d): require_grad follows the input", ops[k].label, g);
                CHECK(out->require_grad == g && ((out->grad != NULL) == (g != 0)), name);
                snprintf(name, sizeof(name), "%s (grad %d): op, single parent, tape entry", ops[k].label, g);
                CHECK(out->op == ops[k].op && out->parents[0] == a && out->parents[1] == NULL && e.tape->count == 1, name);
                env_free(&e);
            }
        }
    }

    // ------------------------------------------------------------------
    SECTION("forward values");
    {
        Env e = env_new(KiB(64), 64);

        float av[4] = { 1, 2, 3, 4 }, bv[4] = { 10, 20, 30, 40 };
        Node *a = make_node(e.arena, 2, 2, 1, av), *b = make_node(e.arena, 2, 2, 1, bv);
        float add_want[4] = { 11, 22, 33, 44 }, sub_want[4] = { -9, -18, -27, -36 };
        CHECK_MAT("node_add", node_add(e.arena, a, b, e.tape)->main, add_want, 2, 2);
        CHECK_MAT("node_sub", node_sub(e.arena, a, b, e.tape)->main, sub_want, 2, 2);

        float xv[12] = { 0,1,2,3, 4,5,6,7, 8,9,10,11 }, biasv[4] = { 1, 2, 3, 4 };
        Node *x = make_node(e.arena, 3, 4, 0, xv), *bias = make_node(e.arena, 1, 4, 1, biasv);
        float rows_want[12] = { 1,3,5,7, 5,7,9,11, 9,11,13,15 };
        CHECK_MAT("node_add_rows (3x4 + 1x4)", node_add_rows(e.arena, x, bias, e.tape)->main, rows_want, 3, 4);

        float mav[6] = { 1,2,3,4,5,6 }, mbv[6] = { 7,8,9,10,11,12 };
        Node *ma = make_node(e.arena, 2, 3, 1, mav), *mb = make_node(e.arena, 3, 2, 1, mbv);
        float mul_want[4] = { 58, 64, 139, 154 };
        CHECK_MAT("node_mul (2x3 * 3x2)", node_mul(e.arena, ma, mb, e.tape)->main, mul_want, 2, 2);

        float rv[3] = { 1, 2, 3 }, cv[3] = { 4, 5, 6 }, dot_want[1] = { 32 };
        Node *row = make_node(e.arena, 1, 3, 1, rv), *col = make_node(e.arena, 3, 1, 1, cv);
        CHECK_MAT("node_mul (1x3 * 3x1) gives 1x1", node_mul(e.arena, row, col, e.tape)->main, dot_want, 1, 1);

        float rl[5] = { -2, -0.5f, 0, 0.5f, 2 }, rl_want[5] = { 0, 0, 0, 0.5f, 2 };
        Node *r = make_node(e.arena, 1, 5, 1, rl);
        CHECK_MAT("node_relu (negative, zero, positive)", node_relu(e.arena, r, e.tape)->main, rl_want, 1, 5);

        float th[4] = { 0, 1, -1, 10 }, th_want[4] = { 0, 0.761594f, -0.761594f, 1.0f };
        Node *t = make_node(e.arena, 1, 4, 1, th);
        CHECK_MAT("node_tanh", node_tanh(e.arena, t, e.tape)->main, th_want, 1, 4);

        float sg[5] = { 0, 1, -1, 50, -50 }, sg_want[5] = { 0.5f, 0.731059f, 0.268941f, 1.0f, 0.0f };
        Node *s = make_node(e.arena, 1, 5, 1, sg);
        Node *sout = node_sigmoid(e.arena, s, e.tape);
        CHECK_MAT("node_sigmoid (includes +-50, no NaN/inf)", sout->main, sg_want, 1, 5);

        float pv[2] = { 1, 2 }, tv[2] = { 0, 0 }, mse_want[1] = { 2.5f };
        Node *p = make_node(e.arena, 1, 2, 1, pv), *tg = make_node(e.arena, 1, 2, 0, tv);
        CHECK_MAT("node_mse (mean of squared errors)", node_mse(e.arena, p, tg, e.tape)->main, mse_want, 1, 1);
        float same[4] = { 1, 2, 3, 4 }, zero_want[1] = { 0 };
        Node *s1 = make_node(e.arena, 2, 2, 1, same), *s2 = make_node(e.arena, 2, 2, 0, same);
        CHECK_MAT("node_mse of identical matrices is 0", node_mse(e.arena, s1, s2, e.tape)->main, zero_want, 1, 1);
        env_free(&e);
    }

    // ------------------------------------------------------------------
    SECTION("shape errors abort");
    {
        Env e = env_new(KiB(64), 16);
        float v6[6] = { 1, 2, 3, 4, 5, 6 };
        Node *a22 = make_node(e.arena, 2, 2, 1, v6), *b22 = make_node(e.arena, 2, 2, 1, v6);
        Node *a23 = make_node(e.arena, 2, 3, 1, v6), *a32 = make_node(e.arena, 3, 2, 1, v6);
        Node *bias12 = make_node(e.arena, 1, 2, 1, v6), *bias22 = make_node(e.arena, 2, 2, 1, v6);
        Node *bias13 = make_node(e.arena, 1, 3, 1, v6);

        CHECK_ABORTS("node_add with different shapes", node_add(e.arena, a22, a23, e.tape));
        CHECK_ABORTS("node_sub with different shapes", node_sub(e.arena, a22, a32, e.tape));
        CHECK_ABORTS("node_mul with mismatched inner dimensions", node_mul(e.arena, a23, a23, e.tape));
        CHECK_ABORTS("node_add_rows with a 2-row bias", node_add_rows(e.arena, a22, bias22, e.tape));
        CHECK_ABORTS("node_add_rows with the wrong column count", node_add_rows(e.arena, a22, bias13, e.tape));
        CHECK_ABORTS("node_mse with different shapes", node_mse(e.arena, a22, a23, e.tape));
        CHECK(node_add_rows(e.arena, a22, bias12, e.tape) != NULL, "node_add_rows with a correct bias does not abort");
        (void) b22;
        env_free(&e);
    }

    // ------------------------------------------------------------------
    SECTION("out-of-memory and tape-full paths give their memory back");
    {
        float v[4] = { 1, 2, 3, 4 };

        // tape full
        Env e = env_new(KiB(64), 2);
        Node *a = make_node(e.arena, 2, 2, 1, v), *b = make_node(e.arena, 2, 2, 1, v);
        node_add(e.arena, a, b, e.tape);
        node_add(e.arena, a, b, e.tape);
        size_t before = arena_save(e.arena);
        CHECK(node_add(e.arena, a, b, e.tape) == NULL, "op on a full tape returns NULL");
        CHECK(arena_save(e.arena) == before, "full tape: the result matrix is released");
        CHECK(e.tape->count == 2, "full tape: count does not change");
        env_free(&e);

        // result matrix does not fit
        e = env_new(4096, 8);
        a = make_node(e.arena, 2, 2, 1, v); b = make_node(e.arena, 2, 2, 1, v);
        leave_room(e.arena, 8);
        before = arena_save(e.arena);
        CHECK(node_add(e.arena, a, b, e.tape) == NULL, "no room for the result matrix returns NULL");
        CHECK(arena_save(e.arena) == before && e.tape->count == 0, "no room (result): nothing leaked, nothing on the tape");
        env_free(&e);

        // result fits, Node struct does not
        e = env_new(4096, 8);
        a = make_node(e.arena, 2, 2, 1, v); b = make_node(e.arena, 2, 2, 1, v);
        leave_room(e.arena, sizeof(Matrix) + 4 * sizeof(float));
        before = arena_save(e.arena);
        CHECK(node_add(e.arena, a, b, e.tape) == NULL, "no room for the output Node returns NULL");
        CHECK(arena_save(e.arena) == before && e.tape->count == 0, "no room (node): result matrix is released");
        env_free(&e);

        // result and Node fit, grad does not
        e = env_new(4096, 8);
        a = make_node(e.arena, 2, 2, 1, v); b = make_node(e.arena, 2, 2, 1, v);
        leave_room(e.arena, sizeof(Matrix) + 4 * sizeof(float) + sizeof(Node));
        before = arena_save(e.arena);
        CHECK(node_add(e.arena, a, b, e.tape) == NULL, "no room for the output grad returns NULL");
        CHECK(arena_save(e.arena) == before && e.tape->count == 0, "no room (grad): everything is released");
        env_free(&e);
    }

    // ------------------------------------------------------------------
    SECTION("backward: hand-computed gradients");
    {
        float ones4[4] = { 1, 1, 1, 1 };

        { // add
            Env e = env_new(KiB(64), 8);
            float av[4] = { 1, 2, 3, 4 };
            Node *x = make_node(e.arena, 2, 2, 1, av), *y = make_node(e.arena, 2, 2, 1, av);
            node_add(e.arena, x, y, e.tape);
            node_backward(e.arena, e.tape);
            CHECK_MAT("add: grad of first input", x->grad, ones4, 2, 2);
            CHECK_MAT("add: grad of second input", y->grad, ones4, 2, 2);
            env_free(&e);
        }
        { // sub
            Env e = env_new(KiB(64), 8);
            float av[4] = { 1, 2, 3, 4 }, minus[4] = { -1, -1, -1, -1 };
            Node *x = make_node(e.arena, 2, 2, 1, av), *y = make_node(e.arena, 2, 2, 1, av);
            node_sub(e.arena, x, y, e.tape);
            node_backward(e.arena, e.tape);
            CHECK_MAT("sub: grad of first input is +1", x->grad, ones4, 2, 2);
            CHECK_MAT("sub: grad of second input is -1", y->grad, minus, 2, 2);
            env_free(&e);
        }
        { // add_rows
            Env e = env_new(KiB(64), 8);
            float xv[6] = { 1,2, 3,4, 5,6 }, bv[2] = { 0.5f, -0.5f };
            Node *x = make_node(e.arena, 3, 2, 1, xv), *b = make_node(e.arena, 1, 2, 1, bv);
            node_add_rows(e.arena, x, b, e.tape);
            node_backward(e.arena, e.tape);
            float ones6[6] = { 1,1,1,1,1,1 }, bias_want[2] = { 3, 3 };
            CHECK_MAT("add_rows: grad of the activation passes through", x->grad, ones6, 3, 2);
            CHECK_MAT("add_rows: bias grad is the sum over the 3 rows", b->grad, bias_want, 1, 2);
            env_free(&e);
        }
        { // mul 2x2
            Env e = env_new(KiB(64), 8);
            float av[4] = { 1, 2, 3, 4 }, bv[4] = { 5, 6, 7, 8 };
            Node *A = make_node(e.arena, 2, 2, 1, av), *B = make_node(e.arena, 2, 2, 1, bv);
            node_mul(e.arena, A, B, e.tape);
            node_backward(e.arena, e.tape);
            float gA[4] = { 11, 15, 11, 15 }, gB[4] = { 4, 4, 6, 6 };
            CHECK_MAT("mul 2x2: dA = G * B^T", A->grad, gA, 2, 2);
            CHECK_MAT("mul 2x2: dB = A^T * G", B->grad, gB, 2, 2);
            env_free(&e);
        }
        { // mul non-square
            Env e = env_new(KiB(64), 8);
            float av[2] = { 2, 3 }, bv[6] = { 1,2,3, 4,5,6 };
            Node *A = make_node(e.arena, 1, 2, 1, av), *B = make_node(e.arena, 2, 3, 1, bv);
            node_mul(e.arena, A, B, e.tape);
            node_backward(e.arena, e.tape);
            float gA[2] = { 6, 15 }, gB[6] = { 2,2,2, 3,3,3 };
            CHECK_MAT("mul 1x2 * 2x3: dA has A's shape", A->grad, gA, 1, 2);
            CHECK_MAT("mul 1x2 * 2x3: dB has B's shape", B->grad, gB, 2, 3);
            env_free(&e);
        }
        { // mul with only one differentiable parent
            Env e = env_new(KiB(64), 8);
            float av[4] = { 1, 2, 3, 4 }, bv[4] = { 5, 6, 7, 8 };
            Node *A = make_node(e.arena, 2, 2, 1, av), *B = make_node(e.arena, 2, 2, 0, bv);
            node_mul(e.arena, A, B, e.tape);
            node_backward(e.arena, e.tape);
            float gA[4] = { 11, 15, 11, 15 };
            CHECK_MAT("mul, only A needs grad: dA is correct", A->grad, gA, 2, 2);
            CHECK(B->grad == NULL, "mul, only A needs grad: B stays without grad");
            env_free(&e);
        }
        { // relu
            Env e = env_new(KiB(64), 8);
            float xv[4] = { -1, 0, 2, 3 }, want[4] = { 0, 0, 1, 1 };
            Node *x = make_node(e.arena, 1, 4, 1, xv);
            node_relu(e.arena, x, e.tape);
            node_backward(e.arena, e.tape);
            CHECK_MAT("relu: gradient is 0 for x <= 0 and 1 for x > 0", x->grad, want, 1, 4);
            env_free(&e);
        }
        { // tanh
            Env e = env_new(KiB(64), 8);
            float xv[2] = { 0, 1 }, want[2] = { 1.0f, 0.419974f };
            Node *x = make_node(e.arena, 1, 2, 1, xv);
            node_tanh(e.arena, x, e.tape);
            node_backward(e.arena, e.tape);
            CHECK_MAT("tanh: gradient is 1 - tanh(x)^2", x->grad, want, 1, 2);
            env_free(&e);
        }
        { // sigmoid
            Env e = env_new(KiB(64), 8);
            float xv[2] = { 0, 1 }, want[2] = { 0.25f, 0.196612f };
            Node *x = make_node(e.arena, 1, 2, 1, xv);
            node_sigmoid(e.arena, x, e.tape);
            node_backward(e.arena, e.tape);
            CHECK_MAT("sigmoid: gradient is s(x) * (1 - s(x))", x->grad, want, 1, 2);
            env_free(&e);
        }
        { // mse
            Env e = env_new(KiB(64), 8);
            float pv[2] = { 1, 2 }, tv[2] = { 0, 0 }, gp[2] = { 1, 2 }, gt[2] = { -1, -2 };
            Node *p = make_node(e.arena, 1, 2, 1, pv), *t = make_node(e.arena, 1, 2, 1, tv);
            node_mse(e.arena, p, t, e.tape);
            node_backward(e.arena, e.tape);
            CHECK_MAT("mse: grad of pred is 2*(pred - target)/n", p->grad, gp, 1, 2);
            CHECK_MAT("mse: grad of target is the opposite", t->grad, gt, 1, 2);
            env_free(&e);

            e = env_new(KiB(64), 8);
            Node *p2 = make_node(e.arena, 1, 2, 1, pv), *t2 = make_node(e.arena, 1, 2, 0, tv);
            node_mse(e.arena, p2, t2, e.tape);
            node_backward(e.arena, e.tape);
            CHECK_MAT("mse, target without grad: pred grad is correct", p2->grad, gp, 1, 2);
            CHECK(t2->grad == NULL, "mse, target without grad: target stays without grad");
            env_free(&e);

            e = env_new(KiB(64), 8);
            float pv4[4] = { 1, 2, 3, 4 }, tv4[4] = { 0, 0, 0, 0 }, g4[4] = { 0.5f, 1, 1.5f, 2 };
            Node *p4 = make_node(e.arena, 2, 2, 1, pv4), *t4 = make_node(e.arena, 2, 2, 0, tv4);
            node_mse(e.arena, p4, t4, e.tape);
            node_backward(e.arena, e.tape);
            CHECK_MAT("mse over 4 elements: scale is 2/n with n = 4", p4->grad, g4, 2, 2);
            env_free(&e);
        }
        { // same node used twice
            Env e = env_new(KiB(64), 8);
            float three[1] = { 3 }, two_want[1] = { 2 }, six_want[1] = { 6 };
            Node *x = make_node(e.arena, 1, 1, 1, three);
            node_add(e.arena, x, x, e.tape);
            node_backward(e.arena, e.tape);
            CHECK_MAT("x + x: gradients from both uses accumulate (2)", x->grad, two_want, 1, 1);
            env_free(&e);

            e = env_new(KiB(64), 8);
            Node *y = make_node(e.arena, 1, 1, 1, three);
            node_mul(e.arena, y, y, e.tape);
            node_backward(e.arena, e.tape);
            CHECK_MAT("x * x at x = 3: gradient is 2x = 6", y->grad, six_want, 1, 1);
            env_free(&e);
        }
    }

    // ------------------------------------------------------------------
    SECTION("backward: guards and bookkeeping");
    {
        Env e = env_new(KiB(64), 8);
        CHECK_SURVIVES("node_backward(NULL tape) does not crash", node_backward(e.arena, NULL));
        CHECK_SURVIVES("node_backward on an empty tape does not crash", node_backward(e.arena, e.tape));
        env_free(&e);

        e = env_new(KiB(64), 8);
        float v[4] = { 1, 2, 3, 4 };
        Node *p = make_node(e.arena, 2, 2, 1, v);               // has grad but is not part of the loss
        Node *a = make_node(e.arena, 2, 2, 0, v), *b = make_node(e.arena, 2, 2, 0, v);
        Node *loss = node_mse(e.arena, a, b, e.tape);
        CHECK(loss->require_grad == 0 && loss->grad == NULL, "loss built from constants has no grad");
        node_backward(e.arena, e.tape);
        CHECK(all_zero(p->grad), "backward with a grad-less loss is a no-op");
        env_free(&e);

        e = env_new(KiB(64), 8);
        float av[4] = { 1, 2, 3, 4 }, bv[4] = { 5, 6, 7, 8 };
        Node *A = make_node(e.arena, 2, 2, 1, av), *B = make_node(e.arena, 2, 2, 1, bv);
        node_mul(e.arena, A, B, e.tape);
        size_t before = arena_save(e.arena);
        node_backward(e.arena, e.tape);
        CHECK(arena_save(e.arena) == before, "backward through mul releases its temporary matrices");
        env_free(&e);

        e = env_new(KiB(64), 8);
        Node *x = make_node(e.arena, 2, 2, 1, av), *y = make_node(e.arena, 2, 2, 1, av);
        node_add(e.arena, x, y, e.tape);
        node_backward(e.arena, e.tape);
        node_backward(e.arena, e.tape);
        float twice[4] = { 2, 2, 2, 2 };
        CHECK_MAT("calling backward twice accumulates (parameters need a zero_grad)", x->grad, twice, 2, 2);
        env_free(&e);
    }

    // ------------------------------------------------------------------
    SECTION("numerical gradient check (batch of 3)");
    {
        Net n;
        n.arena = arena_create(MiB(1));
        n.tape  = tape_create(n.arena, 64);
        n.x = node_create(n.arena, 3, 2, 0);
        n.W = node_create(n.arena, 2, 2, 1);
        n.b = node_create(n.arena, 1, 2, 1);
        n.c = node_create(n.arena, 3, 2, 1);
        n.t = node_create(n.arena, 3, 2, 0);

        float xv[6] = {  0.5f, -1.0f,   1.0f, 0.5f,   -0.5f,  0.8f };
        float Wv[4] = {  0.1f, -0.3f,   0.8f, 0.2f };
        float bv[2] = {  0.05f, -0.1f };
        float cv[6] = {  0.1f, 0.2f,   -0.1f, 0.3f,   0.2f, -0.2f };
        float tv[6] = {  0.3f, 0.7f,    0.1f, -0.2f,  0.5f,  0.4f };
        memcpy(n.x->main->data, xv, sizeof(xv)); memcpy(n.W->main->data, Wv, sizeof(Wv));
        memcpy(n.b->main->data, bv, sizeof(bv)); memcpy(n.c->main->data, cv, sizeof(cv));
        memcpy(n.t->main->data, tv, sizeof(tv));
        n.base = arena_save(n.arena);

        const char *labels[] = { "add_rows + tanh", "add_rows + relu", "add_rows + sigmoid", "add_rows + sub + tanh" };
        for (int mode = 0; mode < 4; mode++) {
            mat_fill(n.W->grad, 0.0f); mat_fill(n.b->grad, 0.0f); mat_fill(n.c->grad, 0.0f);
            arena_restore(n.arena, n.base);
            net_forward(&n, mode);
            node_backward(n.arena, n.tape);

            float dW = max_grad_diff(&n, mode, n.W);
            float db = max_grad_diff(&n, mode, n.b);
            snprintf(name, sizeof(name), "%s: dW matches finite differences (max diff %.1e)", labels[mode], dW);
            CHECK(dW < 1e-3f, name);
            snprintf(name, sizeof(name), "%s: db matches finite differences (max diff %.1e)", labels[mode], db);
            CHECK(db < 1e-3f, name);
            if (mode == 3) {
                float dc = max_grad_diff(&n, mode, n.c);
                snprintf(name, sizeof(name), "%s: dc matches finite differences (max diff %.1e)", labels[mode], dc);
                CHECK(dc < 1e-3f, name);
            }
        }
        arena_destroy(n.arena);
    }

    return test_summary();
}

// this code lock like fucking wizardry wth
// gcc -Wall -Wextra -o bin/node_test tests/node_test.c -lm && ./bin/node_test