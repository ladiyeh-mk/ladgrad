#include <stdio.h>
#include <string.h>

#define ARENA_IMPLEMENTATION
#include "../include/arena.h"

#define MATRIX_IMPLEMENTATION
#include "../include/matrix.h"

#define NODE_IMPLEMENTATION
#include "../include/node.h"

#define IMG_SIZE 784
#define CLASSES  10
#define TOTAL_N  60000
#define TRAIN_N  50000      // rows 0..49999 train, 50000..59999 validation
#define VAL_N    10000
#define BATCH    100        // divides both 50000 and 10000
#define HIDDEN   128
#define EPOCHS   5
#define LR       0.5f       // MSE + sigmoid needs a big rate; lower it if the loss blows up

#define CHECK(p) do { if (!(p)) { fprintf(stderr, "failed: " #p "\n"); return 1; } } while (0)

typedef struct {
    Mem_Arena *persist;
    Mem_Arena *scratch;
    Tape *tape;
    Tape *params;
} LG_Model;

typedef struct {
    Node *W1, *b1, *W2, *b2;
} Net;

// ===================================================================
// Data helpers
// ===================================================================

static Node *node_load(Mem_Arena *arena, size_t rows, size_t cols, const char *filename) {
    size_t checkpoint = arena_save(arena);

    Node *node = node_create(arena, rows, cols, 0);
    if (!node) {
        fprintf(stderr, "[LOAD ERROR] arena too small for '%s' (%zu x %zu)\n", filename, rows, cols);
        return NULL;
    }

    FILE *file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "[LOAD ERROR] cannot open '%s'\n", filename);
        arena_restore(arena, checkpoint);
        return NULL;
    }

    size_t total = rows * cols;
    size_t got   = fread(node->main->data, sizeof(float), total, file);
    fclose(file);

    if (got != total) {
        fprintf(stderr, "[LOAD ERROR] '%s': expected %zu floats, got %zu\n", filename, total, got);
        arena_restore(arena, checkpoint);
        return NULL;
    }

    printf("[INFOS] loaded '%s' as (%zu, %zu)\n", filename, rows, cols);
    return node;
}

static Node *node_onehot(Mem_Arena *arena, const Node *labels, size_t classes) {
    size_t rows       = labels->main->rows;
    size_t checkpoint = arena_save(arena);

    Node *node = node_create(arena, rows, classes, 0);
    if (!node) {
        fprintf(stderr, "[ONEHOT ERROR] arena too small for (%zu, %zu)\n", rows, classes);
        return NULL;
    }

    for (size_t i = 0; i < rows; i++) {
        int c = (int) labels->main->data[i];
        if (c < 0 || (size_t) c >= classes) {
            fprintf(stderr, "[ONEHOT ERROR] row %zu has label %d\n", i, c);
            arena_restore(arena, checkpoint);
            return NULL;
        }
        node->main->data[i * classes + (size_t) c] = 1.0f;
    }
    return node;
}

// Prints one image as ASCII with its label (the alignment check).
static void print_digit(const Node *images, const Node *labels, size_t i) {
    const char *shades = " .:-=+*#%@";
    printf("[INFOS] image %zu, label %d\n", i, (int) labels->main->data[i]);
    for (int r = 0; r < 28; r++) {
        for (int c = 0; c < 28; c++) {
            float v = images->main->data[i * IMG_SIZE + r * 28 + c];
            putchar(shades[(int) (v * 9.0f)]);
        }
        putchar('\n');
    }
}

// Copies the rows listed in idx from src into dest (dest has one row per index).
static void fetch_rows(const Node *src, const size_t *idx, Node *dest) {
    size_t cols = src->main->cols;
    if (dest->main->cols != cols) {
        fprintf(stderr, "[FETCH ERROR] cols differ: %zu vs %zu\n", cols, dest->main->cols);
        abort();
    }
    for (size_t i = 0; i < dest->main->rows; i++) {
        memcpy(dest->main->data + i * cols, src->main->data + idx[i] * cols, cols * sizeof(float));
    }
}

// Fisher-Yates
static void shuffle(size_t *a, size_t n) {
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t) rand() % (i + 1);
        size_t t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

// ===================================================================
// Network
// ===================================================================

static int net_init(LG_Model *model, Net *net) {
    net->W1 = node_create(model->persist, IMG_SIZE, HIDDEN, 1);
    net->b1 = node_create(model->persist, 1, HIDDEN, 1);
    net->W2 = node_create(model->persist, HIDDEN, CLASSES, 1);
    net->b2 = node_create(model->persist, 1, CLASSES, 1);
    if (!net->W1 || !net->b1 || !net->W2 || !net->b2) return 0;

    node_init_he(net->W1);      // biases stay at zero
    node_init_he(net->W2);

    node_params_add(model->params, net->W1);
    node_params_add(model->params, net->b1);
    node_params_add(model->params, net->W2);
    node_params_add(model->params, net->b2);
    return 1;
}

// x is (BATCH, 784). Returns the (BATCH, 10) output.
static Node *forward(Mem_Arena *arena, Tape *tape, const Net *net, Node *x) {
    Node *h = node_mul(arena, x, net->W1, tape);
    h       = node_add_rows(arena, h, net->b1, tape);
    h       = node_relu(arena, h, tape);

    Node *o = node_mul(arena, h, net->W2, tape);
    o       = node_add_rows(arena, o, net->b2, tape);
    return node_sigmoid(arena, o, tape);
}

// Fraction of correct predictions on rows start .. start+count-1.
static float accuracy(LG_Model *model, const Net *net, const Node *images, const Node *labels,
                    size_t start, size_t count) {
    size_t correct = 0;

    for (size_t b = 0; b < count / BATCH; b++) {
        size_t idx[BATCH];
        for (size_t i = 0; i < BATCH; i++) idx[i] = start + b * BATCH + i;

        model->tape->count = 0;
        arena_restore(model->scratch, 0);

        Node *x = node_create(model->scratch, BATCH, IMG_SIZE, 0);
        if (!x) { fprintf(stderr, "scratch too small\n"); return 0.0f; }
        fetch_rows(images, idx, x);

        Node *out = forward(model->scratch, model->tape, net, x);
        if (!out) { fprintf(stderr, "forward failed\n"); return 0.0f; }

        for (size_t i = 0; i < BATCH; i++) {
            const float *row = out->main->data + i * CLASSES;
            int best = 0;
            for (int c = 1; c < CLASSES; c++) if (row[c] > row[best]) best = c;
            if (best == (int) labels->main->data[idx[i]]) correct++;
        }
    }
    return (float) correct / (float) ((count / BATCH) * BATCH);
}

// ===================================================================

// Type an index from the test set, see the digit, the prediction and the target.
// Type -1 (or anything out of range) to quit.
static void repl(LG_Model *model, const Net *net, const Node *images, const Node *labels) {
    size_t n = images->main->rows;

    while (1) {
        printf("\nindex (0..%zu, -1 to quit): ", n - 1);
        long i;
        if (scanf("%ld", &i) != 1 || i < 0 || (size_t) i >= n) break;

        model->tape->count = 0;
        arena_restore(model->scratch, 0);

        Node *x = node_create(model->scratch, 1, IMG_SIZE, 0);
        if (!x) { fprintf(stderr, "scratch too small\n"); break; }
        size_t idx[1] = { (size_t) i };
        fetch_rows(images, idx, x);

        Node *out = forward(model->scratch, model->tape, net, x);
        if (!out) { fprintf(stderr, "forward failed\n"); break; }

        int best = 0;
        for (int c = 1; c < CLASSES; c++) if (out->main->data[c] > out->main->data[best]) best = c;

        print_digit(images, labels, (size_t) i);
        printf("scores:");
        for (int c = 0; c < CLASSES; c++) printf("%s%.2f%s", (c == 0) ? "[" : "", out->main->data[c], (c < CLASSES - 1) ? ", " : "");
        printf("\npredicted: %d   target: %d   %s\n", best, (int) labels->main->data[i],
                best == (int) labels->main->data[i] ? "OK" : "WRONG");
    }
}

// ===================================================================

int main(void) {
    srand(42);

    LG_Model model = {0};
    model.persist = arena_create(MiB(400));
    model.scratch = arena_create(MiB(400));
    CHECK(model.persist && model.scratch);

    // Both tapes live in persist so rewinding scratch never touches them.
    model.tape   = tape_create(model.persist, 16);
    model.params = tape_create(model.persist, 16);
    CHECK(model.tape && model.params);

    // ---- data ----
    Node *images  = node_load(model.persist, TOTAL_N, IMG_SIZE, "ressources/train_images.mat");
    Node *labels  = node_load(model.persist, TOTAL_N, 1,        "ressources/train_labels.mat");
    CHECK(images && labels);
    Node *targets = node_onehot(model.persist, labels, CLASSES);
    CHECK(targets);

    // alignment check: look at these and confirm the picture matches the label
    // for (size_t i = 0; i < 3; i++) print_digit(images, labels, i);

    // ---- network ----
    Net net;
    CHECK(net_init(&model, &net));

    size_t *order = arena_push_array(model.persist, size_t, TRAIN_N);
    CHECK(order);
    for (size_t i = 0; i < TRAIN_N; i++) order[i] = i;

    printf("[INFOS] before training: val accuracy = %.2f%% (expect ~10%%)\n",
           100.0f * accuracy(&model, &net, images, labels, TRAIN_N, VAL_N));

    // ---- training ----
    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        shuffle(order, TRAIN_N);
        float loss_sum = 0.0f;

        for (size_t b = 0; b < TRAIN_N / BATCH; b++) {
            const size_t *idx = order + b * BATCH;

            model.tape->count = 0;
            arena_restore(model.scratch, 0);

            Node *x = node_create(model.scratch, BATCH, IMG_SIZE, 0);
            Node *y = node_create(model.scratch, BATCH, CLASSES, 0);
            CHECK(x && y);
            fetch_rows(images,  idx, x);
            fetch_rows(targets, idx, y);

            Node *out  = forward(model.scratch, model.tape, &net, x);
            CHECK(out);
            Node *loss = node_mse(model.scratch, out, y, model.tape);
            CHECK(loss);

            node_zero_grad(model.params);
            node_backward(model.scratch, model.tape);
            node_optimizer(model.params, LR);

            loss_sum += loss->main->data[0];
        }

        printf("[INFOS] epoch %d: loss = %.5f, val accuracy = %.2f%%\n", epoch,
                loss_sum / (float) (TRAIN_N / BATCH),
               100.0f * accuracy(&model, &net, images, labels, TRAIN_N, VAL_N));
    }

    Node *test_images = node_load(model.persist, VAL_N, IMG_SIZE, "ressources/test_images.mat");
    Node *test_labels = node_load(model.persist, VAL_N, 1,        "ressources/test_labels.mat");
    CHECK(test_images && test_labels);

    printf("[INFOS] final TEST accuracy = %.2f%%\n",
        100.0f * accuracy(&model, &net, test_images, test_labels, 0, VAL_N));

    repl(&model, &net, test_images, test_labels);

    arena_destroy(model.persist);
    arena_destroy(model.scratch);
    return 0;
}