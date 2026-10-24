#ifndef NODE_H
#define NODE_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

#include "arena.h"
#include "matrix.h"

typedef enum {OP_NONE, OP_ADD, OP_ADD_ROWS, OP_SUB, OP_MUL, OP_RELU, OP_TANH, OP_SIGMOID, OP_MSE} Ops;
typedef struct Node Node;
typedef struct Node {
    int require_grad;
    Ops op;

    Matrix *main;
    Matrix *grad;

    Node *parents[2];
} Node;

typedef struct {
    Node **nodes;
    size_t count;
    size_t capacity;
} Tape;

// Function Declarations
Tape *tape_create(Mem_Arena *arena, const size_t capacity);
Node *node_create(Mem_Arena *arena, const size_t rows, const size_t cols, int require_grad);

Node *node_add(Mem_Arena *arena, Node *a, Node *b, Tape *tape);
Node *node_add_rows(Mem_Arena *arena, Node *a, Node *bias, Tape *tape);
Node *node_sub(Mem_Arena *arena, Node *a, Node *b, Tape *tape);
Node *node_mul(Mem_Arena *arena, Node *a, Node *b, Tape *tape);
Node *node_relu(Mem_Arena *arena, Node *a, Tape *tape);
Node *node_tanh(Mem_Arena *arena, Node *a, Tape *tape);
Node *node_sigmoid(Mem_Arena *arena, Node *a, Tape *tape);
Node *node_mse(Mem_Arena *arena, Node *pred, Node *target, Tape *tape);

void node_backward(Mem_Arena *arena, Tape *tape);
void node_params_add(Tape *params, Node *n);
void node_optimizer(Tape *tape, float lr);
void node_zero_grad(Tape *tape);
void node_init_uniform(Node *n, float limit);
void node_init_he(Node *n);

#ifdef NODE_IMPLEMENTATION

Tape *tape_create(Mem_Arena *arena, const size_t capacity) {
    size_t checkpoint = arena_save(arena);
    Tape *tape = arena_push(arena, sizeof(Tape), _Alignof(Tape));
    if (!tape) return NULL;

    tape->count    = 0;
    tape->capacity = capacity;
    tape->nodes    = arena_push(arena, sizeof(Node *) * capacity, _Alignof(Node *));

    if (!tape->nodes) { arena_restore(arena, checkpoint); return NULL; }
    return tape;
}

Node *node_create(Mem_Arena *arena, const size_t rows, const size_t cols, int require_grad) {
    if (!rows || !cols) return NULL;

    size_t checkpoint = arena_save(arena);
    Node *node = (Node *) arena_push(arena, sizeof(Node), _Alignof(Node));
    if (!node) return NULL;

    node->main = mat_create(arena, rows, cols);
    if (!node->main) { arena_restore(arena, checkpoint); return NULL; }

    if (require_grad) {
        node->grad = mat_create(arena, rows, cols);
        if (!node->grad) { arena_restore(arena, checkpoint); return NULL; }
    } else { node->grad = NULL; }

    node->require_grad = require_grad;
    node->op           = OP_NONE;
    node->parents[0]   = NULL;
    node->parents[1]   = NULL;

    return node;
}

static Node *operations_helper(Mem_Arena *arena, Node *a, Node *b, Tape *tape, Matrix *res, int require_grad, Ops op, size_t arena_checkpoint) {
    if (tape->count >= tape->capacity) {
        fprintf(stderr, "Error: Tape capacity exceeded! (%zu / %zu)\n", tape->count, tape->capacity);
        arena_restore(arena, arena_checkpoint);
        return NULL;
    }

    Node *out = (Node *) arena_push(arena, sizeof(Node), _Alignof(Node));
    if (!out) { arena_restore(arena, arena_checkpoint); return NULL; }

    out->require_grad = require_grad;
    out->op           = op;
    out->main         = res;
    out->parents[0]   = a;
    out->parents[1]   = b;

    if (require_grad) {
        out->grad = mat_create(arena, res->rows, res->cols);
        if (!out->grad) { arena_restore(arena, arena_checkpoint); return NULL; }
    } else { out->grad = NULL; }
    tape->nodes[tape->count++] = out;

    return out;
}

Node *node_add(Mem_Arena *arena, Node *a, Node *b, Tape *tape) {
    size_t checkpoint = arena_save(arena);

    Matrix *result = mat_create(arena, a->main->rows, a->main->cols);
    if (!result) { arena_restore(arena, checkpoint); return NULL; }
    mat_add(a->main, b->main, result);

    int require_grad = (a->require_grad || b->require_grad);
    return operations_helper(arena, a, b, tape, result, require_grad, OP_ADD, checkpoint);
}

Node *node_add_rows(Mem_Arena *arena, Node *a, Node *bias, Tape *tape) {
    if (bias->main->rows != 1 || bias->main->cols != a->main->cols) {
        fprintf(stderr, "[NODE ERROR] node_add_rows: bias is (%zu, %zu) but must be (1, %zu)\n",
                bias->main->rows, bias->main->cols, a->main->cols);
        abort();
    }

    size_t checkpoint = arena_save(arena);

    Matrix *result = mat_create(arena, a->main->rows, a->main->cols);
    if (!result) { arena_restore(arena, checkpoint); return NULL; }
    mat_add_rows(a->main, bias->main, result);

    int require_grad = (a->require_grad || bias->require_grad);
    return operations_helper(arena, a, bias, tape, result, require_grad, OP_ADD_ROWS, checkpoint);
}

Node *node_sub(Mem_Arena *arena, Node *a, Node *b, Tape *tape) {
    size_t checkpoint = arena_save(arena);

    Matrix *result = mat_create(arena, a->main->rows, a->main->cols);
    if (!result) { arena_restore(arena, checkpoint); return NULL; }
    mat_sub(a->main, b->main, result);

    int require_grad = (a->require_grad || b->require_grad);
    return operations_helper(arena, a, b, tape, result, require_grad, OP_SUB, checkpoint);
}

Node *node_mul(Mem_Arena *arena, Node *a, Node *b, Tape *tape) {
    size_t checkpoint = arena_save(arena);

    Matrix *result = mat_create(arena, a->main->rows, b->main->cols);
    if (!result) { arena_restore(arena, checkpoint); return NULL; }
    mat_mul(a->main, b->main, result);

    int require_grad = (a->require_grad || b->require_grad);
    return operations_helper(arena, a, b, tape, result, require_grad, OP_MUL, checkpoint);
}

Node *node_relu(Mem_Arena *arena, Node *a, Tape *tape) {
    size_t checkpoint = arena_save(arena);

    Matrix *result = mat_create(arena, a->main->rows, a->main->cols);
    if (!result) { arena_restore(arena, checkpoint); return NULL; }

    size_t size = result->rows * result->cols;
    for (size_t i = 0; i < size; i++) result->data[i] = MAX(0, a->main->data[i]);

    int require_grad = (a->require_grad);
    return operations_helper(arena, a, NULL, tape, result, require_grad, OP_RELU, checkpoint);
}

Node *node_tanh(Mem_Arena *arena, Node *a, Tape *tape) {
    size_t checkpoint = arena_save(arena);

    Matrix *result = mat_create(arena, a->main->rows, a->main->cols);
    if (!result) { arena_restore(arena, checkpoint); return NULL; }

    size_t size = result->rows * result->cols;
    for (size_t i = 0; i < size; i++) result->data[i] = tanhf(a->main->data[i]);

    int require_grad = (a->require_grad);
    return operations_helper(arena, a, NULL, tape, result, require_grad, OP_TANH, checkpoint);
}

Node *node_sigmoid(Mem_Arena *arena, Node *a, Tape *tape) {
    size_t checkpoint = arena_save(arena);

    Matrix *result = mat_create(arena, a->main->rows, a->main->cols);
    if (!result) { arena_restore(arena, checkpoint); return NULL; }

    size_t size = result->rows * result->cols;
    for (size_t i = 0; i < size; i++) result->data[i] = 1.0f / (1.0f + expf(-a->main->data[i]));

    int require_grad = (a->require_grad);
    return operations_helper(arena, a, NULL, tape, result, require_grad, OP_SIGMOID, checkpoint);
}

Node *node_mse(Mem_Arena *arena, Node *pred, Node *target, Tape *tape) {
    if (pred->main->rows != target->main->rows || pred->main->cols != target->main->cols) {
        fprintf(stderr, "[NODE ERROR] node_mse: pred is (%zu, %zu) but target is (%zu, %zu)\n",
                pred->main->rows, pred->main->cols, target->main->rows, target->main->cols);
        abort();
    }

    size_t checkpoint = arena_save(arena);

    Matrix *result = mat_create(arena, 1, 1);
    if (!result) { arena_restore(arena, checkpoint); return NULL; }

    float sum_sq_diff = 0;
    size_t size = pred->main->rows * pred->main->cols;
    for (size_t i = 0; i < size; i++) { float diff = pred->main->data[i] - target->main->data[i]; sum_sq_diff += diff * diff; }

    result->data[0] = sum_sq_diff / (float) size;
    int require_grad = (pred->require_grad || target->require_grad);
    return operations_helper(arena, pred, target, tape, result, require_grad, OP_MSE, checkpoint);
}

void node_backward(Mem_Arena *arena, Tape *tape) {
    if (!tape || !tape->nodes || tape->count == 0) return;

    Node *loss = tape->nodes[tape->count - 1];
    if (!loss->require_grad || !loss->grad) return;
    mat_fill(loss->grad, 1.0f);

    for (int i = (int) tape->count - 1; i >= 0; i--) {

        Node *current_node = tape->nodes[i];
        if (!current_node->require_grad) continue;

        Node *parent_one = current_node->parents[0];
        Node *parent_two = current_node->parents[1];

        switch (current_node->op) {

            case OP_NONE:
                break;

            case OP_ADD: {
                if (parent_one && parent_one->require_grad) { mat_add_inplace(current_node->grad, parent_one->grad); }
                if (parent_two && parent_two->require_grad) { mat_add_inplace(current_node->grad, parent_two->grad); }
                break;
            }

            case OP_ADD_ROWS: {
                if (parent_one && parent_one->require_grad) { mat_add_inplace(current_node->grad, parent_one->grad); }
                if (parent_two && parent_two->require_grad) { mat_sum_rows_inplace(current_node->grad, parent_two->grad); }
                break;
            }

            case OP_SUB: {
                if (parent_one && parent_one->require_grad) { mat_add_inplace(current_node->grad, parent_one->grad); }
                if (parent_two && parent_two->require_grad) { mat_sub_inplace(current_node->grad, parent_two->grad); }
                break;
            }

            case OP_MUL: {

                if (parent_one && parent_one->require_grad) {
                    size_t checkpoint = arena_save(arena);

                    Matrix *parent_two_transposed = mat_create(arena, parent_two->main->cols, parent_two->main->rows);
                    mat_transpose(parent_two->main, parent_two_transposed);

                    Matrix *out_one = mat_create(arena, current_node->grad->rows, parent_two_transposed->cols);
                    mat_mul(current_node->grad, parent_two_transposed, out_one);
                    mat_add_inplace(out_one, parent_one->grad);

                    arena_restore(arena, checkpoint);
                }

                if (parent_two && parent_two->require_grad) {
                    size_t checkpoint = arena_save(arena);

                    Matrix *parent_one_transposed = mat_create(arena, parent_one->main->cols, parent_one->main->rows);
                    mat_transpose(parent_one->main, parent_one_transposed);

                    Matrix *out_two = mat_create(arena, parent_one_transposed->rows, current_node->grad->cols);
                    mat_mul(parent_one_transposed, current_node->grad, out_two);
                    mat_add_inplace(out_two, parent_two->grad);

                    arena_restore(arena, checkpoint);
                }
                break;
            }

            case OP_RELU: {
                if (parent_one && parent_one->require_grad) {
                    size_t size = current_node->main->rows * current_node->main->cols;
                    for (size_t j = 0; j < size; j++) {
                        if (parent_one->main->data[j] > 0.0f) { parent_one->grad->data[j] += current_node->grad->data[j]; }
                    }
                }
                break;
            }

            case OP_TANH: {
                if (parent_one && parent_one->require_grad) {
                    size_t size = current_node->main->rows * current_node->main->cols;
                    for (size_t j = 0; j < size; j++) {
                        float y = current_node->main->data[j];
                        parent_one->grad->data[j] += current_node->grad->data[j] * (1.0f - y * y);
                    }
                }
                break;
            }

            case OP_SIGMOID: {
                if (parent_one && parent_one->require_grad) {
                    size_t size = current_node->main->rows * current_node->main->cols;
                    for (size_t j = 0; j < size; j++) {
                        float y = current_node->main->data[j];
                        parent_one->grad->data[j] += current_node->grad->data[j] * y * (1.0f - y);
                    }
                }
                break;
            }

            case OP_MSE: {
                size_t size = parent_one->main->rows * parent_one->main->cols;
                float scale = (2.0f * current_node->grad->data[0]) / (float) size;

                if (parent_one && parent_one->require_grad) {
                    for (size_t j = 0; j < size; j++) {
                        float diff = parent_one->main->data[j] - parent_two->main->data[j];
                        parent_one->grad->data[j] += scale * diff;
                    }
                }

                if (parent_two && parent_two->require_grad) {
                    for (size_t j = 0; j < size; j++) {
                        float diff = parent_one->main->data[j] - parent_two->main->data[j];
                        parent_two->grad->data[j] -= scale * diff;
                    }
                }
                break;
            }
        }
    }
}

void node_params_add(Tape *params, Node *n) {
    if (!params || !n) return;
    if (params->count >= params->capacity) {
        fprintf(stderr, "[NODE ERROR] node_params_add: params list full (%zu)\n", params->capacity);
        abort();
    }
    params->nodes[params->count++] = n;
}

// Zeroes every gradient that exists. Nodes without grad (data, targets) are skipped.
void node_zero_grad(Tape *tape) {
    if (!tape || !tape->nodes) return;

    for (size_t i = 0; i < tape->count; i++) {
        Node *n = tape->nodes[i];
        if (n && n->grad) mat_fill(n->grad, 0.0f);
    }
}

// SGD step over the params list: each weight is updated exactly once,
// and its grad is left untouched (zero_grad handles it next step).
void node_optimizer(Tape *params, float lr) {
    if (!params || !params->nodes) return;

    for (size_t i = 0; i < params->count; i++) {
        Node *p = params->nodes[i];
        if (!p || !p->require_grad || !p->grad) continue;

        size_t size = p->main->rows * p->main->cols;
        for (size_t j = 0; j < size; j++) p->main->data[j] -= lr * p->grad->data[j];
    }
}

void node_init_uniform(Node *n, float limit) {
    if (!n || !n->main || !n->main->data) {
        fprintf(stderr, "[NODE ERROR] node_init_uniform: NULL node or matrix\n");
        abort();
    }

    size_t size = n->main->rows * n->main->cols;
    for (size_t i = 0; i < size; i++) {
        float u = (float) rand() / (float) RAND_MAX;      // 0.0 to 1.0
        n->main->data[i] = (2.0f * u - 1.0f) * limit;     // -limit to +limit
    }
}

void node_init_he(Node *n) {
    if (!n || !n->main || !n->main->data) {
        fprintf(stderr, "[NODE ERROR] node_init_he: NULL node or matrix\n");
        abort();
    }

    // weights are (inputs x outputs), so rows = number of inputs to the layer
    float limit = sqrtf(6.0f / (float) n->main->rows);
    node_init_uniform(n, limit);
}

#endif // NODE_IMPLEMENTATION
#endif // NODE_H