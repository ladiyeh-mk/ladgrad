# ladgrad

A tiny autograd engine in C, built from scratch with no dependencies beyond the standard library. It has an arena allocator, a matrix library, and tape-based reverse-mode backpropagation. As a demo, it trains a 784-128-10 neural network on MNIST.

**Result: 92.14% test accuracy** after 5 epochs of plain SGD (MSE loss, sigmoid output).

```
[INFOS] before training: val accuracy = 14.54%
[INFOS] epoch 1: loss = 0.04779, val accuracy = 86.67%
[INFOS] epoch 2: loss = 0.02475, val accuracy = 88.99%
[INFOS] epoch 3: loss = 0.02041, val accuracy = 90.25%
[INFOS] epoch 4: loss = 0.01809, val accuracy = 90.94%
[INFOS] epoch 5: loss = 0.01654, val accuracy = 91.58%
[INFOS] final TEST accuracy = 92.14%
```

Validation uses rows 50,000 to 59,999 of the training set. The test set is only used once, at the end.

## Quick start

```sh
# 1. Get the data (once)
pip install tensorflow tensorflow-datasets numpy
python utils/main.py

# 2. Build and run
gcc -o bin/ladgrad src/ladgrad.c -lm
./bin/ladgrad
```

After training, the program starts a small REPL: type a test-set index and it prints the digit as ASCII art, the 10 output scores, the predicted class and the target. Type `-1` to quit.

## Project layout

| Path | Purpose |
|---|---|
| `include/arena.h` | Arena (bump) allocator with save/restore |
| `include/matrix.h` | Row-major `float` matrices and operations |
| `include/node.h` | Autograd nodes, tape, backward pass, SGD, initializers |
| `include/test_util.h` | Helpers for the unit tests |
| `src/ladgrad.c` | MNIST data loading, training loop, evaluation, REPL |
| `tests/` | Unit tests for arena, matrix, node and data loading |
| `utils/main.py` | Downloads MNIST and writes the raw data files |
| `ressources/` | Data files (generated, git-ignored) |

All three libraries are single-header: define `ARENA_IMPLEMENTATION`, `MATRIX_IMPLEMENTATION` and `NODE_IMPLEMENTATION` in exactly one `.c` file before including them.

## How it works

### Memory: two arenas

There is no `malloc` in the hot path. Everything lives in two arenas:

- **persist**: the dataset, the weights, the tapes. Allocated once, lives for the whole run.
- **scratch**: every temporary node and matrix of one training step. At the start of each step it is rewound with `arena_restore(scratch, 0)`, so memory use stays flat.

`arena_save` / `arena_restore` also let the backward pass allocate temporary transposes and free them right away.

### Autograd: the tape

Every operation creates a `Node` holding its result (`main`), its gradient (`grad`), the op type and up to two parents. Each op is also appended to a `Tape`. Backpropagation walks the tape in reverse, and each op adds its contribution to its parents' gradients.

Supported ops: `add`, `add_rows` (bias broadcast), `sub`, `mul` (matrix product), `relu`, `tanh`, `sigmoid`, and `mse`.

Weights are not recorded on the tape. They are registered once in a separate **params list**, so zeroing gradients and the SGD update touch each weight exactly once.

### One training step

```c
model.tape->count = 0;                 // forget last step's graph
arena_restore(model.scratch, 0);       // free last step's temporaries

// fetch a batch, build the graph
Node *out  = forward(model.scratch, model.tape, &net, x);
Node *loss = node_mse(model.scratch, out, y, model.tape);

node_zero_grad(model.params);          // weights persist, so zero by hand
node_backward(model.scratch, model.tape);
node_optimizer(model.params, LR);      // w -= lr * grad
```

### The network

```
input (100 x 784) -> W1,b1 -> ReLU -> W2,b2 -> sigmoid -> MSE vs one-hot (100 x 10)
```

Weights use He uniform initialization and biases start at zero. Batches are drawn from a shuffled index array (Fisher-Yates), reshuffled each epoch.

## Data format

`utils/main.py` writes four files to `ressources/` as raw row-major `float32`, with no header:

| File | Shape |
|---|---|
| `train_images.mat` | 60000 x 784, pixels scaled to 0..1 |
| `train_labels.mat` | 60000 x 1, class index stored as float |
| `test_images.mat` | 10000 x 784 |
| `test_labels.mat` | 10000 x 1 |

Labels are turned into one-hot targets at load time.

## Tests

Each test file builds into its own binary, for example:

```sh
gcc -o bin/node_test tests/node_test.c -lm && ./bin/node_test
```

## Limitations and ideas

- Only MSE loss, so there is no softmax or cross-entropy yet. Softmax with cross-entropy would train faster and score higher on MNIST.
- Plain SGD, with no momentum or Adam.
- Single-threaded, naive `ikj` matrix multiplication.
- The tape capacity is fixed at creation (16 nodes here), so deeper networks need a larger value.
- Hyperparameters (`LR`, `BATCH`, `HIDDEN`, `EPOCHS`) are `#define`s at the top of `src/ladgrad.c`.

## License

MIT, see [LICENSE](LICENSE).