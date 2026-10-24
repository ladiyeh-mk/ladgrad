# ladgrad

A tiny autograd engine in C: arena allocator, matrices, and tape-based backpropagation.
It trains a 784-128-10 network on MNIST and reaches about 93% validation accuracy
after 5 epochs (MSE loss, sigmoid output).

## Layout

- `include/arena.h`: arena allocator
- `include/matrix.h`: matrix operations
- `include/node.h`: autograd nodes, tape, backward pass, SGD
- `src/ladgrad.c`: MNIST training and an interactive test REPL
- `tests/`: unit tests
- `utils/main.py`: downloads MNIST and writes the data files

## Data

The data files are not in the repo. Generate them once:

    pip install tensorflow-datasets numpy
    python utils/main.py

This writes `train_images.mat`, `train_labels.mat`, `test_images.mat` and
`test_labels.mat` to `ressources/` as raw row-major float32. Images are scaled
to 0..1, and labels are stored as floats.

## Build and run

    gcc -o bin/ladgrad src/ladgrad.c -lm
    ./bin/ladgrad

It trains, prints the final test accuracy, then lets you type a test index to
see the digit, the predicted class and the target.