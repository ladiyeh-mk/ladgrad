#include <stdio.h>
#include <stdlib.h>

typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} Matrix;

Matrix mat_create(const size_t rows, const size_t cols) {
    if (!rows || !cols) return (Matrix) {0};
    Matrix mat = {0};

    mat.rows = rows;
    mat.cols = cols;
    mat.data = malloc(rows * cols * sizeof(float));

    if (!mat.data) return (Matrix) {0};
    return mat;
}

void mat_free(Matrix *mat) {
    if (!mat) return;
    free(mat->data);
    mat->rows = 0;
    mat->cols = 0;
    mat->data = NULL;
}

void mat_print(Matrix *m) {
    if (!m || !m->data) {
        printf("[INFOS] Matrix is NULL or uninitialized.\n");
        return;
    }

    printf("=================================================================================================\n");
    printf("[INFOS]\n\t>> shape: (%zu, %zu)\n\t>> Content:\n", m->rows, m->cols);

    for (size_t r = 0; r < m->rows; r++) {
        for (size_t c = 0; c < m->cols; c++) {
            printf("%s%.2f%s%s", (c == 0) ? "\t\t[" : "", m->data[r * m->cols + c], (c < m->cols - 1) ? " " : "", (c == m->cols - 1) ? "]\n" : "");
        }
    }
    printf("=================================================================================================\n");
}

void mat_show(Matrix *m) {
    if (!m || !m->data) {
        printf("[INFOS] Matrix is NULL or uninitialized.\n");
        return;
    }

    printf("=================================================================================================\n");
    printf("[INFOS]\n\t>> shape: (%zu, %zu)\n\t>> Visual Render:\n", m->rows, m->cols);

    for (size_t r = 0; r < m->rows; r++) {
        printf("\t\t");
        for (size_t c = 0; c < m->cols; c++) {
            float val = m->data[r * m->cols + c];
            
            // Convert float [0.0, 1.0] to an integer [0, 255]
            int intensity = (int)(val * 255.0f);
            if (intensity < 0) intensity = 0;
            if (intensity > 255) intensity = 255;

            // Print two block characters colored with 24-bit RGB TrueColor
            // If it's pure black (0), you can print spaces to keep the terminal clean, 
            // or let it render dark gray.
            if (intensity == 0) {
                printf(" "); // Empty space for background
            } else {
                printf("\x1b[38;2;%d;%d;%dm██\x1b[0m", intensity, intensity, intensity);
            }
        }
        printf("\n");
    }
    printf("=================================================================================================\n");
}

typedef struct {
    Matrix *items;
    size_t count;
    size_t capacity;
} Matrix_Array;

Matrix_Array mat_array_create(const size_t rows, const size_t cols, const size_t capacity) {
    if (!rows || !cols || !capacity) return (Matrix_Array) {0};
    Matrix_Array array = {0};

    array.capacity = capacity;
    array.items    = malloc(capacity * sizeof(Matrix));

    if (!array.items) return (Matrix_Array) {0};
    return array;
}

void mat_array_push(Matrix_Array *array, Matrix *m) {
    if (!array || !array->items || !m || !m->data) return;
    if (array->count == array->capacity) {

        size_t new_cap = (array->capacity) ? array->capacity * 2 : 32;
        Matrix *temp   = realloc(array->items, new_cap * sizeof(Matrix));
        if (!temp) return;

        array->items = temp;
        array->capacity = new_cap;

    }
    array->items[array->count++] = *m;
}

void mat_array_free(Matrix_Array *array) {
    if (!array) return;

    for (size_t i = 0; i < array->count; i++) mat_free(&array->items[i]);
    free(array->items);
    array->count = 0;
    array->capacity = 0;
    array->items = NULL;
}

int read_int(void) {
    char buffer[64];

    // Read a full line safely
    if (fgets(buffer, sizeof(buffer), stdin) != NULL) {
        char *endptr;
        long val = strtol(buffer, &endptr, 10);

        // Check if parsing succeeded
        if (endptr != buffer) {
            return (int)val;
        }
    }
    return 0; // Return fallback/error value
}

int main(void) {
    FILE *file = fopen("ressources/train_images.mat", "rb");

    size_t total_images = 60000;
    Matrix *array = malloc(total_images * sizeof(Matrix));

    for (size_t i = 0; i < total_images; i++) {
        array[i] = mat_create(28, 28);

        size_t read_count = fread(array[i].data, sizeof(float), array[i].rows * array[i].cols, file);
        if (read_count != array[i].rows * array[i].cols) {
            fprintf(stderr, "Reached end of file or read error at image %zu\n", i);
            break;
        }
    }

    fclose(file);

    // ===================================================

    // int index = 0;
    // while (1) {
    //     printf("\n>> ");
    //     index = read_int();

    //     if (index == -1) break;

    //     mat_show(&array[index]);
    // }

    // for (size_t i = 0; i < total_images; i++) mat_free(&array[i]);
    return 0;
}

// gcc -o bin/test1 tests/test1.c && ./bin/test1