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
    if (!file) {
        fprintf(stderr, "Error: Could not open file.\n");
        return 1;
    }

    size_t img_row = 28;
    size_t img_col = 28;
    size_t total_images = 60000;

    Matrix train_images = mat_create(total_images, img_row * img_col);
    if (!train_images.data) {
        fprintf(stderr, "Error: Memory allocation failed for 188MB matrix.\n");
        fclose(file);
        return 1;
    }

    size_t total_elements = total_images * img_row * img_col;
    size_t read_count = fread(train_images.data, sizeof(float), total_elements, file);

    if (read_count != total_elements) {
        fprintf(stderr, "Error: Expected to read %zu floats, but got %zu.\n", total_elements, read_count);
    } else {
        printf("Successfully loaded all 60,000 images into memory!\n");
    }

    fclose(file);
    // ====================================================

    FILE *file_labels = fopen("ressources/train_labels.mat", "rb");
    if (!file_labels) {
        fprintf(stderr, "Error: Could not open file_labels.\n");
        return 1;
    }

    size_t total_labels = 60000;

    Matrix train_labels = mat_create(total_labels, 1);
    if (!train_labels.data) {
        fprintf(stderr, "Error: Memory allocation failed for labels matrix.\n");
        fclose(file_labels);
        return 1;
    }

    size_t read_count_labels = fread(train_labels.data, sizeof(float), total_labels, file_labels);

    if (read_count_labels != total_labels) {
        fprintf(stderr, "Error: Expected to read %zu floats, but got %zu.\n", total_labels, read_count_labels);
    } else {
        printf("Successfully loaded all 60,000 labels into memory!\n");
    }

    fclose(file_labels);

    int index = 0;
    while (1) {
        printf("\n>> ");
        index = read_int();
        if (index == -1) break;

        Matrix mat_img = (Matrix) {
            .rows = img_row,
            .cols = img_col,
            .data = train_images.data + (index * img_row * img_col)
        };

        Matrix mat_label = (Matrix) {
            .rows = 1,
            .cols = 1,
            .data = train_labels.data + index
        };

        mat_show(&mat_img);
        mat_print(&mat_label);
    }

    mat_free(&train_labels);
    mat_free(&train_images);
    return 0;
}

// gcc -o bin/test1 tests/test1.c && ./bin/test1