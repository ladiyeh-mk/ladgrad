#include <stdio.h>

#define ARENA_IMPLEMENTATION
#include "../include/arena.h"

#define MATRIX_IMPLEMENTATION
#include "../include/matrix.h"


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
    Mem_Arena *arena = arena_create(MiB(200));

    size_t img_row = 28;
    size_t img_col = 28;
    size_t total_images = 60000;

    Matrix *train_images = mat_create(arena, total_images, img_row * img_col);
    if (!train_images) { printf("Error: train images mat didn't make it T_T\n"); return 1; }

    FILE *train_images_file = fopen("ressources/train_images.mat", "rb");

    size_t total_elements = total_images * img_row * img_col;
    size_t read_count = fread(train_images->data, sizeof(float), total_elements, train_images_file);

    if (read_count != total_elements) { fprintf(stderr, "Error: Expected to read %zu floats, but got %zu.\n", total_elements, read_count); }
    else { printf("Successfully loaded all 60,000 images into memory!\n"); }

    fclose(train_images_file);

    size_t total_labels = 60000;
    Matrix *train_labels = mat_create(arena, total_labels, 1);

    FILE *train_labels_file = fopen("ressources/train_labels.mat", "rb");

    size_t read_count_labels = fread(train_labels->data, sizeof(float), total_labels, train_labels_file);
    if (read_count_labels != total_labels) { fprintf(stderr, "Error: Expected to read %zu floats, but got %zu.\n", total_labels, read_count_labels); }
    else { printf("Successfully loaded all 60,000 labels into memory!\n"); }

    fclose(train_labels_file);

    while (1) {
        printf("\n>> ");
        int index = read_int();

        if (index == -1) break;
        if (index < 0 || index >= (int)total_images) continue;

        Matrix mat_img = (Matrix) {
            .rows = img_row,
            .cols = img_col,
            .data = train_images->data + (index * img_row * img_col)
        };
        
        Matrix mat_label = (Matrix) {
            .rows = 1,
            .cols = 1,
            .data = train_labels->data + index
        };

        mat_show(&mat_img);
        mat_print(&mat_label);
    }

    arena_free(arena);
    return 0;
}

// gcc -o bin/data_loading tests/data_loading.c && ./bin/data_loading