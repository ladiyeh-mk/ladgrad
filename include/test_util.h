#ifndef TEST_UTIL_H
#define TEST_UTIL_H

// Shared helpers for the ladgrad test suites.
// Needs a POSIX system (fork/waitpid) so that "this call must abort()" can be tested.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>

#define TEST_EPS 1e-4f

static int tests_passed = 0;
static int tests_failed = 0;

#define INFO(...)    do { printf("[INFOS] "); printf(__VA_ARGS__); printf("\n"); } while (0)
#define SECTION(...) do { printf("\n[INFOS] ---- "); printf(__VA_ARGS__); printf(" ----\n"); } while (0)

static inline void report(int ok, const char *name, const char *detail) {
    if (ok) { tests_passed++; printf("[INFOS] PASS  %s\n", name); }
    else    { tests_failed++; printf("[INFOS] FAIL  %s  (%s)\n", name, detail ? detail : ""); }
}

static inline int test_summary(void) {
    printf("\n[INFOS] ==== %d passed, %d failed ====\n", tests_passed, tests_failed);
    return tests_failed ? 1 : 0;
}

// CHECK(condition, "name")
#define CHECK(cond, name) report((cond) ? 1 : 0, (name), #cond)

// CHECK_NEAR(got, want, eps, "name")
#define CHECK_NEAR(got, want, eps, name) do {                                          \
    float _g = (float) (got), _w = (float) (want);                                     \
    char _d[96];                                                                       \
    snprintf(_d, sizeof(_d), "got %.6f, wanted %.6f", _g, _w);                         \
    report(fabsf(_g - _w) <= (eps), (name), _d);                                       \
} while (0)

// Compares a rows x cols block of floats. `got` may be NULL (reported as a failure).
static inline void check_floats(const char *name, size_t got_rows, size_t got_cols,
                                size_t want_rows, size_t want_cols,
                                const float *got, const float *want) {
    char detail[128];
    if (!got) { report(0, name, "matrix is NULL"); return; }
    if (got_rows != want_rows || got_cols != want_cols) {
        snprintf(detail, sizeof(detail), "shape (%zu, %zu), wanted (%zu, %zu)",
                 got_rows, got_cols, want_rows, want_cols);
        report(0, name, detail);
        return;
    }
    for (size_t i = 0; i < want_rows * want_cols; i++) {
        if (!(fabsf(got[i] - want[i]) <= TEST_EPS)) {
            snprintf(detail, sizeof(detail), "index %zu: got %.5f, wanted %.5f", i, got[i], want[i]);
            report(0, name, detail);
            return;
        }
    }
    report(1, name, NULL);
}

// CHECK_MAT("name", matrix_pointer, want_array, rows, cols)
#define CHECK_MAT(name, m, want, r, c) \
    check_floats((name), (m) ? (m)->rows : 0, (m) ? (m)->cols : 0, (r), (c), (m) ? (m)->data : NULL, (want))

// Runs the statement(s) in a child process, with stdout/stderr silenced and core dumps off.
#define RUN_IN_CHILD(status_var, ...)                                      \
    do {                                                                   \
        fflush(stdout); fflush(stderr);                                    \
        pid_t _pid = fork();                                               \
        if (_pid == 0) {                                                   \
            struct rlimit _rl = {0, 0};                                    \
            setrlimit(RLIMIT_CORE, &_rl);                                  \
            if (!freopen("/dev/null", "w", stdout)) _exit(3);              \
            if (!freopen("/dev/null", "w", stderr)) _exit(3);              \
            __VA_ARGS__;                                                   \
            _exit(0);                                                      \
        }                                                                  \
        (status_var) = 0;                                                  \
        waitpid(_pid, &(status_var), 0);                                   \
    } while (0)

// The statement must die with SIGABRT (the fail-fast behaviour).
#define CHECK_ABORTS(name, ...) do {                                                   \
    int _st;                                                                           \
    RUN_IN_CHILD(_st, __VA_ARGS__);                                                    \
    report(WIFSIGNALED(_st) && WTERMSIG(_st) == SIGABRT, (name), "expected abort()");  \
} while (0)

// The statement must return normally (no crash, no abort).
#define CHECK_SURVIVES(name, ...) do {                                                 \
    int _st;                                                                           \
    RUN_IN_CHILD(_st, __VA_ARGS__);                                                    \
    report(WIFEXITED(_st) && WEXITSTATUS(_st) == 0, (name), "expected a normal return"); \
} while (0)

#endif // TEST_UTIL_H
// this code came from claude code