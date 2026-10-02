#ifndef TINYML_ARENA_H
#define TINYML_ARENA_H
#include <stddef.h>
typedef struct { unsigned char *base; size_t capacity, used; } tinyml_arena_t;
void tinyml_arena_init(tinyml_arena_t *a, void *memory, size_t bytes);
void *tinyml_arena_alloc(tinyml_arena_t *a, size_t bytes, size_t alignment);
#endif
