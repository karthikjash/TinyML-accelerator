#include "arena.h"
#include <stdint.h>
void tinyml_arena_init(tinyml_arena_t*a,void*m,size_t n){a->base=(unsigned char*)m;a->capacity=n;a->used=0;}
void *tinyml_arena_alloc(tinyml_arena_t*a,size_t n,size_t align){size_t p,pad;if(!a||!a->base||!align||(align&(align-1)))return 0;p=(size_t)(uintptr_t)(a->base+a->used);pad=(align-(p&(align-1)))&(align-1);if(pad>a->capacity-a->used||n>a->capacity-a->used-pad)return 0;a->used+=pad+n;return a->base+a->used-n;}
