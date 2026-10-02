#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "runtime/models.h"
extern const int8_t GOLDEN_MLP_DEMO_FULL_IN[8*384],GOLDEN_MLP_DEMO_FULL_OUT[8*4];
extern const int8_t GOLDEN_CNN_DEMO_FULL_IN[8*384],GOLDEN_CNN_DEMO_FULL_OUT[8*4];
static int run(const tinyml_model_t*m,const int8_t*ins,const int8_t*expected,size_t n,size_t ni,size_t no){uint8_t opt_arena[3072],ref_arena[3072];const int8_t*out,*ref;size_t i,j;for(i=0;i<n;i++){if(tinyml_infer(m,ins+i*ni,opt_arena,m->arena_n,&out)||tinyml_infer_reference(m,ins+i*ni,ref_arena,m->arena_n,&ref)){printf("%s sample %lu: runtime error\n",m->name,(unsigned long)i);return 1;}for(j=0;j<no;j++){if(out[j]!=ref[j]){printf("%s sample %lu index %lu reference %d optimized %d\n",m->name,(unsigned long)i,(unsigned long)j,ref[j],out[j]);return 1;}if(out[j]!=expected[i*no+j]){printf("%s sample %lu index %lu expected %d got %d\n",m->name,(unsigned long)i,(unsigned long)j,expected[i*no+j],out[j]);return 1;}}}printf("%s full-model vectors: PASS (%lu samples, optimized/reference/golden exact)\n",m->name,(unsigned long)n);return 0;}
int main(void){int e=0;e|=run(&tinyml_mlp_demo,GOLDEN_MLP_DEMO_FULL_IN,GOLDEN_MLP_DEMO_FULL_OUT,8,384,4);e|=run(&tinyml_cnn_demo,GOLDEN_CNN_DEMO_FULL_IN,GOLDEN_CNN_DEMO_FULL_OUT,8,384,4);return e;}
