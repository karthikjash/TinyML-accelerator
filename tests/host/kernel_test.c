#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "kernels/kernels.h"
#include "memory/arena.h"
static int fails;
#define CHECK(n,x) do{if(!(x)){printf("FAIL %s\n",n);fails++;}else printf("PASS %s\n",n);}while(0)
int main(void){int32_t a[2]={3,-3},acc[3],bias[2]={0,0};int8_t q[2],x[4]={1,2,3,4},w[4]={1,0,0,1},y[4];double lut[256];size_t i;unsigned char mem[16];tinyml_arena_t ar;
 tinyml_requantize(a,2,0.5,0,q);CHECK("requant half-even",q[0]==2&&q[1]==-2);
 tinyml_fc(x,w,bias,2,2,0,acc);tinyml_requantize(acc,2,1.0,0,q);CHECK("FC row-major",q[0]==1&&q[1]==2);
 {int8_t fused[2];tinyml_fc_quant(x,w,bias,2,2,0,1.0,0,0,fused);CHECK("FC fused exact",memcmp(q,fused,2)==0);}
 tinyml_relu(q,2,2);CHECK("ReLU uses quantized zero point",q[0]==2&&q[1]==2);
 tinyml_maxpool1d((int8_t[]){1,7,3,4,5,0,2,8},4,2,2,2,y);CHECK("maxpool time-major",y[0]==3&&y[1]==7&&y[2]==5&&y[3]==8);
 {int8_t cx[3]={1,2,3},cw[3]={1,1,1},ref[3],fused[3];int32_t ca[3];tinyml_conv1d(cx,3,1,cw,(int32_t[]){0},1,3,1,0,ca);CHECK("Conv1D same padding",ca[0]==3&&ca[1]==6&&ca[2]==5);tinyml_requantize(ca,3,1.0,0,ref);tinyml_conv1d_quant(cx,3,1,cw,(int32_t[]){0},1,3,1,0,1.0,0,0,fused);CHECK("Conv1D fused exact",memcmp(ref,fused,3)==0);}
 {int8_t fx[7],fw[42],fr[6],fo[6];int32_t fb[6];for(i=0;i<7;i++)fx[i]=(int8_t)((int)i*29-80);for(i=0;i<42;i++)fw[i]=(int8_t)((int)i*17-100);for(i=0;i<6;i++)fb[i]=(int32_t)i*103-250;tinyml_fc_quant_reference(fx,fw,fb,7,6,-7,0.375,-3,1,fr);tinyml_fc_quant_optimized(fx,fw,fb,7,6,-7,0.375,-3,1,fo);CHECK("optimized FC bit-exact incl. output tail",memcmp(fr,fo,sizeof fr)==0);}
 {int8_t fx[7]={1,-2,3,-4,5,-6,7},fw[7]={2,-3,4,-5,6,-7,8},fr,fo;int32_t fb=INT32_MAX-1;tinyml_fc_quant_reference(fx,fw,&fb,7,1,0,1.0,-128,0,&fr);tinyml_fc_quant_optimized(fx,fw,&fb,7,1,0,1.0,-128,0,&fo);CHECK("optimized FC safe wide-accumulator fallback",fr==fo);}
 {int8_t cx[15],cw[36],cr[20],co[20];int32_t cb[4]={-101,33,0,127};for(i=0;i<15;i++)cx[i]=(int8_t)((int)i*23-90);for(i=0;i<36;i++)cw[i]=(int8_t)((int)i*11-71);tinyml_conv1d_quant_reference(cx,5,3,cw,cb,4,3,1,5,0.25,-4,1,cr);tinyml_conv1d_quant_optimized(cx,5,3,cw,cb,4,3,1,5,0.25,-4,1,co);CHECK("optimized Conv1D bit-exact with same padding",memcmp(cr,co,sizeof cr)==0);tinyml_conv1d_quant_reference(cx,5,3,cw,cb,4,3,0,5,0.25,-4,0,cr);tinyml_conv1d_quant_optimized(cx,5,3,cw,cb,4,3,0,5,0.25,-4,0,co);CHECK("optimized Conv1D bit-exact with valid padding",memcmp(cr,co,3U*4U)==0);}
 for(i=0;i<256;i++){lut[i]=1.0;} tinyml_softmax_lut((int8_t[]){-1,0},2,lut,0.5,-128,q);CHECK("softmax LUT",q[0]==-127&&q[1]==-127);
 tinyml_arena_init(&ar,mem,sizeof mem);CHECK("arena aligned alloc",tinyml_arena_alloc(&ar,4,4)!=0);CHECK("arena bounds",tinyml_arena_alloc(&ar,32,1)==0);
 if(fails)printf("%d failures\n",fails);else puts("ALL KERNEL TESTS PASSED");return fails?1:0;}
