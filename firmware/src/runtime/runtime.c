#include "runtime.h"
#include "kernels/kernels.h"
#include <string.h>
static int tinyml_infer_impl(const tinyml_model_t*m,const int8_t*input,void*memory,size_t bytes,const int8_t**out,int optimized){
    size_t i,even_bytes,odd_bytes;int8_t *buf[2];const int8_t *src=input;
    if(!m||!input||!memory||!out||bytes<m->arena_n||m->arena_n!=m->even_n+m->odd_n)return -1;
    even_bytes=m->even_n;odd_bytes=m->odd_n;buf[0]=(int8_t*)memory;buf[1]=buf[0]+even_bytes;
    for(i=0;i<m->n_layers;i++){
        const tinyml_layer_t*l=&m->layers[i];int8_t*dst=buf[i&1U];
        if(l->out_n>((i&1U)?odd_bytes:even_bytes))return -2;
        switch(l->op){
        case TINYML_FC:
            if(optimized) tinyml_fc_quant_optimized(src,l->weights,l->bias,l->in_n,l->out_n,*l->zp_in,(*l->scale_in**l->scale_w)/ *l->scale_out,*l->zp_out,l->relu,dst);
            else tinyml_fc_quant_reference(src,l->weights,l->bias,l->in_n,l->out_n,*l->zp_in,(*l->scale_in**l->scale_w)/ *l->scale_out,*l->zp_out,l->relu,dst);
            break;
        case TINYML_CONV1D:
            if(optimized) tinyml_conv1d_quant_optimized(src,l->in_t,l->in_c,l->weights,l->bias,l->out_c,l->kernel,l->same,*l->zp_in,(*l->scale_in**l->scale_w)/ *l->scale_out,*l->zp_out,l->relu,dst);
            else tinyml_conv1d_quant_reference(src,l->in_t,l->in_c,l->weights,l->bias,l->out_c,l->kernel,l->same,*l->zp_in,(*l->scale_in**l->scale_w)/ *l->scale_out,*l->zp_out,l->relu,dst);
            break;
        case TINYML_MAXPOOL1D: tinyml_maxpool1d(src,l->in_t,l->in_c,l->kernel,l->stride,dst);break;
        case TINYML_FLATTEN: memcpy(dst,src,l->out_n);break;
        case TINYML_SOFTMAX: tinyml_softmax_lut(src,l->in_n,l->lut,*l->scale_out,*l->zp_out,dst);break;
        default:return -3;
        }
        src=dst;
    }
    *out=src;return 0;
}
int tinyml_infer(const tinyml_model_t*m,const int8_t*input,void*memory,size_t bytes,const int8_t**out){return tinyml_infer_impl(m,input,memory,bytes,out,1);}
int tinyml_infer_reference(const tinyml_model_t*m,const int8_t*input,void*memory,size_t bytes,const int8_t**out){return tinyml_infer_impl(m,input,memory,bytes,out,0);}
