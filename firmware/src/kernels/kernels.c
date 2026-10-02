#include "kernels.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>

static int32_t clamp_i32(int64_t x) { return x > INT32_MAX ? INT32_MAX : (x < INT32_MIN ? INT32_MIN : (int32_t)x); }
int8_t tinyml_requantize_one(int32_t acc,double s,int32_t zp){double v=rint((double)acc*s);if(v>127.0-(double)zp)return 127;if(v<-128.0-(double)zp)return -128;return (int8_t)((int64_t)v+(int64_t)zp);}
void tinyml_requantize(const int32_t *a, size_t n, double s, int32_t zp, int8_t *y) {
    size_t i; for (i=0;i<n;i++) { double v=rint((double)a[i]*s);
        if (v > 127.0-(double)zp) y[i]=127;
        else if (v < -128.0-(double)zp) y[i]=-128;
        else y[i]=(int8_t)((int64_t)v+(int64_t)zp); }
}
void tinyml_fc_quant_reference(const int8_t *x, const int8_t *w, const int32_t *b,
                               size_t ni, size_t no, int32_t zp, double scale,
                               int32_t zpo, int relu, int8_t *y) {
    size_t o, i;
    for (o = 0; o < no; ++o) {
        int64_t sum = 0;
        for (i = 0; i < ni; ++i)
            sum += (int64_t)w[i * no + o] * ((int32_t)x[i] - zp);
        y[o] = tinyml_requantize_one(clamp_i32(sum + b[o]), scale, zpo);
        if (relu && y[o] < zpo) y[o] = (int8_t)zpo;
    }
}

/* Process four adjacent outputs together. The [input][output] weight layout
 * makes each inner output group contiguous; each output still accumulates
 * inputs in the same order and is requantized only once, exactly as reference. */
void tinyml_fc_quant_optimized(const int8_t *x, const int8_t *w, const int32_t *b,
                               size_t ni, size_t no, int32_t zp, double scale,
                               int32_t zpo, int relu, int8_t *y) {
    size_t base, i, j;
    for (base = 0; base < no; base += 4U) {
        size_t count = no - base < 4U ? no - base : 4U;
        int64_t max_bias = 0;
        int64_t max_delta = llabs((int64_t)-128 - zp);
        int64_t other_delta = llabs((int64_t)127 - zp);
        int64_t max_product, allowed;
        for (j = 0; j < count; ++j) {
            int64_t bias_mag = llabs((int64_t)b[base + j]);
            if (bias_mag > max_bias) max_bias = bias_mag;
        }
        if (other_delta > max_delta) max_delta = other_delta;
        max_product = max_delta * 128;
        allowed = (int64_t)INT32_MAX - max_bias;

        /* M33 has native 32-bit arithmetic. Use it when a conservative bound
         * proves every partial sum and bias addition are representable. */
        if (max_product == 0 || (allowed >= 0 && ni <= (size_t)(allowed / max_product))) {
            int32_t sums[4] = { 0, 0, 0, 0 };
            for (i = 0; i < ni; ++i) {
                int32_t centered = (int32_t)x[i] - zp;
                const int8_t *weights = w + i * no + base;
                for (j = 0; j < count; ++j)
                    sums[j] += (int32_t)weights[j] * centered;
            }
            for (j = 0; j < count; ++j) {
                int8_t value = tinyml_requantize_one(sums[j] + b[base + j], scale, zpo);
                if (relu && value < zpo) value = (int8_t)zpo;
                y[base + j] = value;
            }
        } else {
            int64_t sums[4] = { 0, 0, 0, 0 };
            for (i = 0; i < ni; ++i) {
                int32_t centered = (int32_t)x[i] - zp;
                const int8_t *weights = w + i * no + base;
                for (j = 0; j < count; ++j)
                    sums[j] += (int64_t)weights[j] * centered;
            }
            for (j = 0; j < count; ++j) {
                int8_t value = tinyml_requantize_one(clamp_i32(sums[j] + b[base + j]), scale, zpo);
                if (relu && value < zpo) value = (int8_t)zpo;
                y[base + j] = value;
            }
        }
    }
}

void tinyml_fc_quant(const int8_t *x, const int8_t *w, const int32_t *b,
                     size_t ni, size_t no, int32_t zp, double scale,
                     int32_t zpo, int relu, int8_t *y) {
    tinyml_fc_quant_optimized(x, w, b, ni, no, zp, scale, zpo, relu, y);
}
void tinyml_fc(const int8_t *x,const int8_t *w,const int32_t *b,size_t ni,size_t no,int32_t zp,int32_t *a) {
    size_t o,i; for(o=0;o<no;o++){ int64_t s=0; for(i=0;i<ni;i++) s+=(int64_t)w[i*no+o]*((int32_t)x[i]-zp); a[o]=clamp_i32(s+b[o]); }
}
void tinyml_conv1d_quant_reference(const int8_t *x, size_t t, size_t ci,
                                  const int8_t *w, const int32_t *b, size_t co,
                                  size_t k, int same, int32_t zp, double scale,
                                  int32_t zpo, int relu, int8_t *y) {
    size_t to = same ? t : (t >= k ? t - k + 1U : 0U);
    size_t left = same ? (k - 1U) / 2U : 0U;
    size_t ot, oc, kk, ic;
    for (ot = 0; ot < to; ++ot) for (oc = 0; oc < co; ++oc) {
        int64_t sum = 0;
        for (kk = 0; kk < k; ++kk) {
            int64_t ix = (int64_t)ot + (int64_t)kk - (int64_t)left;
            for (ic = 0; ic < ci; ++ic) {
                int32_t xv = (ix < 0 || ix >= (int64_t)t) ? zp : x[(size_t)ix * ci + ic];
                sum += (int64_t)w[(ic * co + oc) * k + kk] * (xv - zp);
            }
        }
        y[ot * co + oc] = tinyml_requantize_one(clamp_i32(sum + b[oc]), scale, zpo);
        if (relu && y[ot * co + oc] < zpo) y[ot * co + oc] = (int8_t)zpo;
    }
}

/* Preserve the reference reduction order (kernel position, then input
 * channel), but traverse each input row linearly and advance through the
 * [input_channel][output_channel][kernel] weights by pointer. Padding contributes
 * exact zero after zero-point subtraction and can skip the channel loop. */
void tinyml_conv1d_quant_optimized(const int8_t *x, size_t t, size_t ci,
                                  const int8_t *w, const int32_t *b, size_t co,
                                  size_t k, int same, int32_t zp, double scale,
                                  int32_t zpo, int relu, int8_t *y) {
    size_t to = same ? t : (t >= k ? t - k + 1U : 0U);
    size_t left = same ? (k - 1U) / 2U : 0U;
    size_t ot, oc, kk, ic;
    for (ot = 0; ot < to; ++ot) {
        for (oc = 0; oc < co; ++oc) {
            int64_t sum = 0;
            for (kk = 0; kk < k; ++kk) {
                int64_t ix = (int64_t)ot + (int64_t)kk - (int64_t)left;
                if (ix >= 0 && ix < (int64_t)t) {
                    const int8_t *input = x + (size_t)ix * ci;
                    const int8_t *weight = w + oc * k + kk;
                    for (ic = 0; ic < ci; ++ic) {
                        sum += (int64_t)weight[0] * ((int32_t)input[0] - zp);
                        ++input;
                        weight += co * k;
                    }
                }
            }
            y[ot * co + oc] = tinyml_requantize_one(clamp_i32(sum + b[oc]), scale, zpo);
            if (relu && y[ot * co + oc] < zpo) y[ot * co + oc] = (int8_t)zpo;
        }
    }
}

void tinyml_conv1d_quant(const int8_t *x, size_t t, size_t ci, const int8_t *w,
                         const int32_t *b, size_t co, size_t k, int same,
                         int32_t zp, double scale, int32_t zpo, int relu, int8_t *y) {
    tinyml_conv1d_quant_optimized(x, t, ci, w, b, co, k, same, zp, scale, zpo, relu, y);
}
void tinyml_conv1d(const int8_t *x,size_t t,size_t ci,const int8_t *w,const int32_t *b,size_t co,size_t k,int same,int32_t zp,int32_t *a) {
    size_t to=same?t:(t>=k?t-k+1:0), left=same?(k-1)/2:0, ot,oc,kk,ic;
    for(ot=0;ot<to;ot++) for(oc=0;oc<co;oc++){ int64_t s=0;
        for(kk=0;kk<k;kk++){ int64_t ix=(int64_t)ot+(int64_t)kk-(int64_t)left;
            for(ic=0;ic<ci;ic++){ int32_t xv=(ix<0||ix>=(int64_t)t)?zp:x[(size_t)ix*ci+ic];
                s+=(int64_t)w[(ic*co+oc)*k+kk]*(xv-zp); }}
        a[ot*co+oc]=clamp_i32(s+b[oc]); }
}
void tinyml_relu(int8_t *x,size_t n,int32_t zp){ size_t i; for(i=0;i<n;i++) if(x[i]<zp)x[i]=(int8_t)zp; }
void tinyml_maxpool1d(const int8_t*x,size_t t,size_t c,size_t win,size_t stride,int8_t*y){ size_t ot,ch,j,no=stride?((t<win)?0:1+(t-win)/stride):0;
    for(ot=0;ot<no;ot++)for(ch=0;ch<c;ch++){ int8_t m=x[(ot*stride)*c+ch]; for(j=1;j<win;j++){int8_t v=x[(ot*stride+j)*c+ch];if(v>m)m=v;} y[ot*c+ch]=m;}}
void tinyml_softmax_lut(const int8_t*x,size_t n,const double lut[256],double scale,int32_t zp,int8_t*y){ size_t i;double sum=0.0;
    for(i=0;i<n;i++) { sum+=lut[(int)x[i]+128]; }
    for(i=0;i<n;i++){double v=rint((lut[(int)x[i]+128]/sum)/scale);
        if(v>127.0-(double)zp)y[i]=127;else if(v<-128.0-(double)zp)y[i]=-128;else y[i]=(int8_t)((int64_t)v+(int64_t)zp);}}
