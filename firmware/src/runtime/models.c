#include "models.h"
#define sm_lut mlp_sm_lut
#include "../../../models/export/mlp_demo/model_data.h"
#undef sm_lut
#define sm_lut cnn_sm_lut
#include "../../../models/export/cnn_demo/model_data.h"
#undef sm_lut
static const double mlp_sm_scale=0.00392156862745098,cnn_sm_scale=0.00392156862745098;
static const int softmax_zp_out=-128;
static const tinyml_layer_t mlp_layers[]={
 {TINYML_FC,384,128,0,0,0,0,0,0,fc1_w,fc1_b,0,&fc1_scale_in,&fc1_scale_w,&fc1_scale_out,&fc1_zp_in,&fc1_zp_out,0,1},
 {TINYML_FC,128,64,0,0,0,0,0,0,fc2_w,fc2_b,0,&fc2_scale_in,&fc2_scale_w,&fc2_scale_out,&fc2_zp_in,&fc2_zp_out,0,1},
 {TINYML_FC,64,4,0,0,0,0,0,0,fc3_w,fc3_b,0,&fc3_scale_in,&fc3_scale_w,&fc3_scale_out,&fc3_zp_in,&fc3_zp_out,0,0},
 {TINYML_SOFTMAX,4,4,0,0,0,0,0,0,0,0,mlp_sm_lut,0,0,&mlp_sm_scale,&fc3_zp_out,&softmax_zp_out,0,0}
};
static const tinyml_layer_t cnn_layers[]={
 {TINYML_CONV1D,384,2048,128,3,128,16,5,1,conv1_w,conv1_b,0,&conv1_scale_in,&conv1_scale_w,&conv1_scale_out,&conv1_zp_in,&conv1_zp_out,1,1},
 {TINYML_MAXPOOL1D,2048,1024,128,16,64,16,2,2,0,0,0,0,0,0,0,0,0,0},
 {TINYML_CONV1D,1024,2048,64,16,64,32,3,1,conv2_w,conv2_b,0,&conv2_scale_in,&conv2_scale_w,&conv2_scale_out,&conv2_zp_in,&conv2_zp_out,1,1},
 {TINYML_MAXPOOL1D,2048,1024,64,32,32,32,2,2,0,0,0,0,0,0,0,0,0,0},
 {TINYML_FLATTEN,1024,1024,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
 {TINYML_FC,1024,4,0,0,0,0,0,0,fc_w,fc_b,0,&fc_scale_in,&fc_scale_w,&fc_scale_out,&fc_zp_in,&fc_zp_out,0,0},
 {TINYML_SOFTMAX,4,4,0,0,0,0,0,0,0,0,cnn_sm_lut,0,0,&cnn_sm_scale,&fc_zp_out,&softmax_zp_out,0,0}
};
const tinyml_model_t tinyml_mlp_demo={"mlp_demo",mlp_layers,4,384,192,128,64,4};
const tinyml_model_t tinyml_cnn_demo={"cnn_demo",cnn_layers,7,384,3072,2048,1024,4};
