/* FRDM-MCXN236 golden verification application (requires the MCUXpresso board project). */
#include "board.h"
#include "app.h"
#include "platform/platform.h"
#include "runtime/models.h"
#include "../../models/export/cnn_demo/model_data.h"
#include "mcuxsdk_version.h"
extern const int8_t GOLDEN_MLP_DEMO_FULL_IN[8*384],GOLDEN_MLP_DEMO_FULL_OUT[8*4];
extern const int8_t GOLDEN_CNN_DEMO_FULL_IN[8*384],GOLDEN_CNN_DEMO_FULL_OUT[8*4];

static uint8_t arena[CNN_DEMO_ARENA_SIZE];

static int verify_model(const tinyml_model_t *model,const int8_t *inputs,
                        const int8_t *expected,size_t samples,size_t in_n,size_t out_n){
    size_t s,j;const int8_t *out;uint32_t begin,end;uint64_t total_cycles=0U;
    for(s=0;s<samples;s++){
        begin=(uint32_t)hal_get_cycle_count();
        if(tinyml_infer(model,inputs+s*in_n,arena,model->arena_n,&out)!=0){hal_print("runtime error\r\n");return 0;}
        end=(uint32_t)hal_get_cycle_count();
        total_cycles+=(uint32_t)(end-begin);
        for(j=0;j<out_n;j++)if(out[j]!=expected[s*out_n+j]){hal_print_int("sample FAIL: ",(int64_t)s);return 0;}
        hal_print_int("sample PASS: ",(int64_t)s);
    }
    hal_print_int("exact samples: ",(int64_t)samples);
    if(total_cycles)hal_print_int("inference cycles (all samples): ",(int64_t)total_cycles);
    else hal_print("Cycle counter unavailable/not verified\r\n");
    return 1;
}

int main(void){
    int mlp_ok,cnn_ok;
    BOARD_InitHardware();
    hal_init();
    hal_print("TinyML MCXN236 Day 3\r\nSDK: ");
    hal_print(MCUXSDK_VERSION_FULL_STR);
    hal_print("\r\n\r\nMLP target verification\r\n");
    mlp_ok=verify_model(&tinyml_mlp_demo,GOLDEN_MLP_DEMO_FULL_IN,GOLDEN_MLP_DEMO_FULL_OUT,8,384,4);
    hal_print(mlp_ok?"MLP golden test: PASS\r\n":"MLP golden test: FAIL\r\n");
    hal_print("\r\nCNN target verification\r\n");
    cnn_ok=verify_model(&tinyml_cnn_demo,GOLDEN_CNN_DEMO_FULL_IN,GOLDEN_CNN_DEMO_FULL_OUT,8,384,4);
    hal_print(cnn_ok?"CNN golden test: PASS\r\n":"CNN golden test: FAIL\r\n");
    hal_print((mlp_ok&&cnn_ok)?"DAY3 TARGET VERIFICATION: PASS\r\n":"DAY3 TARGET VERIFICATION: FAIL\r\n");
    for(;;){}
}
