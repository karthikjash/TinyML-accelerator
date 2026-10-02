cmake_minimum_required(VERSION 3.22.0)

if(NOT DEFINED SdkRootDirPath)
    message(FATAL_ERROR "Pass -DSdkRootDirPath=<MCUXpresso SDK mcuxsdk directory>")
endif()
set(board frdmmcxn236 CACHE STRING "MCUXpresso board id")
set(board_root examples/_boards CACHE STRING "MCUXpresso board project root")
set(TINYML_ROOT ${CMAKE_CURRENT_LIST_DIR}/../..)

# Match the official SDK hello_world project so its board port supplies startup,
# system/clock initialization, UART console, linker selection, and SDK drivers.
include(${SdkRootDirPath}/cmake/extension/mcux.cmake)
project(tinyml_mcxn236 LANGUAGES C CXX ASM
        PROJECT_BOARD_PORT_PATH ${board_root}/${board}/demo_apps/hello_world)
include(${SdkRootDirPath}/CMakeLists.txt)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_compile_definitions(PLATFORM_MCXN236)
mcux_add_include(BASE_PATH ${TINYML_ROOT} INCLUDES firmware/src)
mcux_add_source(
    BASE_PATH ${TINYML_ROOT}
    SOURCES
        firmware/mcxn236/main.c
        firmware/mcxn236/model_data_mlp.c
        firmware/mcxn236/model_data_cnn.c
        firmware/src/platform/platform_mcxn236.c
        firmware/src/kernels/kernels.c
        firmware/src/memory/arena.c
        firmware/src/runtime/runtime.c
        firmware/src/runtime/models.c
        models/export/mlp_demo/golden/vectors.c
        models/export/cnn_demo/golden/vectors.c
)

include(${SdkRootDirPath}/${board_root}/${board}/demo_apps/hello_world/reconfig.cmake OPTIONAL)
mcux_convert_binary(BINARY ${APPLICATION_BINARY_DIR}/${MCUX_SDK_PROJECT_NAME}.bin)
