/*
 *
 * Copyright (c) 2017 Texas Instruments Incorporated
 *
 * All rights reserved not granted herein.
 *
 * Limited License.
 *
 * Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
 * license under copyrights and patents it now or hereafter owns or controls to make,
 * have made, use, import, offer to sell and sell ("Utilize") this software subject to the
 * terms herein.  With respect to the foregoing patent license, such license is granted
 * solely to the extent that any such patent is necessary to Utilize the software alone.
 * The patent license shall not apply to any combinations which include this software,
 * other than combinations with devices manufactured by or for TI ("TI Devices").
 * No hardware patent is licensed hereunder.
 *
 * Redistributions must preserve existing copyright notices and reproduce this license
 * (including the above copyright notice and the disclaimer and (if applicable) source
 * code license limitations below) in the documentation and/or other materials provided
 * with the distribution
 *
 * Redistribution and use in binary form, without modification, are permitted provided
 * that the following conditions are met:
 *
 * *       No reverse engineering, decompilation, or disassembly of this software is
 * permitted with respect to any software provided in binary form.
 *
 * *       any redistribution and use are licensed by TI for use only with TI Devices.
 *
 * *       Nothing shall obligate TI to provide you with source code for the software
 * licensed and provided to you in object code.
 *
 * If software source code is provided to you, modification and redistribution of the
 * source code are permitted provided that the following conditions are met:
 *
 * *       any redistribution and use of the source code, including any resulting derivative
 * works, are licensed by TI for use only with TI Devices.
 *
 * *       any redistribution and use of any object code compiled from the source code
 * and any resulting derivative works, are licensed by TI for use only with TI Devices.
 *
 * Neither the name of Texas Instruments Incorporated nor the names of its suppliers
 *
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * DISCLAIMER.
 *
 * THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <TI/tivx.h>
#include <TI/tivx_task.h>
#include <TI/j7.h>
#include <TI/j7_tidl.h>
#include <tivx_utils_file_rd_wr.h>
#include <tivx_utils_graph_perf.h>
#include <tivx_utils_tidl_trace.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/stat.h>
#include <float.h>
#include <math.h>
#include <utils/draw2d/include/draw2d.h>
#include <utils/perf_stats/include/app_perf_stats.h>
#include "itidl_ti.h"


/* #define APP_DEBUG */
/* #define APP_TIDL_TRACE_DUMP */
/*
 * This is the size of trace buffer allocated in host memory and
 * shared with target.
 */
#define TIVX_TIDL_TRACE_DATA_SIZE  (256 * 1024 * 1024)

#define APP_MAX_FILE_PATH           (256u)
#define APP_ASSERT(x)               assert((x))
#define APP_ASSERT_VALID_REF(ref)   (APP_ASSERT(vxGetStatus((vx_reference)(ref))==VX_SUCCESS));

#define APP_MAX_TENSORS             (4u)
#define APP_MAX_TENSOR_DIMS         (4u)
#define APP_TIDL_MAX_PARAMS         (16u)

#define ABS_FLT(a) ((a) > 0)?(a):(-(a))

extern const char imgnet_labels[1000][256];
static const char *tensor_num_str[] = { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"};

#define MAX_IMG_WIDTH  (2048)
#define MAX_IMG_HEIGHT (1024)
#define DISPLAY_WIDTH  (1280)
#define DISPLAY_HEIGHT (720)

#define APP_DEBUG

typedef struct {

    /* config options */
    char tidl_config_file_path[APP_MAX_FILE_PATH];
    char tidl_network_file_path[APP_MAX_FILE_PATH];
    char input_file_path[APP_MAX_FILE_PATH];
    char input_file_list[APP_MAX_FILE_PATH];
    char output_file_path[APP_MAX_FILE_PATH];
    char ti_logo_file_path[APP_MAX_FILE_PATH];

    uint32_t num_input_tensors;
    uint32_t num_output_tensors;

    /* Input image params */
    vx_df_image df_image;
    void *data_ptr;
    void *bmp_context;
    vx_uint32 img_width;
    vx_uint32 img_height;
    vx_uint32 img_stride;

    sTIDL_IOBufDesc_t   ioBufDesc;

    uint8_t *pInPlanes;
    uint8_t *pOutPlanes;

    uint8_t  *pDisplayBuf888;
    uint16_t *pDisplayBuf565;

    /* OpenVX references */
    vx_context context;
    vx_graph   graph;
    vx_kernel  kernel;
    vx_node    tidl_node;

    vx_user_data_object  config;
    vx_user_data_object  network;
    vx_user_data_object  createParams;
    vx_user_data_object  inArgs;
    vx_user_data_object  outArgs;
    vx_user_data_object  traceData;

    vx_tensor  input_tensors[APP_MAX_TENSORS];
    vx_tensor  output_tensors[APP_MAX_TENSORS];

    vx_graph   disp_graph;
    vx_node disp_node;
    vx_image disp_image;

    vx_user_data_object disp_params_obj;
    tivx_display_params_t disp_params;
    vx_rectangle_t disp_rect;
    vx_imagepatch_addressing_t image_addr;

    vx_uint32 display_option;
    vx_uint32 delay_in_msecs;
    vx_uint32 num_iterations;

    Draw2D_Handle  pHndl;

    uint32_t is_interactive;
    tivx_task task;
    uint32_t stop_task;
    uint32_t stop_task_done;

    app_perf_point_t total_perf;
    app_perf_point_t fileio_perf;
    app_perf_point_t draw_perf;

} AppObj;

AppObj gAppObj;

typedef struct 
{
    uint32_t width;
    uint32_t heights;
    uint32_t channels;
    unsigned char* imageData;
} ImageInfo;

static int app_parse_cmd_line_args(AppObj *obj, int argc, char *argv[]);
static int app_init(AppObj *obj);
static void app_deinit(AppObj *obj);
static vx_status app_create_graph(AppObj *obj);
static vx_status app_verify_graph(AppObj *obj);
static vx_status app_run_graph(AppObj *obj);
static vx_status app_run_graph_interactive(AppObj *obj);
static void app_delete_graph(AppObj *obj);

static vx_user_data_object readConfig(AppObj *obj, vx_context context, char *config_file, uint32_t *num_input_tensors, uint32_t *num_output_tensors);
static vx_user_data_object readNetwork(vx_context context, char *network_file);
static vx_user_data_object setCreateParams(vx_context context);
static vx_user_data_object setInArgs(vx_context context);
static vx_user_data_object setOutArgs(vx_context context);
static void createInputTensors(AppObj *obj, vx_context context, vx_user_data_object config, vx_tensor *input_tensors);
static void createOutputTensors(AppObj *obj, vx_context context, vx_user_data_object config, vx_tensor *output_tensors);
static vx_status readInput(AppObj *obj, vx_context context, vx_user_data_object config, vx_tensor *input_tensors, char *input_file);
static void displayOutput(AppObj *obj, vx_user_data_object config, vx_tensor *output_tensors, char *output_file);

static uint32_t num_params;
static uint32_t max_params;


int app_tidl_main(int argc, char* argv[])
{
    int status = 0;

    AppObj *obj = &gAppObj;

    status = app_parse_cmd_line_args(obj, argc, argv);
    if(status != -1) {
      status = app_init(obj);
      if(status != -1) {
        app_create_graph(obj);
        status = app_verify_graph(obj);
        if(status == 0)
        {
            if(obj->is_interactive)
            {
                app_run_graph_interactive(obj);
            }
            else
            {
                app_run_graph(obj);
            }
        }
        app_delete_graph(obj);
        app_deinit(obj);
      }
    }

    return status;
}

static int app_init(AppObj *obj)
{

    int status = 0;

    uint32_t num_input_tensors = 0;
    uint32_t num_output_tensors = 0;

    #ifdef APP_DEBUG
    printf("app_tidl: Init ... \n");
    #endif

    obj->context = vxCreateContext();
    APP_ASSERT_VALID_REF(obj->context);

    /* Create a vx_array object and read the config data*/
    obj->config = readConfig(obj, obj->context, &obj->tidl_config_file_path[0], &num_input_tensors, &num_output_tensors);
    APP_ASSERT_VALID_REF(obj->config)

    /* Save a copy of number of input/output tensors required as per config */
    obj->num_input_tensors  = num_input_tensors;
    obj->num_output_tensors = num_output_tensors;

    /* Create a vx_tensor object and read the network data */
    obj->network = readNetwork(obj->context, &obj->tidl_network_file_path[0]);
    APP_ASSERT_VALID_REF(obj->network)

    obj->createParams = setCreateParams(obj->context);
    APP_ASSERT_VALID_REF(obj->createParams)

    obj->inArgs = setInArgs(obj->context);
    APP_ASSERT_VALID_REF(obj->inArgs)

    obj->outArgs = setOutArgs(obj->context);
    APP_ASSERT_VALID_REF(obj->outArgs)

#ifdef APP_TIDL_TRACE_DUMP
    obj->traceData = vxCreateUserDataObject(obj->context, "TIDL_traceData", TIVX_TIDL_TRACE_DATA_SIZE, NULL);
    APP_ASSERT_VALID_REF(obj->traceData)
#endif

    obj->kernel = tivxAddKernelTIDL(obj->context, num_input_tensors, num_output_tensors);
    APP_ASSERT_VALID_REF(obj->kernel)

    if ((vx_true_e == tivxIsTargetEnabled(TIVX_TARGET_DISPLAY1)) && (obj->display_option == 1))
    {
        obj->disp_image = vxCreateImage(obj->context, DISPLAY_WIDTH, DISPLAY_HEIGHT, VX_DF_IMAGE_RGB);
        APP_ASSERT_VALID_REF(obj->disp_image)

        obj->image_addr.dim_x = DISPLAY_WIDTH;
        obj->image_addr.dim_y = DISPLAY_HEIGHT;
        obj->image_addr.stride_x = 3; /* RGB */
        obj->image_addr.stride_y = DISPLAY_WIDTH * 3;
        obj->image_addr.scale_x = VX_SCALE_UNITY;
        obj->image_addr.scale_y = VX_SCALE_UNITY;
        obj->image_addr.step_x = 1;
        obj->image_addr.step_y = 1;

        obj->disp_rect.start_x = 0;
        obj->disp_rect.start_y = 0;
        obj->disp_rect.end_x = DISPLAY_WIDTH;
        obj->disp_rect.end_y = DISPLAY_HEIGHT;

        memset(&obj->disp_params, 0, sizeof(tivx_display_params_t));

        obj->disp_params.opMode = TIVX_KERNEL_DISPLAY_BUFFER_COPY_MODE;
        obj->disp_params.pipeId = 0;
        obj->disp_params.outWidth = DISPLAY_WIDTH;
        obj->disp_params.outHeight = DISPLAY_HEIGHT;
        obj->disp_params.posX = (1920-DISPLAY_WIDTH)/2;
        obj->disp_params.posY = (1080-DISPLAY_HEIGHT)/2;

        obj->disp_params_obj = vxCreateUserDataObject(obj->context, "tivx_display_params_t", sizeof(tivx_display_params_t), &obj->disp_params);
        APP_ASSERT_VALID_REF(obj->disp_params_obj)

        tivxHwaLoadKernels(obj->context);
    }

    tivxTIDLLoadKernels(obj->context);

    obj->pInPlanes = tivxMemAlloc(MAX_IMG_WIDTH * MAX_IMG_HEIGHT * 3, TIVX_MEM_EXTERNAL);
    if(obj->pInPlanes == NULL) {
        printf("app_tidl: ERROR: Unable to allocate memory for inPlanes, size = %d\n", MAX_IMG_WIDTH * MAX_IMG_HEIGHT * 3);
        status = -1;
    }

    obj->pOutPlanes = tivxMemAlloc(MAX_IMG_WIDTH * MAX_IMG_HEIGHT * 3, TIVX_MEM_EXTERNAL);
    if(obj->pOutPlanes == NULL) {
        printf("app_tidl: ERROR: Unable to allocate memory for outPlanes, size = %d\n", MAX_IMG_WIDTH * MAX_IMG_HEIGHT * 3);
        status = -1;
    }

    obj->pDisplayBuf888 = tivxMemAlloc(DISPLAY_WIDTH * DISPLAY_HEIGHT * 3, TIVX_MEM_EXTERNAL);
    if(obj->pDisplayBuf888 == NULL) {
        printf("app_tidl: ERROR: Unable to allocate memory for displayBuf888, size = %d\n", DISPLAY_WIDTH * DISPLAY_HEIGHT * 3);
        status = -1;
    }

    obj->pDisplayBuf565 = tivxMemAlloc(DISPLAY_WIDTH * DISPLAY_HEIGHT * sizeof(uint16_t), TIVX_MEM_EXTERNAL);
    if(obj->pDisplayBuf565 == NULL) {
        printf("app_tidl: ERROR: Unable to allocate memory for displayBuf565, size = %ld\n", DISPLAY_WIDTH * DISPLAY_HEIGHT * sizeof(uint16_t));
        status = -1;
    }

    {
      Draw2D_BufInfo sBufInfo;
      Draw2D_LinePrm sLinePrm;
      Draw2D_FontPrm sTop5Prm;

      char banner_file[APP_MAX_FILE_PATH];

      snprintf(banner_file, APP_MAX_FILE_PATH, "%s/ti_logo.bmp", obj->ti_logo_file_path);

      Draw2D_create(&obj->pHndl);

      if(obj->pHndl != NULL)
      {
          sBufInfo.bufWidth    = DISPLAY_WIDTH;
          sBufInfo.bufHeight   = DISPLAY_HEIGHT;
          sBufInfo.bufPitch[0] = DISPLAY_WIDTH * 2;
          sBufInfo.dataFormat = DRAW2D_DF_BGR16_565;
          sBufInfo.transperentColor = 0;
          sBufInfo.transperentColorFormat = DRAW2D_DF_BGR16_565;

          sBufInfo.bufAddr[0] = (uint8_t *)obj->pDisplayBuf565;

          Draw2D_setBufInfo(obj->pHndl, &sBufInfo);

          Draw2D_clearBuf(obj->pHndl);

          Draw2D_insertBmp(obj->pHndl, banner_file, 0, 0);

          sLinePrm.lineColor = RGB888_TO_RGB565(255, 255, 255);
          sLinePrm.lineSize  = 3;
          sLinePrm.lineColorFormat = DRAW2D_DF_BGR16_565;

          /* Draw a vertial line */
          Draw2D_drawLine(obj->pHndl, DISPLAY_WIDTH/2, 0, DISPLAY_WIDTH/2, DISPLAY_HEIGHT, &sLinePrm);

          /* green color heading */
          Draw2D_setFontColor(RGB888_TO_RGB565(0, 255, 0), RGB888_TO_RGB565(0, 0, 0), RGB888_TO_RGB565(0, 0, 0));

          sTop5Prm.fontIdx = 0;
          Draw2D_drawString(obj->pHndl, (DISPLAY_WIDTH/2) + 40, 140, "Top 5 classes", &sTop5Prm);

          Draw2D_resetFontColor();
      }
    }

    appPerfPointSetName(&obj->total_perf , "TOTAL");
    appPerfPointSetName(&obj->draw_perf  , "DRAW");
    appPerfPointSetName(&obj->fileio_perf, "FILEIO");

    #ifdef APP_DEBUG
    printf("app_tidl: Init ... Done.\n");
    #endif

    return status;
}

static void app_deinit(AppObj *obj)
{
    #ifdef APP_DEBUG
    printf("app_tidl: De-init ... \n");
    #endif

    tivxTIDLUnLoadKernels(obj->context);

    if (vx_true_e == tivxIsTargetEnabled(TIVX_TARGET_DISPLAY1) && (obj->display_option == 1))
    {
        tivxHwaUnLoadKernels(obj->context);
    }

    vxRemoveKernel(obj->kernel);

    vxReleaseContext(&obj->context);

    tivxMemFree(obj->pInPlanes, MAX_IMG_WIDTH * MAX_IMG_HEIGHT * 3, TIVX_MEM_EXTERNAL);
    tivxMemFree(obj->pOutPlanes, MAX_IMG_WIDTH * MAX_IMG_HEIGHT * 3, TIVX_MEM_EXTERNAL);

    tivxMemFree(obj->pDisplayBuf888, DISPLAY_WIDTH * DISPLAY_HEIGHT * 3, TIVX_MEM_EXTERNAL);
    tivxMemFree(obj->pDisplayBuf565 , DISPLAY_WIDTH * DISPLAY_HEIGHT * sizeof(uint16_t), TIVX_MEM_EXTERNAL);


    {
        Draw2D_delete(obj->pHndl);
    }

    #ifdef APP_DEBUG
    printf("app_tidl: De-init ... Done.\n");
    #endif
}

static void app_delete_graph(AppObj *obj)
{
    uint32_t id;

    #ifdef APP_DEBUG
    printf("app_tidl: Delete ... \n");
    #endif

    if (vx_true_e == tivxIsTargetEnabled(TIVX_TARGET_DISPLAY1) && (obj->display_option == 1))
    {
        vxReleaseNode(&obj->disp_node);
        vxReleaseGraph(&obj->disp_graph);
    }

    vxReleaseNode(&obj->tidl_node);
    vxReleaseGraph(&obj->graph);

    vxReleaseUserDataObject(&obj->config);
    vxReleaseUserDataObject(&obj->network);

    vxReleaseUserDataObject(&obj->createParams);
    vxReleaseUserDataObject(&obj->inArgs);
    vxReleaseUserDataObject(&obj->outArgs);

#ifdef APP_TIDL_TRACE_DUMP
    vxReleaseUserDataObject(&obj->traceData);
#endif

    for(id = 0; id < obj->num_input_tensors; id++) {
        vxReleaseTensor(&obj->input_tensors[id]);
    }

    for(id = 0; id < obj->num_output_tensors; id++) {
        vxReleaseTensor(&obj->output_tensors[id]);
    }

    if ((vx_true_e == tivxIsTargetEnabled(TIVX_TARGET_DISPLAY1)) && (obj->display_option == 1))
    {
      vxReleaseImage(&obj->disp_image);
      vxReleaseUserDataObject(&obj->disp_params_obj);
    }

    #ifdef APP_DEBUG
    printf("app_tidl: Delete ... Done.\n");
    #endif
}

static void app_show_usage(int argc, char* argv[])
{
    printf("\n");
    printf(" TIDL Demo - (c) Texas Instruments 2018\n");
    printf(" ========================================================\n");
    printf("\n");
    printf(" Usage,\n");
    printf("  %s --cfg <config file>\n", argv[0]);
    printf("\n");
}

static void app_set_cfg_default(AppObj *obj)
{
    snprintf(obj->tidl_config_file_path,APP_MAX_FILE_PATH, "test_data/tivx/tidl_models/mobilenetv1/config.bin");
    snprintf(obj->tidl_network_file_path,APP_MAX_FILE_PATH, "test_data/tivx/tidl_models/mobilenetv1/network.bin");
    snprintf(obj->input_file_path,APP_MAX_FILE_PATH, "test_data/psdkra/app_tidl");
    snprintf(obj->input_file_list,APP_MAX_FILE_PATH, "test_data/psdkra/app_tidl/names.txt");
    snprintf(obj->output_file_path,APP_MAX_FILE_PATH, "app_tidl_out");
    snprintf(obj->ti_logo_file_path,APP_MAX_FILE_PATH, "test_data/tivx/tidl_models/");
    obj->display_option = 1;
    obj->delay_in_msecs = 0;
    obj->num_iterations = 1;
    obj->is_interactive = 0;
}

static int app_parse_cfg_file(AppObj *obj, char *cfg_file_name)
{
    FILE *fp = fopen(cfg_file_name, "r");
    char line_str[1024];
    char *token;

    if(fp==NULL)
    {
        printf("app_tidl: ERROR: Unable to open config file [%s]\n", cfg_file_name);
        return -1;
    }

    while(fgets(line_str, sizeof(line_str), fp)!=NULL)
    {
        char s[]=" \t";

        if (strchr(line_str, '#'))
        {
            continue;
        }

        /* get the first token */
        token = strtok(line_str, s);

        if(token != NULL)
        {
            if(strcmp(token, "tidl_config_file_path")==0)
            {
                token = strtok(NULL, s);
                if(token != NULL)
                {
                  token[strlen(token)-1]=0;
                  strcpy(obj->tidl_config_file_path, token);
                }
            }
            else
            if(strcmp(token, "tidl_network_file_path")==0)
            {
                token = strtok(NULL, s);
                if(token != NULL)
                {
                  token[strlen(token)-1]=0;
                  strcpy(obj->tidl_network_file_path, token);
                }
            }
            else
            if(strcmp(token, "input_file_path")==0)
            {
                token = strtok(NULL, s);
                if(token != NULL)
                {
                  token[strlen(token)-1]=0;
                  strcpy(obj->input_file_path, token);
                }
            }
            else
            if(strcmp(token, "ti_logo_file_path")==0)
            {
                token = strtok(NULL, s);
                if(token != NULL)
                {
                  token[strlen(token)-1]=0;
                  strcpy(obj->ti_logo_file_path, token);
                }
            }
            else
            if(strcmp(token, "input_file_list")==0)
            {
                token = strtok(NULL, s);
                if(token != NULL)
                {
                  token[strlen(token)-1]=0;
                  strcpy(obj->input_file_list, token);
                }
            }
            else
            if(strcmp(token, "output_file_path")==0)
            {
                token = strtok(NULL, s);
                if(token != NULL)
                {
                  token[strlen(token)-1]=0;
                  strcpy(obj->output_file_path, token);
                }
            }
            else
            if(strcmp(token, "display_option")==0)
            {
                token = strtok(NULL, s);
                if(token != NULL)
                {
                  token[strlen(token)-1]=0;
                  obj->display_option = atoi(token);
                  if(obj->display_option > 1)
                      obj->display_option = 1;
                }
            }
            else
            if(strcmp(token, "delay")==0)
            {
                token = strtok(NULL, s);
                if(token != NULL)
                {
                  token[strlen(token)-1]=0;
                  obj->delay_in_msecs = atoi(token);
                  if(obj->delay_in_msecs > 2000)
                      obj->delay_in_msecs = 2000;
                }
            }
            else
            if(strcmp(token, "num_iterations")==0)
            {
                token = strtok(NULL, s);
                if(token != NULL)
                {
                  token[strlen(token)-1]=0;
                  obj->num_iterations = atoi(token);
                  if(obj->num_iterations == 0)
                      obj->num_iterations = 1;
                }
            }
            else
            if(strcmp(token, "is_interactive")==0)
            {
                token = strtok(NULL, s);
                if(token != NULL)
                {
                  token[strlen(token)-1]=0;
                  obj->is_interactive = atoi(token);
                  if(obj->is_interactive > 1)
                      obj->is_interactive = 1;
                }
          }
        }
    }

    fclose(fp);

    return 0;
}

static int app_parse_cmd_line_args(AppObj *obj, int argc, char *argv[])
{
    int i;

    app_set_cfg_default(obj);

    if(argc==1)
    {
        app_show_usage(argc, argv);
        return -1;
    }

    for(i=0; i<argc; i++)
    {
        if(strcmp(argv[i], "--cfg")==0)
        {
            i++;
            if(i>=argc)
            {
                app_show_usage(argc, argv);
            }
            app_parse_cfg_file(obj, argv[i]);
            break;
        }
        else
        if(strcmp(argv[i], "--help")==0)
        {
            app_show_usage(argc, argv);
            return -1;
        }
    }

    #ifdef x86_64
    obj->display_option = 0;
    obj->is_interactive = 0;
    #endif

    return 0;
}

static void initParam(vx_reference params[], uint32_t _max_params)
{
   num_params  = 0;
   max_params = _max_params;
}

static void addParam(vx_reference params[], vx_reference obj)
{
   APP_ASSERT(num_params <= max_params);

   params[num_params] = obj;

   num_params++;
}

static vx_status app_create_graph(AppObj *obj)
{
    vx_status status = VX_SUCCESS;

    vx_reference params[APP_TIDL_MAX_PARAMS];
    uint32_t i;

    #ifdef APP_DEBUG
    printf("app_tidl: Creating graph ... \n");
    #endif

    /* Create OpenVx Graph */
    obj->graph = vxCreateGraph(obj->context);
    APP_ASSERT_VALID_REF(obj->graph)
    vxSetReferenceName((vx_reference)obj->graph, "Image Classify");

    /* Create array of input tensors */
    createInputTensors(obj, obj->context, obj->config, obj->input_tensors);

    /* Create array of output tensors */
    createOutputTensors(obj, obj->context, obj->config, obj->output_tensors);

    /* Initialize param array */
    initParam(params, APP_TIDL_MAX_PARAMS);

    /* The 1st param MUST be config */
    addParam(params, (vx_reference)obj->config);

    /* The 2nd param MUST be network */
    addParam(params, (vx_reference)obj->network);

    /* The 3rd param MUST be create params */
    addParam(params, (vx_reference)obj->createParams);

    /* The 4th param MUST be inArgs */
    addParam(params, (vx_reference)obj->inArgs);

    /* The 5th param MUST be outArgs */
    addParam(params, (vx_reference)obj->outArgs);

#ifdef APP_TIDL_TRACE_DUMP
    addParam(params, (vx_reference)obj->traceData);
#else
/* The 6th param MUST be NULL if trace data dump is not enabled */
    addParam(params, (vx_reference)NULL);
#endif

    /* Create TIDL Node */
    obj->tidl_node = tivxTIDLNode(obj->graph, obj->kernel, params, obj->input_tensors, obj->output_tensors);
    APP_ASSERT_VALID_REF(obj->tidl_node)

    if ((vx_true_e == tivxIsTargetEnabled(TIVX_TARGET_DISPLAY1)) && (obj->display_option == 1))
    {
        /* Create OpenVx Graph */
        obj->disp_graph = vxCreateGraph(obj->context);
        APP_ASSERT_VALID_REF(obj->disp_graph)
        vxSetReferenceName((vx_reference)obj->disp_graph, "Display");

        obj->disp_node = tivxDisplayNode(obj->disp_graph, obj->disp_params_obj, obj->disp_image);
        APP_ASSERT_VALID_REF(obj->disp_node)

        vxSetNodeTarget(obj->disp_node, VX_TARGET_STRING, TIVX_TARGET_DISPLAY1);
    }

    /* Set names for diferent OpenVX objects */
    vxSetReferenceName((vx_reference)obj->config, "Config");
    vxSetReferenceName((vx_reference)obj->network, "Network");
    vxSetReferenceName((vx_reference)obj->createParams, "CreateParams");
    vxSetReferenceName((vx_reference)obj->inArgs, "InArgs");
    vxSetReferenceName((vx_reference)obj->outArgs, "OutArgs");

    for(i = 0; i < obj->num_input_tensors; i++) {
        char tensor_name[] = "InputTensor_";
        char ref_name[64];
        snprintf(ref_name, 64, "%s%s", tensor_name, tensor_num_str[i]);
        vxSetReferenceName((vx_reference)obj->input_tensors[i], ref_name);
    }

    for(i = 0; i < obj->num_output_tensors; i++) {
        char tensor_name[] = "OutputTensor_";
        char ref_name[64];
        snprintf(ref_name, 64, "%s%s", tensor_name, tensor_num_str[i]);
        vxSetReferenceName((vx_reference)obj->output_tensors[i], ref_name);
    }

    vxSetReferenceName((vx_reference)obj->kernel, "TIDLKernel");
    vxSetReferenceName((vx_reference)obj->tidl_node, "TIDLNode");

    if (vx_true_e == tivxIsTargetEnabled(TIVX_TARGET_DISPLAY1) && (obj->display_option == 1))
    {
        vxSetReferenceName((vx_reference)obj->disp_params_obj, "DisplayParams");
        vxSetReferenceName((vx_reference)obj->disp_node, "DisplayNode");
    }

    #ifdef APP_DEBUG
    printf("app_tidl: Creating graph ... Done.\n");
    #endif

    return status;
}

static void app_run_task(void *app_var)
{
    AppObj *obj = (AppObj *)app_var;

    appPerfStatsCpuLoadResetAll();

    while(!obj->stop_task)
    {
        app_run_graph(obj);
    }
    obj->stop_task_done = 1;
}

static int32_t app_run_task_create(AppObj *obj)
{
    tivx_task_create_params_t params;
    int32_t status;

    tivxTaskSetDefaultCreateParams(&params);
    params.task_main = app_run_task;
    params.app_var = obj;

    obj->stop_task_done = 0;
    obj->stop_task = 0;

    status = tivxTaskCreate(&obj->task, &params);

    return status;
}

static void app_run_task_delete(AppObj *obj)
{
    while(obj->stop_task_done==0)
    {
         tivxTaskWaitMsecs(100);
    }

    tivxTaskDelete(&obj->task);
}

static char menu[] = {
    "\n"
    "\n ================================="
    "\n Demo : TIDL Object Classification"
    "\n ================================="
    "\n"
    "\n p: Print performance statistics"
    "\n"
    "\n x: Exit"
    "\n"
    "\n Enter Choice: "
};

static vx_status app_run_graph_interactive(AppObj *obj)
{
    vx_status status;
    uint32_t done = 0;
    char ch;
    FILE *fp;
    app_perf_point_t *perf_arr[1];

    status = app_run_task_create(obj);
    if(status!=0)
    {
        printf("app_tidl: ERROR: Unable to create task\n");
    }
    else
    {
        while(!done)
        {
            printf(menu);
            ch = getchar();
            printf("\n");

            switch(ch)
            {
                case 'p':
                    appPerfStatsPrintAll();
                    tivx_utils_graph_perf_print(obj->graph);
                    tivx_utils_graph_perf_print(obj->disp_graph);
                    appPerfPointPrint(&obj->fileio_perf);
                    appPerfPointPrint(&obj->draw_perf);
                    appPerfPointPrint(&obj->total_perf);
                    printf("\n");
                    appPerfPointPrintFPS(&obj->total_perf);
                    printf("\n");
                    break;
                case 'e':
                    perf_arr[0] = &obj->total_perf;
                    fp = appPerfStatsExportOpenFile(".", "app_tidl_datasheet");
                    if (NULL != fp)
                    {
                        appPerfStatsExportAll(fp, perf_arr, 1);
                        tivx_utils_graph_perf_export(fp, obj->graph);
                        appPerfStatsExportCloseFile(fp);
                        appPerfStatsResetAll();
                    }
                    else
                    {
                        printf("fp is null\n");
                    }
                    break;
                case 'x':
                    obj->stop_task = 1;
                    done = 1;
                    break;
            }
        }
        app_run_task_delete(obj);
    }
    return status;
}

static vx_status app_verify_graph(AppObj *obj)
{
    vx_status status = VX_SUCCESS;

    #ifdef APP_DEBUG
    printf("app_tidl: Verifying graph ... \n");
    #endif

    /* Verify the TIDL Graph */
    status = vxVerifyGraph(obj->graph);
    if(status!=VX_SUCCESS)
    {
        printf("app_tidl: ERROR: Verifying graph ... Failed !!!\n");
        return status;
    }

    #ifdef APP_DEBUG
    printf("app_tidl: Verifying graph ... Done.\n");
    #endif

    if ((vx_true_e == tivxIsTargetEnabled(TIVX_TARGET_DISPLAY1)) && (obj->display_option == 1))
    {
        #ifdef APP_DEBUG
        printf("app_tidl: Verifying display graph ... \n");
        #endif

        /* Verify the TIDL Graph */
        status = vxVerifyGraph(obj->disp_graph);
        if(status!=VX_SUCCESS)
        {
            printf("app_tidl: ERROR: Verifying display graph ... Failed !!!\n");
            return status;
        }

        #ifdef APP_DEBUG
        printf("app_tidl: Verifying display graph ... Done.\n");
        #endif
    }

    /* wait a while for prints to flush */
    tivxTaskWaitMsecs(100);

#if 0
    status = tivxExportGraphToDot(obj->graph,".", "vx_app_tidl");
    APP_ASSERT(status==VX_SUCCESS);
#endif
    return status;
}

static void app_run_graph_for_one_frame(AppObj *obj, char *curFileName)
{
    vx_char input_file_name[APP_MAX_FILE_PATH];
    vx_char output_file_name[APP_MAX_FILE_PATH];

    appPerfPointBegin(&obj->total_perf);

    snprintf(input_file_name, APP_MAX_FILE_PATH-1, "%s/%s",
          obj->input_file_path,
          curFileName
          );

    snprintf(output_file_name, APP_MAX_FILE_PATH-1, "%s/%s",
          obj->output_file_path,
          curFileName
          );

    #ifdef APP_DEBUG
    printf("app_tidl: Reading input file %s ... \n", input_file_name);
    #endif

    appPerfPointBegin(&obj->fileio_perf);

    /* Read input from file and poplulate the input tensors */
    readInput(obj, obj->context, obj->config, obj->input_tensors, &input_file_name[0]);

    appPerfPointEnd(&obj->fileio_perf);

    #ifdef APP_DEBUG
    printf("app_tidl: Reading input file %s ... Done.\n", input_file_name);
    #endif

    #ifdef APP_DEBUG
    printf("app_tidl: Running graph ... \n");
    #endif

    /* Execute the network */
    vxProcessGraph(obj->graph);

    #ifdef APP_DEBUG
    printf("app_tidl: Running graph ... Done.\n");
    #endif

    appPerfPointBegin(&obj->draw_perf);

    /* Display the output */
    displayOutput(obj, obj->config, obj->output_tensors, &output_file_name[0]);

    appPerfPointEnd(&obj->draw_perf);

    if ((vx_true_e == tivxIsTargetEnabled(TIVX_TARGET_DISPLAY1)) && (obj->display_option == 1)) {
          /* At this point, the output is ready to copy the updated buffer */
          vxCopyImagePatch(obj->disp_image,
              &obj->disp_rect,
              0,
              &obj->image_addr,
              (void *)obj->pDisplayBuf888,
              VX_WRITE_ONLY,
              VX_MEMORY_TYPE_HOST
            );

          #ifdef APP_DEBUG
          printf("app_tidl: Running display graph ... \n");
          #endif
          /* Execute the display graph */
          vxProcessGraph(obj->disp_graph);

          #ifdef APP_DEBUG
          printf("app_tidl: Running display graph ... Done.\n");
          #endif
    }

    appPerfPointEnd(&obj->total_perf);

#ifdef APP_TIDL_TRACE_DUMP
    tivx_utils_tidl_trace_write(obj->traceData, curFileName);
#endif

}

static vx_status app_run_graph(AppObj *obj)
{
    vx_status status = VX_SUCCESS;
    vx_char curFileName[APP_MAX_FILE_PATH];
    uint64_t cur_time;
    FILE* test_case_file;
    uint32_t cur_iteration;

    for(cur_iteration=0; cur_iteration<obj->num_iterations; cur_iteration++)
    {
        #if 1 //def APP_DEBUG
        if ((cur_iteration % 1) == 0)
          printf("app_tidl: Iteration %d of %d ... \n", cur_iteration, obj->num_iterations);
        #endif

        test_case_file =  fopen(obj->input_file_list,"r");
        if(test_case_file==NULL)
        {
            break;
        }
        while (fgets(curFileName, sizeof(curFileName), test_case_file))
        {
            curFileName[strlen(curFileName) - 1] = 0;

            cur_time = tivxPlatformGetTimeInUsecs();

            #ifdef x86_64
            printf("Classifying file %s ...", curFileName);
            #endif

            app_run_graph_for_one_frame(obj, curFileName);

            #ifdef x86_64
            printf("Done!\n");
            #endif

            cur_time = tivxPlatformGetTimeInUsecs() - cur_time;
            /* convert to msecs */
            cur_time = cur_time/1000;

            if(cur_time < obj->delay_in_msecs)
            {
                tivxTaskWaitMsecs(obj->delay_in_msecs - cur_time);
            }

            /* user asked to stop processing */
            if(obj->stop_task)
                break;
       }
       fclose(test_case_file);

       #ifdef APP_DEBUG
       printf("app_tidl: Iteration %d of %d ... Done.\n", cur_iteration, obj->num_iterations);
       #endif

       if(obj->stop_task)
           break;
    }

    return status;
}

static vx_user_data_object readConfig(AppObj *obj, vx_context context, char *config_file, uint32_t *num_input_tensors, uint32_t *num_output_tensors)
{
    vx_status status = VX_SUCCESS;

    tivxTIDLJ7Params  *tidlParams = NULL;
    sTIDL_IOBufDesc_t *ioBufDesc = NULL;
    vx_user_data_object   config = NULL;
    vx_uint32 capacity;
    vx_map_id map_id;

    FILE *fp_config;
    vx_size read_count;

    #ifdef APP_DEBUG
    printf("app_tidl: Reading config file %s ...\n", config_file);
    #endif

    fp_config = fopen(config_file, "rb");

    if(fp_config == NULL)
    {
        printf("app_tidl: ERROR: Unable to open IO config file %s \n", config_file);

        return NULL;
    }

    fseek(fp_config, 0, SEEK_END);
    capacity = ftell(fp_config);
    fseek(fp_config, 0, SEEK_SET);

    if( capacity != sizeof(sTIDL_IOBufDesc_t) )
    {
        printf("app_tidl: ERROR: Config file size (%d bytes) does not match size of sTIDL_IOBufDesc_t (%d bytes)\n", capacity, (vx_uint32)sizeof(sTIDL_IOBufDesc_t));
        return NULL;
    }


    /* Create a user struct type for handling config data*/
    config = vxCreateUserDataObject(context, "tivxTIDLJ7Params", sizeof(tivxTIDLJ7Params), NULL );

    status = vxGetStatus((vx_reference)config);

    if (VX_SUCCESS == status)
    {
        status = vxMapUserDataObject(config, 0, sizeof(tivxTIDLJ7Params), &map_id,
                            (void **)&tidlParams, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

        if (VX_SUCCESS == status)
        {
            if(tidlParams == NULL)
            {
              printf("app_tidl: ERROR: Map of config object failed\n");
              return NULL;
            }

            tivx_tidl_j7_params_init(tidlParams);

            ioBufDesc = (sTIDL_IOBufDesc_t *)&tidlParams->ioBufDesc;

            read_count = fread(ioBufDesc, capacity, 1, fp_config);
            if(read_count != 1)
            {
              printf("app_tidl: ERROR: Unable to read config file\n");
            }
            fclose(fp_config);

            memcpy(&obj->ioBufDesc, ioBufDesc, capacity);

            // printf("*** obj->ioBufDesc = %d\n", obj->ioBufDesc); // ?what in this

            *num_input_tensors  = ioBufDesc->numInputBuf;
            *num_output_tensors = ioBufDesc->numOutputBuf;

            vxUnmapUserDataObject(config, map_id);
        }
    }

    #ifdef APP_DEBUG
    printf("app_tidl: Reading config file %s ... Done. %d bytes\n", config_file, (uint32_t)capacity);
    printf("app_tidl: Tensors, input = %d, output = %d\n", *num_input_tensors, *num_output_tensors);
    #endif

    return config;
}

static vx_user_data_object readNetwork(vx_context context, char *network_file)
{
    vx_status status;

    vx_user_data_object  network;
    vx_map_id  map_id;
    vx_uint32  capacity;
    void      *network_buffer = NULL;
    vx_size read_count;

    FILE *fp_network;

    #ifdef APP_DEBUG
    printf("app_tidl: Reading network file %s ... \n", network_file);
    #endif

    fp_network = fopen(network_file, "rb");

    if(fp_network == NULL)
    {
        printf("app_tidl: ERROR: Unable to open network file %s \n", network_file);

        return NULL;
    }
    fseek(fp_network, 0, SEEK_END);
    capacity = ftell(fp_network);
    fseek(fp_network, 0, SEEK_SET);

    network = vxCreateUserDataObject(context, "TIDL_network", capacity, NULL );

    status = vxGetStatus((vx_reference)network);

    if (VX_SUCCESS == status)
    {
        status = vxMapUserDataObject(network, 0, capacity, &map_id,
                        (void **)&network_buffer, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

        if (VX_SUCCESS == status)
        {

            if(network_buffer) {
                read_count = fread(network_buffer, capacity, 1, fp_network);
                if(read_count != 1)
                {
                    printf("app_tidl: ERROR: Unable to read network file\n");
                }
                fclose(fp_network);
            } else {
                printf("app_tidl: ERROR: Unable to allocate memory for reading network file of %d bytes\n", capacity);
            }

            vxUnmapUserDataObject(network, map_id);
        }
    }



    #ifdef APP_DEBUG
    printf("app_tidl: Reading network file %s ... Done. %d bytes\n", network_file, (uint32_t)capacity);
    #endif

    return network;
}

static vx_user_data_object setCreateParams(vx_context context)
{
    vx_status status;

    vx_user_data_object  createParams;
    vx_map_id  map_id;
    vx_uint32  capacity;
    void *createParams_buffer = NULL;

    capacity = sizeof(TIDL_CreateParams);
    createParams = vxCreateUserDataObject(context, "TIDL_CreateParams", capacity, NULL );

    status = vxGetStatus((vx_reference)createParams);

    if (VX_SUCCESS == status)
    {
        status = vxMapUserDataObject(createParams, 0, capacity, &map_id,
                        (void **)&createParams_buffer, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

        if (VX_SUCCESS == status)
        {
            if(createParams_buffer)
            {
                TIDL_CreateParams *prms = createParams_buffer;

                TIDL_createParamsInit(prms);

                prms->isInbufsPaded                 = 1;
                prms->quantRangeExpansionFactor     = 1.0;
                prms->quantRangeUpdateFactor        = 0.0;
#ifdef APP_TIDL_TRACE_DUMP
                prms->traceLogLevel                 = 1;
                prms->traceWriteLevel               = 1;
#else
                prms->traceLogLevel                 = 0;
                prms->traceWriteLevel               = 0;
#endif

            }
            else
            {
                printf("app_tidl: ERROR: Unable to allocate memory for create time params! %d bytes\n", capacity);
            }

            vxUnmapUserDataObject(createParams, map_id);
        }
    }

    return createParams;
}

static vx_user_data_object setInArgs(vx_context context)
{
    vx_status status;

    vx_user_data_object  inArgs;
    vx_map_id  map_id;
    vx_uint32  capacity;
    void *inArgs_buffer = NULL;

    capacity = sizeof(TIDL_InArgs);
    inArgs = vxCreateUserDataObject(context, "TIDL_InArgs", capacity, NULL );

    status = vxGetStatus((vx_reference)inArgs);

    if (VX_SUCCESS == status)
    {
        status = vxMapUserDataObject(inArgs, 0, capacity, &map_id,
                        (void **)&inArgs_buffer, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

        if (VX_SUCCESS == status)
        {
            if(inArgs_buffer)
            {
              TIDL_InArgs *prms = inArgs_buffer;
              prms->iVisionInArgs.size         = sizeof(TIDL_InArgs);
              prms->iVisionInArgs.subFrameInfo = 0;
            }
            else
            {
                printf("app_tidl: Unable to allocate memory for inArgs! %d bytes\n", capacity);
            }

            vxUnmapUserDataObject(inArgs, map_id);
        }
    }

    return inArgs;
}

static vx_user_data_object setOutArgs(vx_context context)
{
    vx_status status;

    vx_user_data_object  outArgs;
    vx_map_id  map_id;
    vx_uint32  capacity;
    void *outArgs_buffer = NULL;

    capacity = sizeof(TIDL_outArgs);
    outArgs = vxCreateUserDataObject(context, "TIDL_outArgs", capacity, NULL );

    status = vxGetStatus((vx_reference)outArgs);

    if (VX_SUCCESS == status)
    {
        status = vxMapUserDataObject(outArgs, 0, capacity, &map_id,
                        (void **)&outArgs_buffer, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);

        if (VX_SUCCESS == status)
        {
            if(outArgs_buffer)
            {
              TIDL_outArgs *prms = outArgs_buffer;
              prms->iVisionOutArgs.size         = sizeof(TIDL_outArgs);
            }
            else
            {
                printf("app_tidl: Unable to allocate memory for outArgs! %d bytes\n", capacity);
            }

            vxUnmapUserDataObject(outArgs, map_id);
        }
    }

    return outArgs;
}

static void createInputTensors(AppObj *obj, vx_context context, vx_user_data_object config, vx_tensor *input_tensors)
{
    vx_size   input_sizes[APP_MAX_TENSOR_DIMS];

    uint32_t id;

    sTIDL_IOBufDesc_t *ioBufDesc = &obj->ioBufDesc;

    for(id = 0; id < ioBufDesc->numInputBuf; id++) {

        input_sizes[0] = ioBufDesc->inWidth[id]  + ioBufDesc->inPadL[id] + ioBufDesc->inPadR[id];
        input_sizes[1] = ioBufDesc->inHeight[id] + ioBufDesc->inPadT[id] + ioBufDesc->inPadB[id];
        input_sizes[2] = ioBufDesc->inNumChannels[id];

        input_tensors[id] = vxCreateTensor(context, 3, input_sizes, VX_TYPE_UINT8, 0);
    }

    return;
}

static void createOutputTensors(AppObj *obj, vx_context context, vx_user_data_object config, vx_tensor *output_tensors)
{
    vx_size output_sizes[APP_MAX_TENSOR_DIMS];

    uint32_t id;

    sTIDL_IOBufDesc_t *ioBufDesc = &obj->ioBufDesc;

    for(id = 0; id < ioBufDesc->numOutputBuf; id++) {

        output_sizes[0] = ioBufDesc->outWidth[id]  + ioBufDesc->outPadL[id] + ioBufDesc->outPadR[id];
        output_sizes[1] = ioBufDesc->outHeight[id] + ioBufDesc->outPadT[id] + ioBufDesc->outPadB[id];
        output_sizes[2] = ioBufDesc->outNumChannels[id];

        output_tensors[id] = vxCreateTensor(context, 3, output_sizes, VX_TYPE_FLOAT32, 0);
    }


    return;
}
static void resizeImage(uint8_t *inImg, uint8_t *outImg, uint32_t inWidth, uint32_t inHeight, uint32_t inStride, uint32_t outWidth, uint32_t outHeight, uint32_t outStride)
{

  float xScale = (inWidth * 1.0f) / (outWidth * 1.0f);
  float yScale = (inHeight * 1.0f) / (outHeight * 1.0f);

  uint8_t border_constant_value = 128;

  int32_t srcOffsetX = 0;
  int32_t srcOffsetY = 0;
  int32_t dstOffsetX = 0;
  int32_t dstOffsetY = 0;

  int32_t x, y, ch;

  for( ch = 0; ch < 3; ch++) {
    uint8_t *pIn  = inImg + (ch * inStride * inHeight);
    uint8_t *pOut = outImg + (ch * outStride * outHeight);
    for( y = 0; y < outHeight; y++ ) {
      for( x = 0; x < outWidth; x++ ) {

        /* Apply scale factors to find input pixel for each output pixel */
        float src_x_f = ((float)(x+dstOffsetX) + 0.5f)*xScale - 0.5f;
        float src_y_f = ((float)(y+dstOffsetY) + 0.5f)*yScale - 0.5f;

        float xf = floorf(src_x_f);
        float yf = floorf(src_y_f);
        float dx = src_x_f - xf;
        float dy = src_y_f - yf;
        float a[4];

        int32_t src_x = (int32_t)(xf) - srcOffsetX;
        int32_t src_y = (int32_t)(yf) - srcOffsetY;

        uint8_t tl = 0;
        uint8_t tr = 0;
        uint8_t bl = 0;
        uint8_t br = 0;

        a[0] = (1.0f - dx) * (1.0f - dy);
        a[1] = (1.0f - dx) * (dy);
        a[2] = (dx)* (1.0f - dy);
        a[3] = (dx)* (dy);

        tl = (src_x < 0 || src_y < 0 || src_x > (inWidth-1) || src_y > (inHeight-1) ) ?
             border_constant_value :
             pIn[(src_y*inStride) + src_x];
        tr = ((src_x+1) < 0 || src_y < 0 || (src_x+1) > (inWidth-1) || src_y > (inHeight-1) ) ?
             border_constant_value :
             pIn[(src_y*inStride) + src_x + 1];
        bl = (src_x < 0 || (src_y+1) < 0 || src_x > (inWidth-1) || (src_y+1) > (inHeight-1) ) ?
             border_constant_value :
             pIn[((src_y+1)*inStride) + src_x];
        br = ((src_x+1) < 0 || (src_y+1) < 0 || (src_x+1) > (inWidth-1) || (src_y+1) > (inHeight-1) ) ?
             border_constant_value :
             pIn[((src_y+1)*inStride) + src_x + 1];

        pOut[(y*outStride) + x] = (uint8_t)(a[0]*tl + a[2]*tr + a[1]*bl + a[3]*br + 0.5f);
      }
    }
  }

  return;
}

static vx_status readInput(AppObj *obj, vx_context context, vx_user_data_object config, vx_tensor *input_tensors, char *input_file)
{
    vx_status status = VX_SUCCESS;

    void      *input_buffer = NULL;
    int32_t    capacity;
    uint32_t   id;

    vx_map_id map_id_input;

    vx_size    start[APP_MAX_TENSOR_DIMS];
    vx_size    input_strides[APP_MAX_TENSOR_DIMS];
    vx_size    input_sizes[APP_MAX_TENSOR_DIMS];

    sTIDL_IOBufDesc_t *ioBufDesc = &obj->ioBufDesc;

    for(id = 0; id < ioBufDesc->numInputBuf; id++)
    {
        input_sizes[0] = ioBufDesc->inWidth[id]  + ioBufDesc->inPadL[id] + ioBufDesc->inPadR[id];
        input_sizes[1] = ioBufDesc->inHeight[id] + ioBufDesc->inPadT[id] + ioBufDesc->inPadB[id];
        input_sizes[2] = ioBufDesc->inNumChannels[id];

        capacity = input_sizes[0] * input_sizes[1] * input_sizes[2];

        start[0] = start[1] = start[2] = 0;

        input_strides[0] = 1;
        input_strides[1] = input_sizes[0];
        input_strides[2] = input_sizes[1] * input_strides[1];

        #ifdef APP_DEBUG
        printf(" input_sizes[0] = %d, dim = %d padL = %d padR = %d\n", (uint32_t)input_sizes[0], ioBufDesc->inWidth[id], ioBufDesc->inPadL[id], ioBufDesc->inPadR[id]);
        printf(" input_sizes[1] = %d, dim = %d padL = %d padR = %d\n", (uint32_t)input_sizes[1], ioBufDesc->inHeight[id], ioBufDesc->inPadT[id], ioBufDesc->inPadB[id]);
        printf(" input_sizes[2] = %d, dim = %d \n", (uint32_t)input_sizes[2], ioBufDesc->inNumChannels[id]);
        #endif

        status = tivxMapTensorPatch(input_tensors[id], 3, start, input_sizes, &map_id_input, input_strides, &input_buffer, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

        if (VX_SUCCESS == status)
        {
            vx_df_image df_image;
            void *data_ptr = NULL;
            vx_uint32 img_width;
            vx_uint32 img_height;
            vx_uint32 img_stride;
            vx_int32 start_offset;
            vx_uint8 *pR;
            vx_uint8 *pG;
            vx_uint8 *pB;
            vx_uint8 *pData;
            vx_uint8 *pInPlanes;
            vx_uint8 *pOutPlanes;
            vx_int32 i, j;

            #ifdef APP_DEBUG
            printf("app_tidl: Reading bmp file ...\n" );
            #endif

            status = tivx_utils_bmp_file_read(
                        input_file,
                        vx_false_e,
                        &img_width, &img_height, &img_stride, &df_image, &data_ptr,
                        &obj->bmp_context);

            if(status != VX_SUCCESS)
            {
                printf("app_tidl: Reading bmp file ... Failed !!!\n" );
            }
            else
            {
                #ifdef APP_DEBUG
                printf("app_tidl: Reading bmp file ... Done.\n" );
                #endif

                #ifdef APP_DEBUG
                printf("app_tidl: Image Post processing for image of size %d x %d (pitch = %d bytes)...\n", img_width, img_height, img_stride);
                #endif

                #ifdef APP_DEBUG
                printf("app_tidl: Deinterleaving data ...\n");
                #endif

                /* Save image params which are used in displayOutput function */
                obj->img_width   = img_width;
                obj->img_height  = img_height;
                obj->img_stride  = img_stride;
                obj->data_ptr    = data_ptr;
                obj->df_image    = df_image;

                /* Reset the input buffer, this will take care of padding requirement for TIDL */
                memset(input_buffer, 0, capacity);

                start_offset = (0 * input_strides[2]) + (ioBufDesc->inPadT[id] * input_strides[1]) + ioBufDesc->inPadL[id];
                pB = (vx_uint8 *)input_buffer + start_offset;

                start_offset = (1 * input_strides[2]) + (ioBufDesc->inPadT[id] * input_strides[1]) + ioBufDesc->inPadL[id];
                pG = (vx_uint8 *)input_buffer + start_offset;

                start_offset = (2 * input_strides[2]) + (ioBufDesc->inPadT[id] * input_strides[1]) + ioBufDesc->inPadL[id];
                pR = (vx_uint8 *)input_buffer + start_offset;

                pInPlanes = (vx_uint8 *)obj->pInPlanes;
                pOutPlanes = (vx_uint8 *)obj->pOutPlanes;

                /* Deinterleave RGB to planar */
                uint32_t chOffset = img_width * img_height;
                for(i = 0; i < img_height; i ++)
                {
                    pData = (vx_uint8 *)data_ptr + (i * img_stride);
                    for(j = 0; j < img_width; j ++)
                    {
                        pInPlanes[(0 * chOffset) + (i * img_width) + j] = *pData++;
                        pInPlanes[(1 * chOffset) + (i * img_width) + j] = *pData++;
                        pInPlanes[(2 * chOffset) + (i * img_width) + j] = *pData++;
                    }
                }

                #ifdef APP_DEBUG
                printf("app_tidl: Resizing image ...\n");
                #endif

                /* Resize image to network input resolution */
                resizeImage(pInPlanes, pOutPlanes, img_width, img_height, img_width, 256, 256, 256);

                //! *********************
                //syz save resizing image
                char * output_file;
                output_file = "intermediatefile.bin";
                FILE *fp = NULL;
                if(NULL == (fp = fopen(output_file, "wb")))
                {
                    return -1;
                }

                int  length = sizeof(pOutPlanes);
                for(int i = 0; i < length; i++)
                {
                    fwrite(pOutPlanes, sizeof(vx_uint8),1, fp);
                }
                fclose(fp);


                

                // tivx_utils_bmp_file_write(output_file, 256, 256, (256*3), obj->df_image, obj->pDisplayBuf888);

                //!***********************

                #ifdef APP_DEBUG
                printf("app_tidl: Rearranging data ...\n");
                #endif

                chOffset = 256*256;

                /* Write image in network required format */
                for(i = 0; i < ioBufDesc->inHeight[id]; i++)
                {
                    //Center crop 224x224 image from 256x256 resized input
                    uint32_t offset = ((16 + i) * 256) + 16;

                    for(j = 0; j < ioBufDesc->inWidth[id]; j++)
                    {
                        pR[j] = pOutPlanes[(0 * chOffset) + offset + j];
                        pG[j] = pOutPlanes[(1 * chOffset) + offset + j];
                        pB[j] = pOutPlanes[(2 * chOffset) + offset + j];
                    }
                    pR += input_strides[1];
                    pG += input_strides[1];
                    pB += input_strides[1];
                }

                #ifdef APP_DEBUG
                printf("app_tidl: Image Post processing ... Done.\n");
                #endif
            }

            tivxUnmapTensorPatch(input_tensors[id], map_id_input);
        }
    }

    return status;
}

static void displayOutput(AppObj *obj, vx_user_data_object config, vx_tensor *output_tensors, char *output_file)
{
    vx_status status = VX_SUCCESS;

    vx_size output_sizes[APP_MAX_TENSOR_DIMS];

    int32_t id, i, j;

    sTIDL_IOBufDesc_t *ioBufDesc;

    Draw2D_FontPrm sClassPrm;

    ioBufDesc = &obj->ioBufDesc;

    for(id = 0; id < ioBufDesc->numOutputBuf; id++)
    {
        output_sizes[0] = ioBufDesc->outWidth[id]  + ioBufDesc->outPadL[id] + ioBufDesc->outPadR[id];
        output_sizes[1] = ioBufDesc->outHeight[id] + ioBufDesc->outPadT[id] + ioBufDesc->outPadB[id];
        output_sizes[2] = ioBufDesc->outNumChannels[id];

        status = vxGetStatus((vx_reference)output_tensors[id]);

        if (VX_SUCCESS == status)
        {
            void *output_buffer;

            vx_map_id map_id_output;

            vx_size output_strides[APP_MAX_TENSOR_DIMS];
            vx_size start[APP_MAX_TENSOR_DIMS];

            start[0] = start[1] = start[2] = start[3] = 0;

            output_strides[0] = 1;
            output_strides[1] = output_sizes[0];
            output_strides[2] = output_sizes[1] * output_strides[1];

            tivxMapTensorPatch(output_tensors[id], 3, start, output_sizes, &map_id_output, output_strides, &output_buffer, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

            {
                float *pOut;
                float score[5];
                vx_uint32 classid[5];

                #ifdef APP_DEBUG
                printf("app_tidl: Finding top-5 ... \n");
                #endif


                pOut = (float *)output_buffer + (ioBufDesc->outPadT[id] * output_sizes[0]) + ioBufDesc->outPadL[id];

                for(i = 0; i < 5; i++)
                {
                  score[i] = FLT_MIN;
                  classid[i] = (vx_uint32)-1;

                  for(j = 0; j < ioBufDesc->outWidth[id]; j++)
                  {
                    if(pOut[j] > score[i])
                    {
                      score[i] = pOut[j];
                      classid[i] = j;
                    }
                  }
                  if(classid[i] < ioBufDesc->outWidth[id] )
                  {
                    pOut[classid[i]] = FLT_MIN;
                  }
                  else
                  {
                    classid[i] = 0; /* invalid class ID, ideally it should not reach here */
                  }
                }

                #ifdef APP_DEBUG
                printf("app_tidl: Finding top-5 ... Done \n");
                #endif

                #ifdef APP_DEBUG
                printf("app_tidl: Image classification Top-5 results: \n");
                for(i = 0; i < 5; i++)
                {
                  printf("app_tidl:  %s, class-id: %d, score: %f \n", (char *)&imgnet_labels[classid[i]], classid[i], score[i]);
                }
                #endif

                {
                    Draw2D_clearRegion(obj->pHndl, (DISPLAY_WIDTH/2) + 40, 200, 600, 300);
                    sClassPrm.fontIdx = 1;
                    for(i = 0; i < 5; i++)
                    {
                      Draw2D_drawString(obj->pHndl, (DISPLAY_WIDTH/2) + 40, 200 + (i * 40), (char *)&imgnet_labels[classid[i]], &sClassPrm);
                    }
                }
            }

            tivxUnmapTensorPatch(output_tensors[id], map_id_output);
        }
    }



    {
      /* Convert RGB565 to RGB888 before writing output */
      for(i = 0, j = 0; i < (DISPLAY_WIDTH * DISPLAY_HEIGHT); i++)
      {
        uint16_t RGB_565_val = obj->pDisplayBuf565[i];

        obj->pDisplayBuf888[j + 0] = (RGB_565_val & 0x1F) << 3;
        obj->pDisplayBuf888[j + 1] = ((RGB_565_val >> 5) & 0x3F) << 2;
        obj->pDisplayBuf888[j + 2] = ((RGB_565_val >> 11) & 0x1F) << 3;

        j  += 3;
      }

      if((obj->img_width <= (DISPLAY_WIDTH/2)) && (obj->img_height <= (DISPLAY_HEIGHT - 200)))
      {
         uint32_t startX = ((DISPLAY_WIDTH/2) / 2) - (obj->img_width / 2);
         uint32_t startY = ((DISPLAY_HEIGHT/2)  / 2) - (obj->img_height / 2);
         uint32_t imgOffset = 200;

         for(i = 0; i < obj->img_height; i++)
         {
            uint8_t *pOut = &obj->pDisplayBuf888[((imgOffset + startY + i) * DISPLAY_WIDTH * 3) + (startX * 3)];
            uint8_t *pIn  = obj->data_ptr + (i * obj->img_stride);

            for(j = 0; j < obj->img_width; j++)
            {
              *pOut++ = *pIn++;
              *pOut++ = *pIn++;
              *pOut++ = *pIn++;
            }
         }
      }
    }


    if (obj->display_option == 0)
    {
        tivx_utils_bmp_file_write(output_file, DISPLAY_WIDTH, DISPLAY_HEIGHT, (DISPLAY_WIDTH * 3), obj->df_image, obj->pDisplayBuf888);
    }

    /* Release the bmp buffer created in readInput() */
    tivx_utils_bmp_file_read_release(obj->bmp_context);

    #ifdef APP_DEBUG
    printf("app_tidl: Showing output ... Done.\n");
    #endif
}
