#include "yolo_detect.h"

inline const char *get_type_string(rknn_tensor_type type)
{
    switch (type)
    {
    case RKNN_TENSOR_FLOAT32:
        return "FP32";
    case RKNN_TENSOR_FLOAT16:
        return "FP16";
    case RKNN_TENSOR_INT8:
        return "INT8";
    case RKNN_TENSOR_UINT8:
        return "UINT8";
    case RKNN_TENSOR_INT16:
        return "INT16";
    default:
        return "UNKNOW";
    }
}

inline const char *get_qnt_type_string(rknn_tensor_qnt_type type)
{
    switch (type)
    {
    case RKNN_TENSOR_QNT_NONE:
        return "NONE";
    case RKNN_TENSOR_QNT_DFP:
        return "DFP";
    case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:
        return "AFFINE";
    default:
        return "UNKNOW";
    }
}

inline const char *get_format_string(rknn_tensor_format fmt)
{
    switch (fmt)
    {
    case RKNN_TENSOR_NCHW:
        return "NCHW";
    case RKNN_TENSOR_NHWC:
        return "NHWC";
    default:
        return "UNKNOW";
    }
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[3], attr->dims[2], attr->dims[1], attr->dims[0],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static int saveFloat(const char *file_name, float *output, int element_size)
{
    FILE *fp;
    fp = fopen(file_name, "w");
    for (int i = 0; i < element_size; i++)
    {
        fprintf(fp, "%.6f\n", output[i]);
    }
    fclose(fp);
    return 0;
}

static rknn_context ctx;

rknn_tensor_attr input_attrs[1];
rknn_tensor_attr output_attrs[3];
rknn_input inputs[1];
unsigned char *model_data = NULL;
int init_model(const char *model_name)
{
    int ret;

    printf("Loading model...\n");
    int model_data_size = 0;
    model_data = load_model(model_name, &model_data_size);
    ret = rknn_init(&ctx, model_data, model_data_size, 0);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    memset(input_attrs, 0, sizeof(input_attrs));

    input_attrs[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[0]), sizeof(rknn_tensor_attr));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("Here isth input attrs\n");

    dump_tensor_attr(&(input_attrs[0]));

    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < 3; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs[i]));
        if (output_attrs[i].qnt_type != RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC || output_attrs[i].type != RKNN_TENSOR_UINT8)
        {
            fprintf(stderr,
                    "The Demo required for a Affine asymmetric u8 quantized rknn model, but output quant type is %s, output "
                    "data type is %s\n",
                    get_qnt_type_string(output_attrs[i].qnt_type), get_type_string(output_attrs[i].type));
            return -1;
        }
    }

    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = 640 * 640 * 3;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    return 0;
}

void detect(void *img_data, detect_results_t *detect_result_group)
{
    int ret;
    struct timeval start_time, stop_time;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;

    inputs[0].buf = img_data;
    rknn_inputs_set(ctx, 1, inputs);

    rknn_output outputs[3];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 3; i++)
    {
        outputs[i].want_float = 0;
    }
    gettimeofday(&start_time, NULL);
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, 3, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    float scale_w = 1.0f;
    float scale_h = 1.0f;

    std::vector<float> out_scales;
    std::vector<uint32_t> out_zps;
    for (int i = 0; i < 3; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    post_process((uint8_t *)outputs[0].buf, (uint8_t *)outputs[1].buf, (uint8_t *)outputs[2].buf, 640, 640,
                 box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, detect_result_group);

    char text[256];
    const unsigned char blue[] = {0, 0, 255};
    const unsigned char white[] = {255, 255, 255};
    for (int i = 0; i < detect_result_group->count; i++)
    {
        detect_rslt_t *det_result = &detect_result_group->results[i];
        sprintf(text, "%s %.2f", det_result->name, det_result->prop);
        printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
               det_result->box.right, det_result->box.bottom, det_result->prop);
        printf("class id:%d", det_result->class_index);
        
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;
    }

    rknn_outputs_release(ctx, 3, outputs);
}

void release_model()
{
    rknn_destroy(ctx);

    if (model_data)
    {
        free(model_data);
    }
}
