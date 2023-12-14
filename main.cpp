#include "cpu.h"
#include "net.h"
#include "gpu.h"
#include "benchmark.h"

#include <ctime>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;

const static int warmup_N = 50;
const static int N = 100;

void setup_gpu(bool &use_vulkan_compute){
    use_vulkan_compute = true;
    g_vkdev = ncnn::get_gpu_device(0);

    g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
    g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
}

void save_res(const ncnn::Mat out,const char *s){
    cv::Mat res(out.h, out.w, CV_8UC3);
    out.to_pixels(res.data, ncnn::Mat::PIXEL_BGR2RGB);
    cv::imwrite (s, res);
}

void benchmark(const char* model,const char* model_bin,const char* input,const bool gpu_flag){
    bool use_vulkan_compute = false;
    // 180*320
    cv::Mat img = cv::imread(input);
    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
    
    save_res(in,"./res_in.png");

    int num_threads = ncnn::get_physical_big_cpu_count();
    printf("num_threads %d\n",num_threads);

    // ncnn::set_cpu_powersave (int) 绑定大核或小核
    // 0: 全部
    // 1: 小核
    // 2: 大核
    ncnn::set_cpu_powersave(0);
    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    g_blob_pool_allocator.set_size_compare_ratio(0.f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.f);
    // use vulkan compute
    if (gpu_flag && ncnn::get_gpu_count() != 0 ){
        printf("vulkan compute\n");
        setup_gpu(use_vulkan_compute); 
    }

    ncnn::Net net;

    net.opt.num_threads = num_threads;
    net.opt.lightmode = true;
    net.opt.blob_allocator = &g_blob_pool_allocator;
    net.opt.workspace_allocator = &g_workspace_pool_allocator;

    net.opt.blob_vkallocator = g_blob_vkallocator;
    net.opt.workspace_vkallocator = g_blob_vkallocator;
    net.opt.staging_vkallocator = g_staging_vkallocator;

    net.opt.use_winograd_convolution = true;
    net.opt.use_sgemm_convolution = true;
    
    net.opt.use_vulkan_compute = use_vulkan_compute;

    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    // 影响精度, 但快
    net.opt.use_fp16_arithmetic = true;

    // vulkan not support! work for arm cpu?
    net.opt.use_int8_storage = true;
    net.opt.use_int8_arithmetic = true;
    net.opt.use_int8_inference = true;

    net.opt.use_packing_layout = true;
    net.opt.use_shader_pack8 = false;
    net.opt.use_image_storage = false;

    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();
    
    if (use_vulkan_compute){
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
        net.set_vulkan_device(g_vkdev);
    }
    
    net.load_param(model);
    net.load_model(model_bin);
    
    
    printf("Warm up start\n");
    for (int i=0; i<warmup_N; i++){
        ncnn::Extractor ex = net.create_extractor();
        ncnn::Mat _in = in;
        ex.input("in0", _in);
        ncnn::Mat _out;
        ex.extract("out0", _out);
    }
    printf("Warm up end\n");

    double dr = 0.f;
    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    int min_idx = 0;
    int max_idx = 0;

    for (int i=0; i < N; i++){
        
        double start = ncnn::get_current_time();
        ncnn::Extractor ex = net.create_extractor();
        ncnn::Mat _in = in;
        ex.input("in0", _in);
        ncnn::Mat _out;
        ex.extract("out0", _out);
        // cv::Mat res(_out.h, _out.w, CV_8UC3);
        // _out.to_pixels(res.data, ncnn::Mat::PIXEL_BGR2RGB);
        // cv::imwrite("./res_" + std::to_string(i) +".png", res);
        double end = ncnn::get_current_time();
        double time = end - start;

        if (time_min > time){
            time_min = time;
            min_idx = i;
        }

        if (time_max < time){
            time_max = time;
            max_idx = i;
        }

        // time_min = std::min(time_min, time);
        // time_max = std::max(time_max, time);

        dr += end - start;

        if (i == N - 1){
            save_res(_out,"./res.png");
        }
    }
    printf("latency avg %fms , max (%d)%fms , min (%d)%fms\n",dr / N * 1.f,max_idx,time_max,min_idx,time_min);

}

int main(int argc,char** argv)
{   
    char* model = 0;
    char* model_bin = 0;
    char* input = 0;
    bool gpu_flag = false;
   
    for(int i = 1;i < argc;i++){
        char* kv = argv[i];

        char* eqs = strchr(kv, '=');
        if (eqs == NULL)
        {
            fprintf(stderr, "unrecognized arg %s\n", kv);
            continue;
        }

        // split k v
        eqs[0] = '\0';
        const char* key = kv;
        char* value = eqs + 1;

        if (strcmp(key, "param") == 0)
            model = value;
        if (strcmp(key, "bin") == 0)
            model_bin = value;
        if (strcmp(key, "input") == 0)
            input = value;
        if (strcmp(key, "gpu") == 0)
            gpu_flag = atoi(value);
    }

    benchmark(model,model_bin,input,gpu_flag);
    
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
    ncnn::destroy_gpu_instance();

    return 0;
}