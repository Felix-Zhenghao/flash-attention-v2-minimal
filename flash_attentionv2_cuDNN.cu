/*cuDNN by A.K. for me to learn*/

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

// CUDA & cuDNN setup
static bool first_run_validation = true; // always run e.g. permute on 1st run

#ifdef ENABLE_CUDNN
#include <cudnn_frontend.h>
namespace fe = cudnn_frontend;

static cudnnHandle_t cudnn_handle;
static size_t cudnn_workspace_size = 0; // dynamically allocated as needed (up to 256MiB!)
static void* cudnn_workspace = NULL;

#endif // ENABLE_CUDNN

#ifdef ENABLE_CUDNN
using graph_tensors_fwd = std::tuple<std::shared_ptr<fe::graph::Graph>,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Q,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // K,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // V,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // Attn_scale,
                                     std::shared_ptr<fe::graph::Tensor_attributes>,  // O
                                     std::shared_ptr<fe::graph::Tensor_attributes>>; // Stats

// Need a cache because graph->build_operation_graph() is slow but everything else seems fast
using cache_type_fwd = std::unordered_map<std::size_t, graph_tensors_fwd>;

// Loosely based on cuDNN frontend samples functions and massively simplified
template <typename... Args>
auto lookup_cache_or_build_graph_fwd(Args... args) {
    static cache_type_fwd user_maintained_cache_fwd;
    auto [B, H, T, HS, is_inference_only] = std::make_tuple(args...);

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(CUDNN_16BIT)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

    // QKV is (B, T, 3, NH, HS) which cuDNN can handle directly without an external permute
    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Q")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T,  HS, 3 * H * HS, 1}));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("K")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("V")
                               .set_dim({B, H, T, HS})
                               .set_stride({3 * H * HS * T, HS, 3 * H * HS, 1}));
    auto attn_scale = graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("attn_scale")
                                .set_dim({1, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_is_pass_by_value(true)
                                .set_data_type(fe::DataType_t::FLOAT));

    auto sdpa_options = fe::graph::SDPA_attributes().set_name("flash_attention");
    sdpa_options.set_is_inference(is_inference_only);
    sdpa_options.set_attn_scale(attn_scale);
    sdpa_options.set_causal_mask(true);

    // Create the graph operation and get the output tensors back
    auto [O, stats] = graph->sdpa(Q, K, V, sdpa_options);

    // Output is (B, T, NH, HS) BF16/FP16 and stats for backward pass is (B, NH, T) FP32
    O->set_output(true).set_dim({B, H, T, HS}).set_stride({H * HS * T, HS, H * HS, 1});

    assert(stats == nullptr || is_inference_only == false);
    if (is_inference_only == false) {
        stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)
                               .set_dim({B, H, T, 1})
                               .set_stride({H * T, T, 1, 1});
    }

    assert(graph->validate().is_good());
    auto key = graph->key();
    auto it = user_maintained_cache_fwd.find(key);
    if (it != user_maintained_cache_fwd.end()) {
        return it->second;
    }

    // Build the operation graph and execution part (this is the VERY SLOW PART)
    assert(graph->build_operation_graph(cudnn_handle).is_good());
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    assert(graph->check_support(cudnn_handle).is_good());
    assert(graph->build_plans(cudnn_handle).is_good());

    auto tuple = std::make_tuple(graph, Q, K, V, attn_scale, O, stats);
    user_maintained_cache_fwd.insert({key, tuple});
    return tuple;
}

// Used on first run only so we can validate against the CPU results
__global__ void fp32_to_lowp_kernel(floatX* out, const float* inp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = (floatX)inp[idx];
}

__global__ void lowp_to_fp32_kernel(const floatX* inp, float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = (float)inp[idx];
}

void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             float* in_fp32,  // fp32 input
                             float* out_fp32, // fp32 output for validation
                             int B, int T, int C, int NH) {
    static bool first_run_validation = true;
    int HS = C / NH; // number of features per head
    bool is_inference_only = (stats == nullptr);

    // Convert from FP32 to FP16/BF16 on 1st run to get correct results
    const int block_size = 64; // smallest full occupancy block size on modern GPUs
    if (first_run_validation) {
        int total_threads = B * T * C * 3;
        assert(total_threads % block_size == 0);
        int num_blocks = total_threads / block_size;
        fp32_to_lowp_kernel<<<num_blocks, block_size>>>(inp, in_fp32);
    }

    // Get graph and tensors from cache (or generate it on first use)
    auto [graph, Q, K, V, attn_scale, O, softmax_stats] =
        lookup_cache_or_build_graph_fwd(B, NH, T, HS, is_inference_only);

    // Prepare all the tensor pointers for executing the graph
    void* devPtrQ = inp;
    void* devPtrK = (inp + C);
    void* devPtrV = (inp + 2 * C);
    float attn_scale_cpu = 1.0 / sqrtf(HS);
    void* devPtrO = out;

    // Build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ}, {K, devPtrK}, {V, devPtrV}, {attn_scale, &attn_scale_cpu}, {O, devPtrO}};

    // Add the stats tensor unless we are only doing inference (only needed for backward pass)
    if (is_inference_only == false) {
        variant_pack[softmax_stats] = stats;
    }

    // Reallocate the workspace if the required size is greater than the current workspace
    // By default, cuDNN uses up to 256MiB of workspace, so we don't want to just allocate the maximum
    if (graph->get_workspace_size() > cudnn_workspace_size) {
        if (cudnn_workspace_size > 0) {
            cudaCheck(cudaFree(cudnn_workspace));
        }
        cudnn_workspace_size = graph->get_workspace_size();
        cudaCheck(cudaMalloc(&cudnn_workspace, cudnn_workspace_size));
    }

    // Execute graph
    assert(graph->execute(cudnn_handle, variant_pack, cudnn_workspace).is_good());
    cudaCheck(cudaGetLastError());

    // Optionally convert back from FP16/BF16 to FP32
    if (first_run_validation) {
        int total_threads = B * T * C;
        assert(total_threads % block_size == 0);
        int num_blocks = total_threads / block_size;
        lowp_to_fp32_kernel<<<num_blocks, block_size>>>(out, out_fp32);
    }
    cudaCheck(cudaGetLastError());
    first_run_validation = false;
}

#endif // ENABLE_CUDNN
