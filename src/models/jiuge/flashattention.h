#include "mccl.h"
#include "infinicore_infer.h"
#include <vector>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

void calc_paged_attn_meta_graph_compatible(
    void *cu_seq_q_lens, void *seq_kv_lens,
    void *block_table, int64_t num_decode_batch,
    int64_t num_q_heads, int64_t num_kv_heads, int64_t block_size,
    int64_t head_group, int64_t page_tokens, int64_t decode_block_m,
    int64_t decode_block_n, int64_t prefill_block_m, int64_t prefill_block_n,
    int64_t num_mp, std::string const mode, bool is_prefill, void *out, int32_t num_batch, musaStream_t stream);

void flash_attn_with_kvcache(void *q, void *k_cache,
                             void *v_cache,
                             void *block_table,
                             void *attn_metadata, 
                             void *workspace,
                             double rln2_scale,
                             bool with_causal, std::string const mode,
                             bool is_prefill, void *out, 
                             std::vector<int32_t> size, std::vector<int32_t> stride, musaStream_t stream);

#ifdef __cplusplus
}
#endif

void get_paged_gqa_decode_meta_graph_compatible(
    void *cu_seq_q_lens,
    void *seq_kv_lens,
    void *block_table,
    void *out,
    infinirtStream_t stream,
    int num_batch,
    int num_q_heads = 8,
    int num_kv_heads = 1,
    int block_size = 64,
    int head_group = 8,
    int page_tokens = 64,
    int decode_block_m = 64,
    int decode_block_n = 256,
    int prefill_block_m = 256,
    int prefill_block_n = 128,
    int num_mp = 60
){
    calc_paged_attn_meta_graph_compatible(
        cu_seq_q_lens,
        seq_kv_lens,
        block_table,
        num_batch,
        num_q_heads,
        num_kv_heads,
        block_size,
        head_group,
        page_tokens,
        decode_block_m,
        decode_block_n,
        prefill_block_m,
        prefill_block_n,
        num_mp,
        "GQA",
        false,
        out,
        num_batch,
        static_cast<musaStream_t>(stream)
    );
}


void get_paged_gqa_prefill_meta_graph_compatible(
    void *cu_seq_q_lens,
    void *seq_kv_lens,
    void *block_table,
    void *out,
    infinirtStream_t stream,
    int num_batch,
    int num_q_heads = 8,
    int num_kv_heads = 1,
    int block_size = 64,
    int head_group = 8,
    int page_tokens = 64,
    int decode_block_m = 64,
    int decode_block_n = 256,
    int prefill_block_m = 256,
    int prefill_block_n = 128,
    int num_mp = 60
){
    calc_paged_attn_meta_graph_compatible(
        cu_seq_q_lens,
        seq_kv_lens,
        block_table,
        num_batch,
        num_q_heads,
        num_kv_heads,
        block_size,
        head_group,
        page_tokens,
        decode_block_m,
        decode_block_n,
        prefill_block_m,
        prefill_block_n,
        num_mp,
        "GQA",
        true,
        out,
        num_batch,
        static_cast<musaStream_t>(stream)
    );
}

void gqa_attn_decode_with_kvcache(
    void *q,
    void *k_cache,
    void *v_cache,
    void *block_table,
    void *attn_metadata,
    void *workspace,
    bool with_causal,
    float softmax_scale,
    void *out,
    int v_head_size,
    std::vector<int32_t> size, 
    std::vector<int32_t> stride,
    infinirtStream_t stream
){
    flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        block_table,
        attn_metadata,
        workspace,
        softmax_scale,
        with_causal,
        "GQA",
        false,
        out,
        size,
        stride,
        static_cast<musaStream_t>(stream)
    );
}

void gqa_attn_prefill_with_kvcache(
    void *q,
    void *k_cache,
    void *v_cache,
    void *block_table,
    void *attn_metadata,
    bool with_causal,
    float softmax_scale,
    void * out,
    int v_head_size,
    std::vector<int32_t> size,
    std::vector<int32_t> stride,
    infinirtStream_t stream
){
    flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        block_table,
        attn_metadata,
        attn_metadata,
        softmax_scale,
        with_causal,
        "GQA",
        true,
        out,
        size,
        stride,
        static_cast<musaStream_t>(stream)
    );
}
