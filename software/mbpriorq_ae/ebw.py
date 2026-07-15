class GlobalEBW:
    DATA_EBW = 4.0
    NVFP4_SCALE_EBW = 8.0 / 16.0
    MBPRIORQ_REFINED_BLOCK_SIZE = 4
    MBPRIORQ_VMB_SCALE_EBW = (8.0 * (16.0 / MBPRIORQ_REFINED_BLOCK_SIZE)) / 16.0
    MASK_EBW = 1.0 / 16.0
    NVFP4_EBW = DATA_EBW + NVFP4_SCALE_EBW

    # Legacy reference counters. This preserves the previous layer-proportion
    # sampler: accumulated replaced_percent * layer_proportion, divided by
    # accumulated layer_proportion.
    weight_percent = 0
    weight_cnt = 0
    weight_vmb_blocks = 0
    weight_total_blocks = 0

    # Legacy reference counters, same policy as weight_percent/weight_cnt.
    activation_percent = 0
    activation_cnt = 0
    activation_vmb_blocks = 0
    activation_total_blocks = 0

    kv_cache_vmb_blocks = 0
    kv_cache_total_blocks = 0
    kv_cache_generic_data_values = 0
    kv_cache_generic_metadata_bits = 0.0
    kv_cache_generic_payload_bits = None
    kv_cache_generic_metadata_name = None
    kv_cache_generic_method = None

    @classmethod
    def reset(cls):
        """Reset all run-local metadata counters."""
        cls.weight_percent = 0
        cls.weight_cnt = 0
        cls.weight_vmb_blocks = 0
        cls.weight_total_blocks = 0
        cls.activation_percent = 0
        cls.activation_cnt = 0
        cls.activation_vmb_blocks = 0
        cls.activation_total_blocks = 0
        cls.kv_cache_vmb_blocks = 0
        cls.kv_cache_total_blocks = 0
        cls.kv_cache_generic_data_values = 0
        cls.kv_cache_generic_metadata_bits = 0.0
        cls.kv_cache_generic_payload_bits = None
        cls.kv_cache_generic_metadata_name = None
        cls.kv_cache_generic_method = None
        cls.configure_mbpriorq_refined_block_size(4)

    @classmethod
    def configure_mbpriorq_refined_block_size(cls, refined_block_size):
        refined_block_size = int(refined_block_size)
        if refined_block_size not in {2, 4, 8}:
            raise ValueError(
                f"Unsupported MBPriorQ refined block size: {refined_block_size}. "
                "Expected one of [2, 4, 8]."
            )
        cls.MBPRIORQ_REFINED_BLOCK_SIZE = refined_block_size
        cls.MBPRIORQ_VMB_SCALE_EBW = (8.0 * (16.0 / refined_block_size)) / 16.0

    @classmethod
    def record_legacy_layer_percent(cls, target, replaced_percent, layer_proportion):
        if layer_proportion is None:
            return

        if target == "weight":
            cls.weight_percent += float(replaced_percent) * float(layer_proportion)
            cls.weight_cnt += float(layer_proportion)
        elif target == "activation":
            cls.activation_percent += float(replaced_percent) * float(layer_proportion)
            cls.activation_cnt += float(layer_proportion)
        else:
            raise ValueError(f"Unsupported EBW target: {target}")

    @classmethod
    def record_vmb_mask(cls, target, mask):
        total_blocks = int(mask.numel())
        if total_blocks == 0:
            return
        vmb_blocks = int(mask.sum().item())
        cls.record_vmb_blocks(target, vmb_blocks, total_blocks)

    @classmethod
    def record_vmb_blocks(cls, target, vmb_blocks, total_blocks):
        if total_blocks <= 0:
            return

        if target == "weight":
            cls.weight_vmb_blocks += int(vmb_blocks)
            cls.weight_total_blocks += int(total_blocks)
        elif target == "activation":
            cls.activation_vmb_blocks += int(vmb_blocks)
            cls.activation_total_blocks += int(total_blocks)
        elif target == "kv_cache":
            cls.kv_cache_vmb_blocks += int(vmb_blocks)
            cls.kv_cache_total_blocks += int(total_blocks)
        else:
            raise ValueError(f"Unsupported EBW target: {target}")

    @classmethod
    def record_kv_cache_metadata(
        cls,
        *,
        method,
        data_values,
        metadata_values,
        metadata_bits_per_value,
        payload_bits_per_value=4,
        metadata_name="metadata",
    ):
        """Record generic KV-cache metadata EBW for an MBPriorQ experiment."""
        if data_values <= 0 or metadata_values < 0:
            return
        metadata_bits = float(metadata_values) * float(metadata_bits_per_value)
        cls.kv_cache_generic_data_values += int(data_values)
        cls.kv_cache_generic_metadata_bits += metadata_bits
        cls.kv_cache_generic_payload_bits = float(payload_bits_per_value)
        cls.kv_cache_generic_metadata_name = metadata_name
        cls.kv_cache_generic_method = method

    @classmethod
    def summarize(cls, target):
        if target == "weight":
            vmb_blocks = cls.weight_vmb_blocks
            total_blocks = cls.weight_total_blocks
        elif target == "activation":
            vmb_blocks = cls.activation_vmb_blocks
            total_blocks = cls.activation_total_blocks
        elif target == "kv_cache":
            vmb_blocks = cls.kv_cache_vmb_blocks
            total_blocks = cls.kv_cache_total_blocks
        else:
            raise ValueError(f"Unsupported EBW target: {target}")

        if total_blocks == 0:
            if target == "kv_cache" and cls.kv_cache_generic_data_values > 0:
                payload_ebw = (
                    cls.kv_cache_generic_payload_bits
                    if cls.kv_cache_generic_payload_bits is not None
                    else cls.DATA_EBW
                )
                metadata_ebw = (
                    cls.kv_cache_generic_metadata_bits
                    / float(cls.kv_cache_generic_data_values)
                )
                effective_ebw = payload_ebw + metadata_ebw
                extra_over_nvfp4_ebw = effective_ebw - cls.NVFP4_EBW
                overhead_percent_vs_nvfp4 = extra_over_nvfp4_ebw / cls.NVFP4_EBW * 100.0
                return {
                    "metadata_kind": "generic_kv_cache",
                    "method": cls.kv_cache_generic_method,
                    "metadata_name": cls.kv_cache_generic_metadata_name,
                    "data_values": cls.kv_cache_generic_data_values,
                    "metadata_bits": cls.kv_cache_generic_metadata_bits,
                    "payload_ebw": payload_ebw,
                    "metadata_ebw": metadata_ebw,
                    "effective_ebw": effective_ebw,
                    "extra_over_nvfp4_ebw": extra_over_nvfp4_ebw,
                    "overhead_percent_vs_nvfp4": overhead_percent_vs_nvfp4,
                }
            return None

        vmb_partition = float(vmb_blocks) / float(total_blocks)
        scale_ebw = (
            cls.NVFP4_SCALE_EBW * (1.0 - vmb_partition)
            + cls.MBPRIORQ_VMB_SCALE_EBW * vmb_partition
        )
        scale_extra_ebw = scale_ebw - cls.NVFP4_SCALE_EBW
        metadata_ebw = cls.MASK_EBW + scale_ebw
        effective_ebw = cls.DATA_EBW + metadata_ebw
        extra_over_nvfp4_ebw = effective_ebw - cls.NVFP4_EBW
        overhead_percent_vs_nvfp4 = extra_over_nvfp4_ebw / cls.NVFP4_EBW * 100.0

        return {
            "vmb_blocks": vmb_blocks,
            "total_blocks": total_blocks,
            "refined_block_size": cls.MBPRIORQ_REFINED_BLOCK_SIZE,
            "vmb_partition": vmb_partition,
            "mask_ebw": cls.MASK_EBW,
            "scale_ebw": scale_ebw,
            "scale_extra_ebw": scale_extra_ebw,
            "metadata_ebw": metadata_ebw,
            "effective_ebw": effective_ebw,
            "extra_over_nvfp4_ebw": extra_over_nvfp4_ebw,
            "overhead_percent_vs_nvfp4": overhead_percent_vs_nvfp4,
        }

    @classmethod
    def summarize_legacy(cls, target):
        if target == "weight":
            percent_acc = cls.weight_percent
            cnt = cls.weight_cnt
        elif target == "activation":
            percent_acc = cls.activation_percent
            cnt = cls.activation_cnt
        else:
            raise ValueError(f"Unsupported EBW target: {target}")

        if cnt <= 0:
            return None

        vmb_partition = float(percent_acc) / float(cnt) / 100.0
        scale_ebw = cls.NVFP4_SCALE_EBW + 1.5 * vmb_partition
        effective_ebw = cls.DATA_EBW + scale_ebw
        return {
            "vmb_partition": vmb_partition,
            "scale_ebw": scale_ebw,
            "effective_ebw": effective_ebw,
            "percent_acc": float(percent_acc),
            "cnt": float(cnt),
        }

    Qwen2_5_0_5B_weight = {
        "q_proj": 7,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 7,
        "gate_proj": 38,
        "up_proj": 38,
        "down_proj": 38,
        "lm_head": 1187
    }

    Qwen2_5_0_5B_input = {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 7,
        "gate_proj": 1,
        "up_proj": 1,
        "down_proj": 5.43,
        "lm_head": 1
    }

    Qwen2_5_1_5B_weight = {
        "q_proj": 6,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 6,
        "gate_proj": 35,
        "up_proj": 35,
        "down_proj": 35,
        "lm_head": 593.5
    }

    Qwen2_5_1_5B_input = {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 1,
        "gate_proj": 1,
        "up_proj": 1,
        "down_proj": 5.83,
        "lm_head": 1
    }



    Qwen2_5_3B_weight = {
        "q_proj": 8,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 8,
        "gate_proj": 43,
        "up_proj": 43,
        "down_proj": 43,
        "lm_head": 593.5
    }

    Qwen2_5_3B_input = {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 1,
        "gate_proj": 1,
        "up_proj": 1,
        "down_proj": 52.9,
        "lm_head": 1
    }

    Qwen3_0_6B_weight = {
        "q_proj": 2,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 2,
        "gate_proj": 3,
        "up_proj": 3,
        "down_proj": 3,
        "lm_head": 148.375
    }
    Qwen3_0_6B_input = {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 2,
        "gate_proj": 1,
        "up_proj": 1,
        "down_proj": 3,
        "lm_head": 1
    }

    Qwen3_8B_input = {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 2,
        "gate_proj": 1,
        "up_proj": 1,
        "down_proj": 3,
        "lm_head": 1
    }

    Qwen3_8B_weight = {
        "q_proj": 4,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 4,
        "gate_proj": 12,
        "up_proj": 12,
        "down_proj": 12,
        "lm_head": 148.375
    }



    Qwen3_30B_A3B_weight = {
        "q_proj": 8,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 8,
        "gate":0.25,
        "gate_proj": 1.5,
        "up_proj": 1.5,
        "down_proj": 1.5,
        "lm_head": 296.75
    }

    Qwen3_30B_A3B_weight = {
        "q_proj": 8,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 8,
        "gate":0.25,
        "gate_proj": 1.5,
        "up_proj": 1.5,
        "down_proj": 1.5,
        "lm_head": 296.75
    }

    Qwen3_30B_A3B_input = {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 2,
        "gate":1,
        "gate_proj": 1,
        "up_proj": 1,
        "down_proj": 0.375,
        "lm_head": 1
    }

    Mixtral_8x7B_HM_weight = {
        "q_proj": 512,
        "k_proj": 128,
        "v_proj": 128,
        "o_proj": 512,
        "gate": 1,
        "w1": 1792,
        "w2": 1792,
        "w3": 1792,
        "lm_head": 4000
    }

    Mixtral_8x7B_HM_activation = {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 1,
        "gate": 1,
        "w1": 1,
        "w2": 3.5,
        "w3": 1,
        "lm_head": 1
    }

    Llama_3_2_1B_weight = {
        "q_proj": 4,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 4,
        "gate_proj": 8,
        "up_proj": 8,
        "down_proj": 8,
        "lm_head": 250.5
    }

    Llama_3_2_1B_input = {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 1,
        "gate_proj": 1,
        "up_proj": 1,
        "down_proj": 4,
        "lm_head": 1
    }

    Llama_33_70B_Instruct_weight = {
        "q_proj": 8,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 8,
        "gate_proj": 28,
        "up_proj": 28,
        "down_proj": 28,
        "lm_head": 125.25
    }

    Llama_33_70B_Instruct_input = {
        "q_proj": 1,
        "k_proj": 1,
        "v_proj": 1,
        "o_proj": 1,
        "gate_proj": 1,
        "up_proj": 1,
        "down_proj": 3.5,
        "lm_head": 1
    }

    DeepSeek_V2_Lite_weight = {
        "q_proj": 48,
        "kv_a_proj_with_mqa": 9,
        "kv_b_proj": 16,
        "o_proj": 32,
        "gate_proj": 44,
        "up_proj": 44,
        "down_proj": 44,
        "gate":1,
        "lm_head":1600
    }

    # only first layer is dense, sum as moe
    DeepSeek_V2_Lite_input = {
        "q_proj": 4,
        "kv_a_proj_with_mqa": 4,
        "kv_b_proj": 4,
        "o_proj": 4,
        "gate_proj": 4,
        "up_proj":4,
        "down_proj": 5.5,
        "gate":1,
        "lm_head":1
    }

class GlobalBatchNumber:
    batch_number = {
        "q_proj": 0,
        "k_proj": 0,
        "v_proj": 0,
        "o_proj": 0,
        "gate_proj": 0,
        "up_proj": 0,
        "down_proj": 0,
        "lm_head": 0
    }


class GlobalSQNR:
    """
    Accumulates per-layer SQNR (Signal-to-Quantization-Noise Ratio) records
    for weights, activations, and KV cache.

    SQNR = 10 * log10( ||W||^2 / ||W - Q(W)||^2 )  [dB]

    Usage:
        GlobalSQNR.enabled = True          # must be set before quantization
        GlobalSQNR.record_weight(name, sqnr_db, quantizer="nvfp4", bit_width=4)
        GlobalSQNR.record_act(name, sqnr_db, quantizer="nvfp4", bit_width=4)
        GlobalSQNR.record_kv(name, sqnr_db, quantizer="mbpriorq", bit_width=4)
        df = GlobalSQNR.to_dataframe("weight")  # returns pandas DataFrame
        GlobalSQNR.reset()
    """
    enabled        = False  # master switch — set via --sqnr_enable in Arguments.json
    weight_records = []     # list of dicts: {layer_name, sqnr_db, quantizer, bit_width}
    act_records    = []
    kv_records     = []

    @classmethod
    def record_weight(cls, layer_name, sqnr_db, quantizer=None, bit_width=None):
        cls.weight_records.append({
            "layer_name": layer_name,
            "sqnr_db":    round(float(sqnr_db), 4),
            "quantizer":  quantizer,
            "bit_width":  bit_width,
        })

    @classmethod
    def record_act(cls, layer_name, sqnr_db, quantizer=None, bit_width=None):
        cls.act_records.append({
            "layer_name": layer_name,
            "sqnr_db":    round(float(sqnr_db), 4),
            "quantizer":  quantizer,
            "bit_width":  bit_width,
        })

    @classmethod
    def record_kv(cls, layer_name, sqnr_db, quantizer=None, bit_width=None):
        cls.kv_records.append({
            "layer_name": layer_name,
            "sqnr_db":    round(float(sqnr_db), 4),
            "quantizer":  quantizer,
            "bit_width":  bit_width,
        })

    @classmethod
    def reset(cls):
        cls.weight_records.clear()
        cls.act_records.clear()
        cls.kv_records.clear()

    @classmethod
    def to_dataframe(cls, target="weight"):
        import pandas as pd
        if target == "weight":
            records = cls.weight_records
        elif target == "act":
            records = cls.act_records
        elif target == "kv":
            records = cls.kv_records
        else:
            raise ValueError(f"Unknown SQNR target: {target}")
        return pd.DataFrame(records)
