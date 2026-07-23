class GlobalEBW:
    DATA_EBW = 4.0
    NVFP4_SCALE_EBW = 8.0 / 16.0
    MBPRIORQ_REFINED_BLOCK_SIZE = 4
    MBPRIORQ_VMB_SCALE_EBW = (
        8.0 * (16.0 / MBPRIORQ_REFINED_BLOCK_SIZE)
    ) / 16.0
    MASK_EBW = 1.0 / 16.0
    NVFP4_EBW = DATA_EBW + NVFP4_SCALE_EBW

    weight_vmb_blocks = 0
    weight_total_blocks = 0
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
        cls.weight_vmb_blocks = 0
        cls.weight_total_blocks = 0
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
        cls.MBPRIORQ_VMB_SCALE_EBW = (
            8.0 * (16.0 / refined_block_size)
        ) / 16.0

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
                overhead_percent_vs_nvfp4 = (
                    extra_over_nvfp4_ebw / cls.NVFP4_EBW * 100.0
                )
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
        overhead_percent_vs_nvfp4 = (
            extra_over_nvfp4_ebw / cls.NVFP4_EBW * 100.0
        )

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
