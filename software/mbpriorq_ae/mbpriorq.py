import hashlib

import torch
import numpy as np

from .ebw import GlobalEBW
from .logging import elog, DEBUG
from .vmb_profiler import GlobalVMBProfiler

MBPRIORQ_BLOCK_SIZE = 4
SCALING_VECTOR_SIZE = 16  # NVFP4 block size along K
LAST_LEVEL_BLOCK_SIZE = 16

# FP4 tables (E2M1)
FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP4_MAX = 6
FP4_GLOBAL_SCALE_MAX = FP8_MAX * FP4_MAX
e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])

# MBPriorQ tables
MBPRIORQ_MAX = 6
mbpriorq_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
mbpriorq_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])

class MBPriorQ_Quantizer:
    def __init__(self, args, **kwargs):
        self.model_type = args.get("model_type", "cloud")
        self.ablation_mode = args.get("ablation_mode", "paper")
        allowed_ablation_modes = {"paper", "static", "first2_only", "random_same_ratio", "oracle"}
        if self.ablation_mode not in allowed_ablation_modes:
            raise ValueError(
                f"Invalid MBPriorQ ablation_mode: {self.ablation_mode}. "
                f"Expected one of {sorted(allowed_ablation_modes)}."
            )
        self.random_seed = int(args.get("random_seed", 20260606))
        self.refined_block_size = int(args.get("refined_block_size", MBPRIORQ_BLOCK_SIZE))
        if self.refined_block_size not in {2, 4, 8}:
            raise ValueError(
                f"Invalid MBPriorQ refined_block_size: {self.refined_block_size}. "
                "Expected one of [2, 4, 8]."
            )
        if LAST_LEVEL_BLOCK_SIZE % self.refined_block_size != 0:
            raise ValueError(
                f"MBPriorQ refined_block_size={self.refined_block_size} must divide "
                f"{LAST_LEVEL_BLOCK_SIZE}."
            )
        self.refined_blocks_per_microblock = LAST_LEVEL_BLOCK_SIZE // self.refined_block_size
        GlobalEBW.configure_mbpriorq_refined_block_size(self.refined_block_size)
        self.random_call_index = 0
        self.first_vulnerable_mask = None
        self.second_vulnerable_mask = None
        self.stable_mask = None
        self.mask_set = False
        self.using_imatrix = args.get("using_imatrix", False)
        self.imatrix_file_name = args["imatrix_file_name"]
        self.calibration_threshold = None
        self.vmb_profile_enable = bool(args.get("vmb_profile_enable", False))
        self.vmb_profile_call_index = 0
        self.profile_calibration_column_mask = None
        self.metadata_target = args.get("metadata_target", "activation")
        if self.metadata_target not in {"activation", "kv_cache"}:
            raise ValueError(f"Invalid MBPriorQ metadata_target: {self.metadata_target}")

    def _record_activation_vmb_mask(self, vulnerable_mask):
        """Record formal VMB metadata for the active activation-like target."""
        GlobalEBW.record_vmb_mask(self.metadata_target, vulnerable_mask)

    def fake_quantize_weight(self, data: torch.Tensor, name=None):
        org_shape = data.shape
        org_dtype = data.dtype
        global_scale_factor = data.abs().amax().float() / (MBPRIORQ_MAX * FP8_MAX)

        # vulnerable mask search
        data_pattern = data.view(data.shape[0], -1, LAST_LEVEL_BLOCK_SIZE)
        local_std = data_pattern.std(dim=-1).float()
        threshold_local_std, num_replaced_blk, replaced_percent = self._search_threshold_for_replacement(data, name)
        elog(f"Weight Quantizer, name: {name}, threshold_local_std: {threshold_local_std}, num_replaced_blk: {num_replaced_blk}, replaced_percent: {replaced_percent}%", DEBUG)
        vulnerable_mask = local_std > threshold_local_std
        GlobalEBW.record_vmb_mask("weight", vulnerable_mask)
        vulnerable_mask_expanded = vulnerable_mask.repeat_interleave(LAST_LEVEL_BLOCK_SIZE, dim=1).reshape(org_shape)

        # Quantize the value
        # 16 blk
        quants, q_scale = self._quantize_regression_nvfp4(data, 16, global_scale_factor, name)
        blk16_dequant_values = self._dequantize_regression_nvfp4(quants, q_scale, global_scale_factor, org_shape, org_dtype)
        # refined VMB sub-block
        quants, q_scale = self._quantize_regression_nvfp4(data, self.refined_block_size, global_scale_factor, name)
        refined_dequant_values = self._dequantize_regression_nvfp4(quants, q_scale, global_scale_factor, org_shape, org_dtype)

        result = torch.where(vulnerable_mask_expanded, refined_dequant_values, blk16_dequant_values)

        del data, quants, q_scale, blk16_dequant_values, refined_dequant_values, vulnerable_mask_expanded, local_std, threshold_local_std
        return result

    def fake_quantize_activation_edge(self, data: torch.Tensor, name=None, tensor_shape=None):
        org_shape = data.shape
        org_dtype = data.dtype

        if len(org_shape) == 3:
            activation_2d = data.view(-1, org_shape[-1])
        else:
            activation_2d = data

        if not self.mask_set:
            micro_block_data = activation_2d.view(activation_2d.shape[0], -1, LAST_LEVEL_BLOCK_SIZE)
            global_std = micro_block_data.std(dim=-1).float()
            threshold_local_std, num_replaced_blk, replaced_percent = self._search_threshold_for_replacement(data, name)
            elog(f"Calibration: Layer{name}, threshold_local_std: {threshold_local_std}, num_replaced_blk: {num_replaced_blk}, replaced_percent: {replaced_percent}%", DEBUG)
            vulnerable_mask = global_std > threshold_local_std
            self._record_activation_vmb_mask(vulnerable_mask)

            vulnerable_mask_expanded = vulnerable_mask.repeat_interleave(LAST_LEVEL_BLOCK_SIZE, dim=1)

            self.mask_set = True
            self.calibration_threshold = threshold_local_std
            if self.vmb_profile_enable:
                self.profile_calibration_column_mask = vulnerable_mask.any(dim=0).detach().cpu()
            self._record_vmb_profile(
                layer_name=name,
                phase="calibration",
                selected_rule="full_threshold_calibration",
                selected_mask=vulnerable_mask,
                oracle_mask=vulnerable_mask,
                calibration_threshold=threshold_local_std,
                selected_threshold=threshold_local_std,
                oracle_threshold=threshold_local_std,
                replaced_percent=replaced_percent,
            )
        else:
            micro_block_data = activation_2d.view(activation_2d.shape[0], -1, LAST_LEVEL_BLOCK_SIZE)
            global_std = micro_block_data.std(dim=-1).float()
            first2_activation = activation_2d[:2]
            local_std = first2_activation.std(dim=-1).float()
            threshold_local_std, num_replaced_blk, replaced_percent = self._search_threshold_for_replacement(first2_activation, name)
            new_threshold_local_std = (self.calibration_threshold + threshold_local_std) / 2
            vulnerable_mask = global_std > new_threshold_local_std
            self._record_activation_vmb_mask(vulnerable_mask)
            vulnerable_mask_expanded = vulnerable_mask.repeat_interleave(LAST_LEVEL_BLOCK_SIZE, dim=1)
            if self.vmb_profile_enable:
                oracle_mask, oracle_threshold, _, oracle_replaced_percent = self._compute_oracle_vmb_mask(activation_2d, name)
                self._record_vmb_profile(
                    layer_name=name,
                    phase="prior",
                    selected_rule="average_calibration_and_first2_threshold",
                    selected_mask=vulnerable_mask,
                    oracle_mask=oracle_mask,
                    calibration_threshold=self.calibration_threshold,
                    selected_threshold=new_threshold_local_std,
                    oracle_threshold=oracle_threshold,
                    replaced_percent=oracle_replaced_percent,
                )

        N, K = tensor_shape[-2], tensor_shape[-1]

        s2_mbpriorq = activation_2d.abs().amax().float() / (MBPRIORQ_MAX * FP8_MAX)
        mbpriorq_quants_refined, mbpriorq_q_scale_refined = self._quantize_mbpriorq(activation_2d, self.refined_block_size, s2_mbpriorq)
        mbpriorq_dequant_values_refined = self._dequantize_mbpriorq(mbpriorq_quants_refined, self.refined_block_size, mbpriorq_q_scale_refined, s2_mbpriorq, (activation_2d.shape[0], activation_2d.shape[1]), org_dtype)

        s2_input_nvfp4 = activation_2d.abs().amax().float() / (FP4_MAX * FP8_MAX)
        nvfp4_quants, nvfp4_q_scale = self._quantize_nvfp4(activation_2d, SCALING_VECTOR_SIZE, s2_input_nvfp4)
        nvfp4_dequant_values = self._dequantize_nvfp4(nvfp4_quants, nvfp4_q_scale, s2_input_nvfp4, (activation_2d.shape[0], K), org_dtype)

        result = torch.where(vulnerable_mask_expanded, mbpriorq_dequant_values_refined, nvfp4_dequant_values)
        return result.view(org_shape)

    def _random_mask_same_count(self, reference_mask: torch.Tensor, name: str):
        """Create a deterministic random mask with the same true-count as reference_mask."""
        count = int(reference_mask.sum().item())
        total = reference_mask.numel()
        if count <= 0:
            return torch.zeros_like(reference_mask, dtype=torch.bool)
        if count >= total:
            return torch.ones_like(reference_mask, dtype=torch.bool)

        key = f"{self.random_seed}:{name}:{self.random_call_index}:{tuple(reference_mask.shape)}"
        self.random_call_index += 1
        seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little") % (2**31 - 1)

        generator = torch.Generator(device=reference_mask.device)
        generator.manual_seed(seed)
        perm = torch.randperm(total, device=reference_mask.device, generator=generator)
        flat_mask = torch.zeros(total, dtype=torch.bool, device=reference_mask.device)
        flat_mask[perm[:count]] = True
        return flat_mask.view_as(reference_mask)

    def _expand_column_mask(self, column_mask: torch.Tensor, rows: int):
        """Broadcast a column-level VMB mask across activation rows."""
        return column_mask.unsqueeze(0).expand(rows, -1)

    def _match_activation_rows(self, mask: torch.Tensor, rows: int, name: str):
        """Reuse a calibration mask only when its row dimension is compatible."""
        if mask.shape[0] == rows:
            return mask
        if mask.shape[0] == 1:
            return mask.expand(rows, -1)
        if rows == 1:
            return mask[:1]
        raise ValueError(
            f"Cannot reuse MBPriorQ calibration mask for layer {name}: "
            f"calibration rows={mask.shape[0]}, current rows={rows}."
        )

    def _first2_column_mask(self, activation_2d: torch.Tensor, name: str):
        """Compute a column mask from up to two currently available tokens."""
        first2_activation = activation_2d[:2]
        first2_activation = first2_activation.reshape(
            first2_activation.shape[0], -1, LAST_LEVEL_BLOCK_SIZE
        )
        local_std = first2_activation.std(dim=-1).float()
        threshold_local_std, num_replaced_blk, replaced_percent = self._search_threshold_for_replacement(first2_activation, name)
        local_mask = local_std > threshold_local_std
        return local_mask.any(dim=0), threshold_local_std, num_replaced_blk, replaced_percent

    def fake_quantize_activation_cloud(self, data: torch.Tensor, name=None, tensor_shape=None):
        org_shape = data.shape
        org_dtype = data.dtype

        if len(org_shape) == 3:
            activation_2d = data.view(-1, org_shape[-1])
        else:
            activation_2d = data

        if not self.mask_set:
            micro_block_data = activation_2d.view(activation_2d.shape[0], -1, LAST_LEVEL_BLOCK_SIZE)
            global_std = micro_block_data.std(dim=-1).float()
            threshold_local_std, num_replaced_blk, replaced_percent = self._search_threshold_for_replacement(data, name)
            elog(f"Calibration: Layer{name}, threshold_local_std: {threshold_local_std}, num_replaced_blk: {num_replaced_blk}, replaced_percent: {replaced_percent}%", DEBUG)
            reference_mask = global_std > threshold_local_std

            if self.ablation_mode == "random_same_ratio":
                vulnerable_mask = self._random_mask_same_count(reference_mask, name)
                selected_rule = "random_same_ratio_calibration"
            else:
                vulnerable_mask = reference_mask
                selected_rule = "full_mask_calibration" if self.ablation_mode == "paper" else f"{self.ablation_mode}_full_mask_calibration"
            self._record_activation_vmb_mask(vulnerable_mask)

            vulnerable_mask_expanded = vulnerable_mask.repeat_interleave(LAST_LEVEL_BLOCK_SIZE, dim=1)

            self.mask_set = True
            self.first_vulnerable_mask = reference_mask
            if self.vmb_profile_enable:
                self.profile_calibration_column_mask = vulnerable_mask.any(dim=0).detach().cpu()
            self._record_vmb_profile(
                layer_name=name,
                phase="calibration",
                selected_rule=selected_rule,
                selected_mask=vulnerable_mask,
                oracle_mask=reference_mask,
                calibration_threshold=threshold_local_std,
                selected_threshold=threshold_local_std,
                oracle_threshold=threshold_local_std,
                replaced_percent=replaced_percent,
            )
        else:
            local_mask, threshold_local_std, num_replaced_blk, replaced_percent = self._first2_column_mask(activation_2d, name)
            static_mask = self._match_activation_rows(self.first_vulnerable_mask, activation_2d.shape[0], name)
            first2_mask = self._expand_column_mask(local_mask, activation_2d.shape[0])
            paper_mask = static_mask | first2_mask
            oracle_mask = None
            oracle_threshold = None
            oracle_replaced_percent = None

            if self.ablation_mode == "paper":
                vulnerable_mask = paper_mask
                selected_rule = "first_batch_mask_or_first2_column_mask"
            elif self.ablation_mode == "static":
                vulnerable_mask = static_mask
                selected_rule = "static_first_batch_mask"
            elif self.ablation_mode == "first2_only":
                vulnerable_mask = first2_mask
                selected_rule = "first2_column_mask_only"
            elif self.ablation_mode == "random_same_ratio":
                vulnerable_mask = self._random_mask_same_count(paper_mask, name)
                selected_rule = "random_same_ratio_to_paper_mask"
            elif self.ablation_mode == "oracle":
                oracle_mask, oracle_threshold, _, oracle_replaced_percent = self._compute_oracle_vmb_mask(activation_2d, name)
                vulnerable_mask = oracle_mask
                selected_rule = "full_token_oracle_mask"
            else:
                raise ValueError(f"Invalid MBPriorQ ablation mode: {self.ablation_mode}")

            self._record_activation_vmb_mask(vulnerable_mask)

            vulnerable_mask_expanded = vulnerable_mask.repeat_interleave(LAST_LEVEL_BLOCK_SIZE, dim=1)
            if self.vmb_profile_enable:
                if oracle_mask is None:
                    oracle_mask, oracle_threshold, _, oracle_replaced_percent = self._compute_oracle_vmb_mask(activation_2d, name)
                self._record_vmb_profile(
                    layer_name=name,
                    phase="prior",
                    selected_rule=selected_rule,
                    selected_mask=vulnerable_mask,
                    oracle_mask=oracle_mask,
                    selected_threshold=threshold_local_std,
                    oracle_threshold=oracle_threshold,
                    replaced_percent=oracle_replaced_percent,
                )

        N, K = tensor_shape[-2], tensor_shape[-1]

        s2_mbpriorq = activation_2d.abs().amax().float() / (MBPRIORQ_MAX * FP8_MAX)
        mbpriorq_quants_refined, mbpriorq_q_scale_refined = self._quantize_mbpriorq(activation_2d, self.refined_block_size, s2_mbpriorq)
        mbpriorq_dequant_values_refined = self._dequantize_mbpriorq(mbpriorq_quants_refined, self.refined_block_size, mbpriorq_q_scale_refined, s2_mbpriorq, (activation_2d.shape[0], activation_2d.shape[1]), org_dtype)

        s2_input_nvfp4 = activation_2d.abs().amax().float() / (FP4_MAX * FP8_MAX)
        nvfp4_quants, nvfp4_q_scale = self._quantize_nvfp4(activation_2d, SCALING_VECTOR_SIZE, s2_input_nvfp4)
        nvfp4_dequant_values = self._dequantize_nvfp4(nvfp4_quants, nvfp4_q_scale, s2_input_nvfp4, (activation_2d.shape[0], K), org_dtype)

        result = torch.where(vulnerable_mask_expanded, mbpriorq_dequant_values_refined, nvfp4_dequant_values)

        return result.view(org_shape)

    def fake_quantize_activation(self, data: torch.Tensor, name=None, tensor_shape=None):
        if self.model_type == "edge":
            if self.ablation_mode != "paper":
                raise ValueError(
                    "MBPriorQ activation ablation modes are currently implemented only for cloud mode. "
                    f"Got model_type=edge and ablation_mode={self.ablation_mode}."
                )
            return self.fake_quantize_activation_edge(data, name, tensor_shape)
        elif self.model_type == "cloud":
            return self.fake_quantize_activation_cloud(data, name, tensor_shape)
        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

    def _compute_oracle_vmb_mask(self, activation_2d: torch.Tensor, name: str):
        """Compute the full-token VMB mask with the same threshold search used by MBPriorQ."""
        micro_block_data = activation_2d.view(activation_2d.shape[0], -1, LAST_LEVEL_BLOCK_SIZE)
        global_std = micro_block_data.std(dim=-1).float()
        threshold, num_replaced_blk, replaced_percent = self._search_threshold_for_replacement(activation_2d, name)
        return global_std > threshold, threshold, num_replaced_blk, replaced_percent

    def _record_vmb_profile(
        self,
        *,
        layer_name: str,
        phase: str,
        selected_rule: str,
        selected_mask: torch.Tensor,
        oracle_mask: torch.Tensor,
        calibration_threshold=None,
        selected_threshold=None,
        oracle_threshold=None,
        replaced_percent=None,
    ):
        """Record one activation VMB profile row without changing quantization behavior."""
        call_index = self.vmb_profile_call_index
        self.vmb_profile_call_index += 1
        if not self.vmb_profile_enable:
            return

        GlobalVMBProfiler.record(
            layer_name=layer_name,
            model_type=self.model_type,
            call_index=call_index,
            phase=phase,
            selected_rule=selected_rule,
            selected_mask=selected_mask,
            oracle_mask=oracle_mask,
            calibration_column_mask=self.profile_calibration_column_mask,
            calibration_threshold=calibration_threshold,
            selected_threshold=selected_threshold,
            oracle_threshold=oracle_threshold,
            replaced_percent=replaced_percent,
        )

    # ── NVFP4 helpers ──────────────────────────────────────────────────────

    def _nvfp4_get_weights_scaling_factor(
        self,
        input: torch.Tensor,
        block_size: int,
        weights_scaling_factor_2: torch.Tensor | None = None,
        keep_high_precision: bool = False,
    ):
        """Returns quantized per block weight scaling factor."""
        if weights_scaling_factor_2 is None:
            weights_scaling_factor_2 = input.abs().amax().float() / (FP4_MAX * FP8_MAX)

        [n, k] = input.shape[-2:]
        assert block_size != 0, "Block size is zero. Cannot return per_block amax for given input."
        assert k % block_size == 0, (
            "Weight shape is not divisible for block size for block quantization."
        )

        input = input.reshape((*tuple(input.shape[:-2]), n, k // block_size, block_size))
        per_block_amax = input.abs().amax(dim=-1).float()
        per_block_scale = per_block_amax / FP4_MAX
        q_per_block_scale = per_block_scale / weights_scaling_factor_2
        q_per_block_scale[per_block_scale == 0] = 1.0
        return q_per_block_scale, weights_scaling_factor_2

    def _quantize_nvfp4(self, input, block_size, weights_scaling_factor_2):
        weights_scaling_factor, weights_scaling_factor_2 = self._nvfp4_get_weights_scaling_factor(
            input, block_size, weights_scaling_factor_2
        )

        input = input.view((*tuple(input.shape[:-1]), -1, SCALING_VECTOR_SIZE))
        scaled_weight = input / (
            (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
        )
        scaled_weight = scaled_weight.view((*tuple(scaled_weight.shape[:-2]), -1))

        q_weight = self._cast_fp4(scaled_weight)
        packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
        return packed_weight, weights_scaling_factor

    def _dequantize_nvfp4(
        self,
        quantized_t: torch.Tensor,
        scale_1: torch.Tensor,
        scale_2: torch.Tensor,
        orig_shape: tuple,
        orig_dtype: torch.dtype,
    ) -> torch.Tensor:
        device = quantized_t.device
        N, K = orig_shape
        num_blocks = N * (K // SCALING_VECTOR_SIZE)
        s1 = scale_1.reshape(-1)[:num_blocks]

        high = (quantized_t >> 4) & 0x0F
        low = quantized_t & 0x0F
        idx = torch.empty(N, (K // 2) * 2, dtype=torch.long, device=device)
        idx[..., 0::2] = low.long()
        idx[..., 1::2] = high.long()

        vals = e2m1_values.to(device)[idx]

        scale_real = (s1.to(torch.float32) * scale_2.to(torch.float32)).view(N, K // SCALING_VECTOR_SIZE, 1)
        vals = vals.view(N, K // SCALING_VECTOR_SIZE, SCALING_VECTOR_SIZE) * scale_real
        return vals.view(N, K).to(orig_dtype)

    def _cast_fp4(self, weight: torch.Tensor):
        """Converts tensor to uint4."""
        device = weight.device

        mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8).to(device)
        mask_shape = list(weight.shape)
        mask = mask.expand([*mask_shape, 7])

        sign_bit = (weight < 0).to(torch.uint8)
        weight_abs = weight.abs()
        ord = torch.searchsorted(e2m1_bounds.to(device), weight_abs, out_int32=True).to(torch.uint8)
        round = torch.any((weight_abs.unsqueeze(-1) == e2m1_bounds.to(device)) * mask, dim=-1)
        fp4_val = (sign_bit * 0b1000 + ord + round).to(torch.uint8)
        return fp4_val

    # ── MBPriorQ helpers ───────────────────────────────────────────────────

    def _MBPriorQ_get_weights_scaling_factor(
        self,
        input: torch.Tensor,
        block_size: int,
        weights_scaling_factor_2: torch.Tensor | None = None,
        keep_high_precision: bool = False,
    ):
        """Returns quantized per block weight scaling factor."""
        if weights_scaling_factor_2 is None:
            weights_scaling_factor_2 = input.abs().amax().float() / (MBPRIORQ_MAX * FP8_MAX)

        [n, k] = input.shape[-2:]
        assert block_size != 0, "Block size is zero. Cannot return per_block amax for given input."
        assert k % block_size == 0, (
            "Weight shape is not divisible for block size for block quantization."
        )

        input = input.reshape((*tuple(input.shape[:-2]), n, k // block_size, block_size))
        per_block_amax = input.abs().amax(dim=-1).float()
        per_block_scale = per_block_amax / MBPRIORQ_MAX
        q_per_block_scale = per_block_scale / weights_scaling_factor_2
        q_per_block_scale[per_block_scale == 0] = 1.0
        return q_per_block_scale, weights_scaling_factor_2

    def _quantize_mbpriorq(self, input, block_size, weights_scaling_factor_2):
        weights_scaling_factor, weights_scaling_factor_2 = self._MBPriorQ_get_weights_scaling_factor(
            input, block_size, weights_scaling_factor_2
        )

        input = input.view((*tuple(input.shape[:-1]), -1, block_size))
        scaled_weight = input / (
            (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
        )
        scaled_weight = scaled_weight.view((*tuple(scaled_weight.shape[:-2]), -1))

        q_weight = self._cast_mbpriorq(scaled_weight)
        packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
        return packed_weight, weights_scaling_factor

    def _dequantize_mbpriorq(
        self,
        quantized_t: torch.Tensor,
        block_size: int,
        scale_1: torch.Tensor,
        scale_2: torch.Tensor,
        orig_shape: tuple,
        orig_dtype: torch.dtype,
    ) -> torch.Tensor:
        device = quantized_t.device
        N, K = orig_shape
        num_blocks = N * (K // block_size)
        s1 = scale_1.reshape(-1)[:num_blocks]

        high = (quantized_t >> 4) & 0x0F
        low = quantized_t & 0x0F
        idx = torch.empty(N, (K // 2) * 2, dtype=torch.long, device=device)
        idx[..., 0::2] = low.long()
        idx[..., 1::2] = high.long()

        vals = mbpriorq_values.to(device)[idx]

        scale_real = (s1.to(torch.float32) * scale_2.to(torch.float32)).view(N, K // block_size, 1)
        vals = vals.view(N, K // block_size, block_size) * scale_real
        return vals.view(N, K).to(orig_dtype)

    def _cast_mbpriorq(self, weight: torch.Tensor):
        """Converts tensor to mbpriorq format."""
        device = weight.device

        mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8).to(device)
        mask_shape = list(weight.shape)
        mask = mask.expand([*mask_shape, 7])

        sign_bit = (weight < 0).to(torch.uint8)
        weight_abs = weight.abs()
        ord = torch.searchsorted(mbpriorq_bounds.to(device), weight_abs, out_int32=True).to(torch.uint8)
        round = torch.any((weight_abs.unsqueeze(-1) == mbpriorq_bounds.to(device)) * mask, dim=-1)
        mbpriorq_val = (sign_bit * 0b1000 + ord + round).to(torch.uint8)
        return mbpriorq_val

    # ── Regression NVFP4 helpers ───────────────────────────────────────────

    def _quantize_regression_nvfp4(self, weight: torch.Tensor, block_size: int, weights_scaling_factor_2: torch.Tensor, name: str):
        weight = weight.reshape(-1, block_size)
        fp4_values = e2m1_values.to(weight.device)

        max_val = weight.abs().amax(dim=1).float()
        sigma2 = weight.square().mean()

        if self.using_imatrix and name != "lm_head":
            imatrix = self._read_tensor_from_imatrix(
                self.imatrix_file_name, name
            ).to(weight.device)
            org_shape_weight = weight.reshape(-1, len(imatrix))
            temp = torch.ones_like(org_shape_weight)
            modification = imatrix * temp
            modification = modification.reshape(weight.shape)
        else:
            modification = weight.square()

        iscale = (FP4_MAX / max_val).unsqueeze(-1).to(torch.float32)
        scale = 1 / iscale
        quants = weight * iscale
        idx = self._cast_fp4(quants).to(torch.long)
        quants = fp4_values[idx]

        diff = quants * scale - weight
        best_mad = (diff.square() * modification).sum(dim=1).unsqueeze(1)

        fitting_iterations = 15
        rmin = -0.5
        rdelta = 0.1

        for i in range(fitting_iterations):
            new_iscale = (rmin + rdelta * i + FP4_MAX) / max_val
            new_iscale = new_iscale.unsqueeze(-1).to(torch.float32)
            new_scale = 1 / new_iscale
            new_quants = weight * new_iscale
            idx = self._cast_fp4(new_quants).to(torch.long)
            new_quants = fp4_values[idx]

            diff = new_quants * new_scale - weight
            new_mad = (diff.square() * modification).sum(dim=1).unsqueeze(1)

            mask = new_mad < best_mad

            quants = torch.where(mask, new_quants, quants)
            scale = torch.where(mask, new_scale, scale)
            best_mad = torch.where(mask, new_mad, best_mad)
            del diff, new_quants, new_mad, new_scale

        del modification

        scale = scale / weights_scaling_factor_2
        return quants, scale

    def _dequantize_regression_nvfp4(self, data: torch.Tensor, scale1: torch.Tensor, scale2: float, orig_shape: tuple, orig_dtype: torch.dtype):
        real_scale = (scale1 * scale2)
        return (data * real_scale).view(orig_shape).to(orig_dtype)

    def _read_tensor_from_imatrix(self, file_path, tensor_name: str):
        import struct
        import re

        def replace_layer(match):
            layer_id = match.group(1) or match.group(2)
            return f"blk.{layer_id}"

        tensor_name = re.sub(r"model\.layers(?:\[(\d+)\]|\.(\d+))", replace_layer, tensor_name)
        tensor_name = tensor_name.replace("self_attn.o", "attn_output").replace("self_attn.", "attn_").replace("mlp.", "ffn_").replace("_proj", ".weight")

        with open(file_path, 'rb') as f:
            n_entries = struct.unpack('i', f.read(4))[0]

            entries = []
            for _ in range(n_entries):
                len_name = struct.unpack('i', f.read(4))[0]
                name = f.read(len_name).decode('utf-8')
                ncall = struct.unpack('i', f.read(4))[0]
                nval = struct.unpack('i', f.read(4))[0]

                if nval > 0:
                    values = struct.unpack(f'{nval}f', f.read(nval * 4))

                if name == tensor_name:
                    return torch.tensor(values)
        elog(f"Tensor not found: {tensor_name}", DEBUG)
        return None

    def _search_threshold_for_replacement(self, data: torch.Tensor, name: str):
        device = data.device
        blk16_data = data.view(data.shape[0], -1, 16)
        blk16_std = blk16_data.std(dim=-1).float()
        blk16_std_repeated = blk16_std.repeat_interleave(self.refined_blocks_per_microblock, dim=1)

        refined_data = data.view(data.shape[0], -1, self.refined_block_size)
        refined_std = refined_data.std(dim=-1).float()

        optimization = blk16_std_repeated - refined_std
        optimization = optimization.view(-1, self.refined_blocks_per_microblock)
        optimization = optimization.sum(dim=-1).float()
        optimization_sorted, indices_sorted = torch.sort(optimization, descending=True)

        num_data_groups = len(optimization_sorted)
        x = range(num_data_groups)
        x_tensor = torch.arange(num_data_groups, device=device) + 1
        f_x = torch.cumsum(optimization_sorted, dim=0) / x_tensor
        max_fval = f_x[0]
        ablation_fx = max_fval - f_x
        ablation_fx_sorted, indices_sorted = torch.sort(ablation_fx, descending=True)

        areas = np.array(x) * ablation_fx_sorted.cpu().numpy()
        max_area_idx = np.argmax(areas)
        max_area_x = x[max_area_idx]

        std_sorted, indices_sorted = torch.sort(blk16_std.flatten())
        threshold = std_sorted[max_area_x]

        num_replaced_blk = (num_data_groups - max_area_x)
        replaced_percent = (1 - max_area_x / num_data_groups) * 100
        return threshold, num_replaced_blk, replaced_percent
