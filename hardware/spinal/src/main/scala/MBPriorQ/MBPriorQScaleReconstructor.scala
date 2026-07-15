package MBPriorQ

import General_IP.{FP32Multiplier, FP8ToFP32}
import spinal.core._

/**
 * Per-block dequant-scale reconstructor for MBPriorQ.
 *
 * Refined mode follows the minimum-difference metadata contract: the normal
 * BlockScale remains the first refined sub-block scale, while ExtendedBlockScale
 * stores only the three extra refined sub-block scales. If only one side is VMB,
 * the non-VMB side reuses its normal BlockScale for all four refined sub-blocks.
 *
 * This module is the numeric scale-selection datapath for one block. It is not
 * the shared FPU-pool scheduler; the pool decides when these FP32 operations are
 * issued and how many blocks are active.
 */
class MBPriorQScaleReconstructor extends Component {
  private val subBlockCount = 4
  private val extraScaleCount = 3

  val io = new Bundle {
    val global_scale_w = in Bits(32 bits)
    val global_scale_i = in Bits(32 bits)

    val weight_scale_base = in Bits(8 bits)
    val input_scale_base = in Bits(8 bits)
    val weight_scale_ext = in Bits(8 * extraScaleCount bits)
    val input_scale_ext = in Bits(8 * extraScaleCount bits)

    val weight_vmb = in Bool()
    val input_vmb = in Bool()

    val factor_valid_mask = out Bits(subBlockCount bits)
    val dequant_factors = out Bits(32 * subBlockCount bits)
  }

  private val combinedVmb = io.weight_vmb || io.input_vmb
  io.factor_valid_mask := Mux(combinedVmb, B"4'b1111", B"4'b0001")

  private val weightFp8 = Vec(Bits(8 bits), subBlockCount)
  private val inputFp8 = Vec(Bits(8 bits), subBlockCount)
  for(i <- 0 until subBlockCount) {
    if(i == 0) {
      weightFp8(i) := io.weight_scale_base
      inputFp8(i) := io.input_scale_base
    } else {
      val extStart = 8 * (i - 1)
      val extEnd = extStart + 7
      weightFp8(i) := Mux(io.weight_vmb, io.weight_scale_ext(extEnd downto extStart), io.weight_scale_base)
      inputFp8(i) := Mux(io.input_vmb, io.input_scale_ext(extEnd downto extStart), io.input_scale_base)
    }
  }

  private val weightFp32 = Vec(Bits(32 bits), subBlockCount)
  private val inputFp32 = Vec(Bits(32 bits), subBlockCount)
  private val weightScaled = Vec(Bits(32 bits), subBlockCount)
  private val inputScaled = Vec(Bits(32 bits), subBlockCount)

  for(i <- 0 until subBlockCount) {
    val weightConverter = new FP8ToFP32
    val inputConverter = new FP8ToFP32
    weightConverter.io.fp8_in := weightFp8(i)
    inputConverter.io.fp8_in := inputFp8(i)
    weightFp32(i) := weightConverter.io.fp32_out
    inputFp32(i) := inputConverter.io.fp32_out

    val weightMul = new FP32Multiplier
    val inputMul = new FP32Multiplier
    weightMul.io.a := weightFp32(i)
    weightMul.io.b := io.global_scale_w
    inputMul.io.a := inputFp32(i)
    inputMul.io.b := io.global_scale_i
    weightScaled(i) := weightMul.io.result
    inputScaled(i) := inputMul.io.result

    val finalMul = new FP32Multiplier
    finalMul.io.a := weightScaled(i)
    finalMul.io.b := inputScaled(i)
    io.dequant_factors(32 * (i + 1) - 1 downto 32 * i) := finalMul.io.result
  }
}

object MBPriorQScaleReconstructorGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl_scale_reconstructor",
      bitVectorWidthMax = 4096
    ).generateVerilog(new MBPriorQScaleReconstructor)
  }
}
