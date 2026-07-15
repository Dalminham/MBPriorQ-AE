package MBPriorQ

import spinal.core._

/**
 * Stores reconstructed dequant factors for blocks accepted by the VPU issue
 * path and releases them when the shared FPU pool reports the matching
 * block_idx as complete.
 *
 * This module connects the numeric refined-scale selection semantics to the
 * block-index synchronization contract. It does not replace the shared FPU
 * scheduler; it assumes issue/done timing is controlled by
 * MBPriorQSharedFpuPool.
 */
class MBPriorQScaleFactorBuffer(blockCount: Int = 16) extends Component {
  require(blockCount > 0, "Block count must be positive.")
  require(blockCount <= 256, "Packet block index is 8 bits.")

  private val blockIdxWidth = log2Up(blockCount)
  private val subBlockCount = 4
  private val factorBits = 32 * subBlockCount
  private val extraScaleCount = 3

  val io = new Bundle {
    val clear = in Bool()

    val issue_valid = in Bool()
    val issue_block_idx = in UInt(blockIdxWidth bits)
    val issue_mask_valid = in Bool()
    val issue_mask = in Bits(blockCount bits)

    val weight_vmb_mask = in Bits(blockCount bits)
    val input_vmb_mask = in Bits(blockCount bits)

    val global_scale_w = in Bits(32 bits)
    val global_scale_i = in Bits(32 bits)
    val weight_scale_base = in Bits(8 * blockCount bits)
    val input_scale_base = in Bits(8 * blockCount bits)
    val weight_scale_ext = in Bits(8 * extraScaleCount * blockCount bits)
    val input_scale_ext = in Bits(8 * extraScaleCount * blockCount bits)

    val done_valid = in Bool()
    val done_block_idx = in UInt(blockIdxWidth bits)

    val factor_valid = out Bool()
    val factor_block_idx = out UInt(blockIdxWidth bits)
    val factor_valid_mask = out Bits(subBlockCount bits)
    val dequant_factors = out Bits(factorBits bits)
  }

  private val storedValid = Reg(Bits(blockCount bits)) init(0)
  private val storedFactorMask = Vec(Reg(Bits(subBlockCount bits)) init(0), blockCount)
  private val storedFactors = Vec(Reg(Bits(factorBits bits)) init(0), blockCount)
  private val factorValidReg = Reg(Bool()) init(False)
  private val factorBlockReg = Reg(UInt(blockIdxWidth bits)) init(0)
  private val factorMaskReg = Reg(Bits(subBlockCount bits)) init(0)
  private val factorDataReg = Reg(Bits(factorBits bits)) init(0)

  private val reconstructors = Array.fill(blockCount)(new MBPriorQScaleReconstructor)

  private def scale8(flat: Bits, idx: Int): Bits =
    flat(8 * (idx + 1) - 1 downto 8 * idx)

  private def scaleExt(flat: Bits, idx: Int): Bits =
    flat(8 * extraScaleCount * (idx + 1) - 1 downto 8 * extraScaleCount * idx)

  for(i <- 0 until blockCount) {
    reconstructors(i).io.global_scale_w := io.global_scale_w
    reconstructors(i).io.global_scale_i := io.global_scale_i
    reconstructors(i).io.weight_scale_base := scale8(io.weight_scale_base, i)
    reconstructors(i).io.input_scale_base := scale8(io.input_scale_base, i)
    reconstructors(i).io.weight_scale_ext := scaleExt(io.weight_scale_ext, i)
    reconstructors(i).io.input_scale_ext := scaleExt(io.input_scale_ext, i)
    reconstructors(i).io.weight_vmb := io.weight_vmb_mask(i)
    reconstructors(i).io.input_vmb := io.input_vmb_mask(i)
  }

  io.factor_valid := factorValidReg
  io.factor_block_idx := factorBlockReg
  io.factor_valid_mask := factorMaskReg
  io.dequant_factors := factorDataReg

  when(io.clear) {
    storedValid := 0
    factorValidReg := False
    factorBlockReg := 0
    factorMaskReg := 0
    factorDataReg := 0
    for(i <- 0 until blockCount) {
      storedFactorMask(i) := 0
      storedFactors(i) := 0
    }
  } otherwise {
    factorValidReg := False

    for(i <- 0 until blockCount) {
      val scalarIssue = io.issue_valid && io.issue_block_idx === U(i, blockIdxWidth bits)
      val maskIssue = io.issue_mask_valid && io.issue_mask(i)
      when(scalarIssue || maskIssue) {
        storedValid(i) := True
        storedFactorMask(i) := reconstructors(i).io.factor_valid_mask
        storedFactors(i) := reconstructors(i).io.dequant_factors
      }
    }

    when(io.done_valid && storedValid(io.done_block_idx)) {
      factorValidReg := True
      factorBlockReg := io.done_block_idx
      factorMaskReg := storedFactorMask(io.done_block_idx)
      factorDataReg := storedFactors(io.done_block_idx)
      storedValid(io.done_block_idx) := False
    }
  }
}

object MBPriorQScaleFactorBufferGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl_scale_factor_buffer",
      bitVectorWidthMax = 131072
    ).generateVerilog(new MBPriorQScaleFactorBuffer(16))
  }
}
