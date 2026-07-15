package MBPriorQ

import spinal.core._

/**
 * Joins MSA partial matrices and VPU dequant factors into matrix-scale pairs.
 *
 * This is the output-side contract needed by MBPriorQOutputPacketizer:
 *   block_index + sub_block_idx + 16x16 BF16 partial matrix + FP32 scale
 *
 * Normal blocks use factor_valid_mask = 0001 and require one MSA partial
 * matrix at sub_block_idx 0. Refined blocks use factor_valid_mask = 1111 and
 * require four MSA partial matrices at sub_block_idx 0..3.
 */
class MBPriorQOutputPairJoinBuffer(
  blockCount: Int = 16,
  matrixBits: Int = 16 * 16 * 16
) extends Component {
  require(blockCount > 0, "blockCount must be positive.")
  require(blockCount <= 256, "block_index is 8 bits.")

  private val blockIdxWidth = log2Up(blockCount)
  private val readyCntWidth = log2Up(blockCount + 1)

  val io = new Bundle {
    val clear = in Bool()

    val msa_valid = in Bool()
    val msa_block_idx = in UInt(blockIdxWidth bits)
    val msa_sub_block_idx = in UInt(2 bits)
    val msa_matrix = in Bits(matrixBits bits)

    val factor_valid = in Bool()
    val factor_block_idx = in UInt(blockIdxWidth bits)
    val factor_valid_mask = in Bits(4 bits)
    val factor_data = in Bits(128 bits)

    val output_ready = in Bool()
    val output_valid = out Bool()
    val output_block_idx = out UInt(8 bits)
    val output_sub_block_idx = out UInt(2 bits)
    val output_matrix = out Bits(matrixBits bits)
    val output_dequant_scale = out Bits(32 bits)

    val done = out Bool()
    val ready_count = out UInt(readyCntWidth bits)
    val max_ready_count = out UInt(readyCntWidth bits)
    val commit_ptr_dbg = out UInt(blockIdxWidth bits)
  }

  private val msaReady = Vec(Reg(Bits(4 bits)) init(0), blockCount)
  private val factorReady = Reg(Bits(blockCount bits)) init(0)
  private val committed = Reg(Bits(blockCount bits)) init(0)
  private val matrices = Vec(Vec(Reg(Bits(matrixBits bits)) init(0), 4), blockCount)
  private val factorMask = Vec(Reg(Bits(4 bits)) init(0), blockCount)
  private val factors = Vec(Reg(Bits(128 bits)) init(0), blockCount)
  private val commitPtr = Reg(UInt(blockIdxWidth bits)) init(0)
  private val emitSubIdx = Reg(UInt(2 bits)) init(0)
  private val maxReady = Reg(UInt(readyCntWidth bits)) init(0)

  private val currentMask = factorMask(commitPtr)
  private val currentMsaReady = msaReady(commitPtr)
  private val currentReady = factorReady(commitPtr) && ((currentMsaReady & currentMask) === currentMask) && !committed(commitPtr)
  private val currentIsRefined = currentMask === B"4'b1111"
  private val currentLastSub = Mux(currentIsRefined, emitSubIdx === U(3, 2 bits), emitSubIdx === U(0, 2 bits))

  private var readyCountComb = U(0, readyCntWidth bits)
  for(i <- 0 until blockCount) {
    val ready = factorReady(i) && ((msaReady(i) & factorMask(i)) === factorMask(i)) && !committed(i)
    readyCountComb = readyCountComb + ready.asUInt.resize(readyCntWidth)
  }

  private val selectedFactor0 = factors(commitPtr)(31 downto 0)
  private val selectedFactor1 = factors(commitPtr)(63 downto 32)
  private val selectedFactor2 = factors(commitPtr)(95 downto 64)
  private val selectedFactor3 = factors(commitPtr)(127 downto 96)
  private val selectedFactor = Bits(32 bits)
  selectedFactor := selectedFactor0
  when(emitSubIdx === U(1, 2 bits)) {
    selectedFactor := selectedFactor1
  } elsewhen(emitSubIdx === U(2, 2 bits)) {
    selectedFactor := selectedFactor2
  } elsewhen(emitSubIdx === U(3, 2 bits)) {
    selectedFactor := selectedFactor3
  }

  io.output_valid := currentReady
  io.output_block_idx := commitPtr.resize(8)
  io.output_sub_block_idx := emitSubIdx
  io.output_matrix := matrices(commitPtr)(emitSubIdx)
  io.output_dequant_scale := selectedFactor
  io.done := committed.andR
  io.ready_count := readyCountComb
  io.max_ready_count := maxReady
  io.commit_ptr_dbg := commitPtr

  when(io.clear) {
    factorReady := 0
    committed := 0
    commitPtr := 0
    emitSubIdx := 0
    maxReady := 0
    for(i <- 0 until blockCount) {
      msaReady(i) := 0
      factorMask(i) := 0
      factors(i) := 0
      for(j <- 0 until 4) {
        matrices(i)(j) := 0
      }
    }
  } otherwise {
    when(io.msa_valid) {
      msaReady(io.msa_block_idx)(io.msa_sub_block_idx) := True
      matrices(io.msa_block_idx)(io.msa_sub_block_idx) := io.msa_matrix
    }

    when(io.factor_valid) {
      factorReady(io.factor_block_idx) := True
      factorMask(io.factor_block_idx) := io.factor_valid_mask
      factors(io.factor_block_idx) := io.factor_data
    }

    when(readyCountComb > maxReady) {
      maxReady := readyCountComb
    }

    when(io.output_valid && io.output_ready) {
      when(currentLastSub) {
        committed(commitPtr) := True
        emitSubIdx := 0
        when(commitPtr =/= U(blockCount - 1, blockIdxWidth bits)) {
          commitPtr := commitPtr + 1
        }
      } otherwise {
        emitSubIdx := emitSubIdx + 1
      }
    }
  }
}

object MBPriorQOutputPairJoinBufferGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl_output_pair_join_buffer",
      bitVectorWidthMax = 131072
    ).generateVerilog(new MBPriorQOutputPairJoinBuffer())
  }
}
