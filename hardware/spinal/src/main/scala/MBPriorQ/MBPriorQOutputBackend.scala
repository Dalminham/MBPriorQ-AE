package MBPriorQ

import spinal.core._

/**
 * Output backend for the software-derived MBPriorQ packet contract.
 *
 * It joins MSA partial matrices with matching dequant factors and then
 * packetizes each matrix-scale pair into 1024-bit packets.
 */
class MBPriorQOutputBackend(
  blockCount: Int = 16,
  matrixBits: Int = 16 * 16 * 16
) extends Component {
  require(matrixBits == 16 * 16 * 16, "MBPriorQOutputPacketizer currently expects a 16x16 BF16 matrix.")

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
    val output_packet = out Bits(1024 bits)

    val done = out Bool()
    val pair_ready_count = out UInt(readyCntWidth bits)
    val pair_max_ready_count = out UInt(readyCntWidth bits)
    val packetizer_busy = out Bool()
  }

  private val pairJoin = new MBPriorQOutputPairJoinBuffer(blockCount, matrixBits)
  private val packetizer = new MBPriorQOutputPacketizer

  pairJoin.io.clear := io.clear
  pairJoin.io.msa_valid := io.msa_valid
  pairJoin.io.msa_block_idx := io.msa_block_idx
  pairJoin.io.msa_sub_block_idx := io.msa_sub_block_idx
  pairJoin.io.msa_matrix := io.msa_matrix
  pairJoin.io.factor_valid := io.factor_valid
  pairJoin.io.factor_block_idx := io.factor_block_idx
  pairJoin.io.factor_valid_mask := io.factor_valid_mask
  pairJoin.io.factor_data := io.factor_data
  pairJoin.io.output_ready := packetizer.io.input_ready

  packetizer.io.clear := io.clear
  packetizer.io.input_valid := pairJoin.io.output_valid
  packetizer.io.input_block_idx := pairJoin.io.output_block_idx
  packetizer.io.input_sub_block_idx := pairJoin.io.output_sub_block_idx
  packetizer.io.input_matrix := pairJoin.io.output_matrix
  packetizer.io.input_dequant_scale := pairJoin.io.output_dequant_scale
  packetizer.io.output_ready := io.output_ready

  io.output_valid := packetizer.io.output_valid
  io.output_packet := packetizer.io.output_packet
  io.done := pairJoin.io.done && !packetizer.io.busy
  io.pair_ready_count := pairJoin.io.ready_count
  io.pair_max_ready_count := pairJoin.io.max_ready_count
  io.packetizer_busy := packetizer.io.busy
}

object MBPriorQOutputBackendGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl_output_backend",
      bitVectorWidthMax = 131072
    ).generateVerilog(new MBPriorQOutputBackend())
  }
}
