package MBPriorQ

import spinal.core._

/**
 * Public MBPriorQ accelerator top.
 *
 * This class keeps the public external IO and 1024-bit input packet format,
 * and instantiates the paper-scale metadata-handling path: 16 logical metadata
 * blocks are aligned with 16 MSA lane groups.
 *
 * External input packet format:
 *   [1023:1016] packet type
 *   [1015:1008] packet block index / scale-entry start index
 *   [1007:0]    payload
 *
 * External output packet format is produced by MBPriorQOutputPacketizer:
 *   one normal block -> one matrix-scale group -> five 1024-bit packets
 *   one refined block -> four matrix-scale groups -> twenty 1024-bit packets
 */
class MBPriorQ(
  PE_Nums: Int = 16,
  WeightNum: Int = 16,
  InputNum: Int = 16,
  MSA_Num: Int = 16
) extends Component {
  require(PE_Nums == 16, "Current paper-scale MBPriorQ top uses 16x16 MSAs.")
  require(WeightNum == 16, "Current output packetizer expects a 16x16 BF16 MSA result.")
  require(InputNum == 16, "Current output packetizer expects a 16x16 BF16 MSA result.")
  require(MSA_Num == 16, "Current paper-scale MBPriorQ top uses 16 MSA lane groups.")

  val io = new Bundle {
    val MSA_EN = in Bool()
    val data_packet = in Bits(1024 bits)
    val data_valid = in Bool()

    val output_pulse = out Bool()
    val output_packet = out Bits(1024 bits)
    val output_ready = in Bool()
  }

  private val top = new MBPriorQExternal1024PacketTop(
    blockCount = MSA_Num,
    weightNum = WeightNum,
    inputNum = InputNum,
    laneCount = MSA_Num,
    outputBufferCapacity = 16,
    fpuCount = 64,
    fpuOpLatency = 3
  )

  top.io.MSA_EN := io.MSA_EN
  top.io.data_packet := io.data_packet
  top.io.data_valid := io.data_valid
  top.io.output_ready := io.output_ready

  io.output_pulse := top.io.output_pulse
  io.output_packet := top.io.output_packet
}

object MBPriorQGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl",
      bitVectorWidthMax = 131072
    ).generateVerilog(new MBPriorQ())
  }
}
