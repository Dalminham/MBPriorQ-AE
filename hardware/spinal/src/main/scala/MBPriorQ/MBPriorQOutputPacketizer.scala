package MBPriorQ

import spinal.core._

/**
 * Packetizes one MBPriorQ output matrix-scale pair into 1024-bit packets.
 *
 * Logical pair:
 *   block_index + sub_block_idx + 16x16 BF16 partial matrix + FP32 dequant scale
 *
 * Physical packet format:
 *   [1023:1016] packet type, fixed 0x80 for output matrix-scale packets
 *   [1015:1008] block_index
 *   [1007:1006] sub_block_idx, 0 for normal, 0..3 for refined
 *   [1005:1003] segment_idx, 0..4
 *   [1002]      last segment flag
 *   [1001:1000] reserved
 *   [999:0]     payload
 *
 * Segments 0..3 carry 1000 bits each of the 4096-bit BF16 matrix. Segment 4
 * carries the remaining 96 matrix bits and the 32-bit FP32 dequant scale.
 */
class MBPriorQOutputPacketizer extends Component {
  private val matrixBits = 16 * 16 * 16
  private val scaleBits = 32
  private val payloadBits = 1000
  private val segmentCount = 5

  val io = new Bundle {
    val clear = in Bool()

    val input_valid = in Bool()
    val input_ready = out Bool()
    val input_block_idx = in UInt(8 bits)
    val input_sub_block_idx = in UInt(2 bits)
    val input_matrix = in Bits(matrixBits bits)
    val input_dequant_scale = in Bits(scaleBits bits)

    val output_ready = in Bool()
    val output_valid = out Bool()
    val output_packet = out Bits(1024 bits)
    val busy = out Bool()
  }

  private val active = Reg(Bool()) init(False)
  private val segmentIdx = Reg(UInt(3 bits)) init(0)
  private val blockIdx = Reg(UInt(8 bits)) init(0)
  private val subBlockIdx = Reg(UInt(2 bits)) init(0)
  private val matrix = Reg(Bits(matrixBits bits)) init(0)
  private val dequantScale = Reg(Bits(scaleBits bits)) init(0)

  private val fireInput = io.input_valid && io.input_ready
  private val fireOutput = io.output_valid && io.output_ready
  private val isLastSegment = segmentIdx === U(segmentCount - 1, 3 bits)

  io.input_ready := !active
  io.output_valid := active
  io.busy := active

  private val payload = Bits(payloadBits bits)
  payload := 0
  switch(segmentIdx) {
    is(U(0, 3 bits)) {
      payload := matrix(999 downto 0)
    }
    is(U(1, 3 bits)) {
      payload := matrix(1999 downto 1000)
    }
    is(U(2, 3 bits)) {
      payload := matrix(2999 downto 2000)
    }
    is(U(3, 3 bits)) {
      payload := matrix(3999 downto 3000)
    }
    is(U(4, 3 bits)) {
      payload(95 downto 0) := matrix(4095 downto 4000)
      payload(127 downto 96) := dequantScale
    }
  }

  io.output_packet := B"8'h80" ##
    blockIdx.asBits ##
    subBlockIdx.asBits ##
    segmentIdx.asBits ##
    isLastSegment.asBits ##
    B(2 bits, default -> false) ##
    payload

  when(io.clear) {
    active := False
    segmentIdx := 0
    blockIdx := 0
    subBlockIdx := 0
    matrix := 0
    dequantScale := 0
  } otherwise {
    when(fireInput) {
      active := True
      segmentIdx := 0
      blockIdx := io.input_block_idx
      subBlockIdx := io.input_sub_block_idx
      matrix := io.input_matrix
      dequantScale := io.input_dequant_scale
    } elsewhen(fireOutput) {
      when(isLastSegment) {
        active := False
        segmentIdx := 0
      } otherwise {
        segmentIdx := segmentIdx + 1
      }
    }
  }
}

object MBPriorQOutputPacketizerGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl_output_packetizer",
      bitVectorWidthMax = 131072
    ).generateVerilog(new MBPriorQOutputPacketizer)
  }
}
