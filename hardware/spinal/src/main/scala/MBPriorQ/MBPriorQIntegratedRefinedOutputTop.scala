package MBPriorQ

import spinal.core._

/**
 * Packet-input internal top with refined MultiMSA output packetization.
 *
 * The split-packet input interface feeds MBPriorQRefinedScheduledMultiMSA and
 * MBPriorQOutputBackend, which emit matrix-scale pairs for regular and refined
 * blocks.
 */
class MBPriorQIntegratedRefinedOutputTop(
  blockCount: Int = 16,
  weightNum: Int = 16,
  inputNum: Int = 16,
  laneCount: Int = 16,
  outputBufferCapacity: Int = 16,
  fpuCount: Int = 64,
  fpuOpLatency: Int = 3
) extends Component {
  require(blockCount > 0, "blockCount must be positive.")
  require(blockCount <= 256, "Packet block index is 8 bits.")
  require(laneCount == blockCount, "This internal top maps one block_idx to one MSA lane group.")
  require(laneCount <= 16, "Current refined MultiMSA wrapper supports up to 16 physical lanes.")

  private val blockIdxWidth = log2Up(blockCount)
  private val weightBitsPerLane = 4 * weightNum
  private val inputBits = 4 * inputNum
  private val outputBitsPerBlock = weightNum * inputNum * 16
  private val extraScalePerRefinedBlock = 3
  private val readyCntWidth = log2Up(blockCount + 1)
  private val busyWidth = log2Up(fpuCount + 1)

  val io = new Bundle {
    val start = in Bool()
    val clear = in Bool()

    val packet_valid = in Bool()
    val packet_type = in UInt(3 bits)
    val packet_block_idx = in UInt(blockIdxWidth bits)
    val packet_target_mask = in Bits(blockCount bits)
    val packet_payload = in Bits(1008 bits)

    val output_ready = in Bool()
    val output_valid = out Bool()
    val output_packet = out Bits(1024 bits)

    val done = out Bool()
    val output_overflow = out Bool()
    val scheduler_running = out Bool()
    val pair_max_ready_count = out UInt(readyCntWidth bits)
    val max_busy_fpus = out UInt(busyWidth bits)
    val active_msa_mask = out Bits(laneCount bits)
    val pending_msa_mask = out Bits(laneCount bits)
    val packetizer_busy = out Bool()
  }

  private val PKT_HEAD = U(0, 3 bits)
  private val PKT_WEIGHT_SCALE = U(1, 3 bits)
  private val PKT_INPUT_SCALE = U(2, 3 bits)
  private val PKT_WEIGHT_SCALE_EXT = U(3, 3 bits)
  private val PKT_INPUT_SCALE_EXT = U(4, 3 bits)
  private val PKT_WEIGHT_DATA = U(5, 3 bits)
  private val PKT_INPUT_DATA = U(6, 3 bits)

  private val clearAll = io.clear || io.start

  private val weightMask = Reg(Bits(blockCount bits)) init(0)
  private val inputMask = Reg(Bits(blockCount bits)) init(0)
  private val globalScaleW = Reg(Bits(32 bits)) init(0)
  private val globalScaleI = Reg(Bits(32 bits)) init(0)
  private val weightScaleBase = Reg(Bits(8 * blockCount bits)) init(0)
  private val inputScaleBase = Reg(Bits(8 * blockCount bits)) init(0)
  private val weightScaleExt = Reg(Bits(8 * extraScalePerRefinedBlock * blockCount bits)) init(0)
  private val inputScaleExt = Reg(Bits(8 * extraScalePerRefinedBlock * blockCount bits)) init(0)
  private val weightData = Reg(Bits(weightBitsPerLane * laneCount bits)) init(0)
  private val inputData = Reg(Bits(inputBits bits)) init(0)

  private val packetOneHot = Bits(blockCount bits)
  packetOneHot := B(blockCount bits, default -> false)
  packetOneHot(io.packet_block_idx) := True
  private val packetTargets = Mux(io.packet_target_mask.orR, io.packet_target_mask, packetOneHot)
  private val refinedMask = weightMask | inputMask

  when(clearAll) {
    weightMask := 0
    inputMask := 0
    globalScaleW := 0
    globalScaleI := 0
    weightScaleBase := 0
    inputScaleBase := 0
    weightScaleExt := 0
    inputScaleExt := 0
    weightData := 0
    inputData := 0
  } otherwise {
    when(io.packet_valid) {
      switch(io.packet_type) {
        is(PKT_HEAD) {
          globalScaleW := io.packet_payload(1007 downto 976)
          globalScaleI := io.packet_payload(975 downto 944)
          weightMask := io.packet_payload(943 downto 688).resize(blockCount)
          inputMask := io.packet_payload(687 downto 432).resize(blockCount)
        }
        is(PKT_WEIGHT_SCALE) {
          for(i <- 0 until blockCount) {
            when(packetTargets(i)) {
              weightScaleBase(8 * (i + 1) - 1 downto 8 * i) := io.packet_payload(8 * (i + 1) - 1 downto 8 * i)
            }
          }
        }
        is(PKT_INPUT_SCALE) {
          for(i <- 0 until blockCount) {
            when(packetTargets(i)) {
              inputScaleBase(8 * (i + 1) - 1 downto 8 * i) := io.packet_payload(8 * (i + 1) - 1 downto 8 * i)
            }
          }
        }
        is(PKT_WEIGHT_SCALE_EXT) {
          for(i <- 0 until blockCount) {
            when(packetTargets(i)) {
              val dstStart = 8 * extraScalePerRefinedBlock * i
              val srcStart = 8 * extraScalePerRefinedBlock * i
              weightScaleExt(dstStart + 8 * extraScalePerRefinedBlock - 1 downto dstStart) :=
                io.packet_payload(srcStart + 8 * extraScalePerRefinedBlock - 1 downto srcStart)
            }
          }
        }
        is(PKT_INPUT_SCALE_EXT) {
          for(i <- 0 until blockCount) {
            when(packetTargets(i)) {
              val dstStart = 8 * extraScalePerRefinedBlock * i
              val srcStart = 8 * extraScalePerRefinedBlock * i
              inputScaleExt(dstStart + 8 * extraScalePerRefinedBlock - 1 downto dstStart) :=
                io.packet_payload(srcStart + 8 * extraScalePerRefinedBlock - 1 downto srcStart)
            }
          }
        }
        is(PKT_WEIGHT_DATA) {
          val lane = io.packet_block_idx
          for(i <- 0 until laneCount) {
            when(lane === U(i, blockIdxWidth bits)) {
              weightData(weightBitsPerLane * (i + 1) - 1 downto weightBitsPerLane * i) :=
                io.packet_payload(weightBitsPerLane - 1 downto 0)
            }
          }
        }
        is(PKT_INPUT_DATA) {
          inputData := io.packet_payload(inputBits - 1 downto 0)
        }
      }
    }
  }

  private val scheduler = new MBPriorQPacketScheduler(blockCount, laneCount, outputBufferCapacity, vpuIssueMaskEnable = true)
  private val fpuPool = new MBPriorQSharedFpuPool(blockCount, fpuCount, fpuOpLatency)
  private val factorBuffer = new MBPriorQScaleFactorBuffer(blockCount)
  private val refinedMsa = new MBPriorQRefinedScheduledMultiMSA(weightNum, inputNum, blockCount, laneCount)
  private val outputBackend = new MBPriorQOutputBackend(blockCount, outputBitsPerBlock)

  scheduler.io.start := io.start
  scheduler.io.clear := io.clear
  scheduler.io.refined_mask := refinedMask
  scheduler.io.packet_valid := io.packet_valid
  scheduler.io.packet_type := io.packet_type
  scheduler.io.packet_block_idx := io.packet_block_idx
  scheduler.io.packet_target_mask := io.packet_target_mask
  scheduler.io.msa_issue_ready := refinedMsa.io.issue_ready
  scheduler.io.vpu_issue_ready := fpuPool.io.issue_ready
  scheduler.io.vpu_issue_mask_ready := fpuPool.io.issue_mask_ready
  scheduler.io.msa_done_valid := refinedMsa.io.block_done_valid
  scheduler.io.msa_done_block_idx := refinedMsa.io.block_done_block_idx
  scheduler.io.dequant_done_valid := fpuPool.io.done_valid
  scheduler.io.dequant_done_block_idx := fpuPool.io.done_block_idx
  scheduler.io.output_ready := True

  fpuPool.io.clear := clearAll
  fpuPool.io.issue_valid := scheduler.io.vpu_issue_valid
  fpuPool.io.issue_block_idx := scheduler.io.vpu_issue_block_idx
  fpuPool.io.issue_refined := scheduler.io.vpu_issue_refined
  fpuPool.io.issue_mask_valid := scheduler.io.vpu_issue_mask_valid
  fpuPool.io.issue_mask := scheduler.io.vpu_issue_mask
  fpuPool.io.issue_refined_mask := refinedMask

  factorBuffer.io.clear := clearAll
  factorBuffer.io.issue_valid := scheduler.io.vpu_issue_valid && fpuPool.io.issue_ready
  factorBuffer.io.issue_block_idx := scheduler.io.vpu_issue_block_idx
  factorBuffer.io.issue_mask_valid := scheduler.io.vpu_issue_mask_valid && fpuPool.io.issue_mask_ready
  factorBuffer.io.issue_mask := scheduler.io.vpu_issue_mask
  factorBuffer.io.weight_vmb_mask := weightMask
  factorBuffer.io.input_vmb_mask := inputMask
  factorBuffer.io.global_scale_w := globalScaleW
  factorBuffer.io.global_scale_i := globalScaleI
  factorBuffer.io.weight_scale_base := weightScaleBase
  factorBuffer.io.input_scale_base := inputScaleBase
  factorBuffer.io.weight_scale_ext := weightScaleExt
  factorBuffer.io.input_scale_ext := inputScaleExt
  factorBuffer.io.done_valid := fpuPool.io.done_valid
  factorBuffer.io.done_block_idx := fpuPool.io.done_block_idx

  refinedMsa.io.clear := clearAll
  refinedMsa.io.issue_valid := scheduler.io.msa_issue_valid
  refinedMsa.io.issue_block_idx := scheduler.io.msa_issue_block_idx.resized
  refinedMsa.io.issue_refined := scheduler.io.msa_issue_refined
  refinedMsa.io.weight_data := weightData
  refinedMsa.io.input_data := inputData
  refinedMsa.io.partial_ready := True

  outputBackend.io.clear := clearAll
  outputBackend.io.msa_valid := refinedMsa.io.partial_valid
  outputBackend.io.msa_block_idx := refinedMsa.io.partial_block_idx
  outputBackend.io.msa_sub_block_idx := refinedMsa.io.partial_sub_block_idx
  outputBackend.io.msa_matrix := refinedMsa.io.partial_data
  outputBackend.io.factor_valid := factorBuffer.io.factor_valid
  outputBackend.io.factor_block_idx := factorBuffer.io.factor_block_idx
  outputBackend.io.factor_valid_mask := factorBuffer.io.factor_valid_mask
  outputBackend.io.factor_data := factorBuffer.io.dequant_factors
  outputBackend.io.output_ready := io.output_ready

  io.output_valid := outputBackend.io.output_valid
  io.output_packet := outputBackend.io.output_packet
  io.done := outputBackend.io.done
  io.output_overflow := scheduler.io.output_overflow
  io.scheduler_running := scheduler.io.running
  io.pair_max_ready_count := outputBackend.io.pair_max_ready_count
  io.max_busy_fpus := fpuPool.io.max_busy_fpus
  io.active_msa_mask := refinedMsa.io.active_mask
  io.pending_msa_mask := refinedMsa.io.pending_mask
  io.packetizer_busy := outputBackend.io.packetizer_busy
}

object MBPriorQIntegratedRefinedOutputTopGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl_integrated_refined_output_top",
      bitVectorWidthMax = 131072
    ).generateVerilog(new MBPriorQIntegratedRefinedOutputTop())
  }
}
