package MBPriorQ

import spinal.core._

/**
 * Packet and dataflow scheduler for MBPriorQ.
 *
 * The scheduler tracks per-block metadata and matrix readiness, issues MSA and
 * VPU work independently, joins their completion events, and commits output
 * packets in block-index order subject to output-buffer capacity.
 */
class MBPriorQPacketScheduler(
  blockCount: Int = 128,
  msaCount: Int = 16,
  outputBufferCapacity: Int = 16,
  vpuIssueMaskEnable: Boolean = false
) extends Component {
  require(blockCount > 0, "Block count must be positive.")
  require(blockCount <= 256, "Packet block index is 8 bits.")
  require(msaCount > 0, "MSA lane count must be positive.")
  require(msaCount <= 16, "Paper-scale scheduler assumes up to 16 MSA lanes.")
  require(outputBufferCapacity > 0, "Output buffer capacity must be positive.")

  private val blockIdxWidth = log2Up(blockCount)
  private val readyCntWidth = log2Up(blockCount + 1)
  private val extraScalePerRefinedBlock = 3
  private val baseScalePackets = (msaCount * msaCount + 125) / 126
  private val extScalePackets = (msaCount * msaCount * extraScalePerRefinedBlock + 125) / 126
  private val normalOutputPackets = 4
  private val refinedOutputPackets = 16
  private val packetCntMax = List(baseScalePackets, extScalePackets, msaCount).max
  private val packetCntWidth = log2Up(packetCntMax + 1)
  private val outputPktWidth = log2Up(refinedOutputPackets + 1)

  val io = new Bundle {
    val start = in Bool()
    val clear = in Bool()

    val refined_mask = in Bits(blockCount bits)

    val packet_valid = in Bool()
    val packet_type = in UInt(3 bits)
    val packet_block_idx = in UInt(blockIdxWidth bits)
    val packet_target_mask = in Bits(blockCount bits)

    val msa_issue_ready = in Bool()
    val msa_issue_valid = out Bool()
    val msa_issue_block_idx = out UInt(blockIdxWidth bits)
    val msa_issue_refined = out Bool()

    val vpu_issue_ready = in Bool()
    val vpu_issue_valid = out Bool()
    val vpu_issue_block_idx = out UInt(blockIdxWidth bits)
    val vpu_issue_refined = out Bool()
    val vpu_issue_mask_ready = in Bool()
    val vpu_issue_mask_valid = out Bool()
    val vpu_issue_mask = out Bits(blockCount bits)

    val msa_done_valid = in Bool()
    val msa_done_block_idx = in UInt(blockIdxWidth bits)
    val dequant_done_valid = in Bool()
    val dequant_done_block_idx = in UInt(blockIdxWidth bits)

    val output_ready = in Bool()
    val output_pulse = out Bool()
    val output_block_idx = out UInt(blockIdxWidth bits)
    val output_packet_idx = out UInt(outputPktWidth bits)

    val done = out Bool()
    val running = out Bool()
    val output_overflow = out Bool()
    val current_ready_outputs = out UInt(readyCntWidth bits)
    val max_ready_outputs = out UInt(readyCntWidth bits)
    val commit_ptr_dbg = out UInt(blockIdxWidth bits)
  }

  private val PKT_HEAD = U(0, 3 bits)
  private val PKT_WEIGHT_SCALE = U(1, 3 bits)
  private val PKT_INPUT_SCALE = U(2, 3 bits)
  private val PKT_WEIGHT_SCALE_EXT = U(3, 3 bits)
  private val PKT_INPUT_SCALE_EXT = U(4, 3 bits)
  private val PKT_WEIGHT_DATA = U(5, 3 bits)
  private val PKT_INPUT_DATA = U(6, 3 bits)

  private val running = Reg(Bool()) init(False)
  private val headSeen = Reg(Bits(blockCount bits)) init(0)
  private val inputDataSeen = Reg(Bits(blockCount bits)) init(0)
  private val metadataReady = Reg(Bits(blockCount bits)) init(0)
  private val matrixDataReady = Reg(Bits(blockCount bits)) init(0)
  private val msaIssued = Reg(Bits(blockCount bits)) init(0)
  private val vpuIssued = Reg(Bits(blockCount bits)) init(0)
  private val msaDone = Reg(Bits(blockCount bits)) init(0)
  private val dequantDone = Reg(Bits(blockCount bits)) init(0)
  private val committed = Reg(Bits(blockCount bits)) init(0)

  private val weightScalePackets = Vec(Reg(UInt(packetCntWidth bits)) init(0), blockCount)
  private val inputScalePackets = Vec(Reg(UInt(packetCntWidth bits)) init(0), blockCount)
  private val weightScaleExtPackets = Vec(Reg(UInt(packetCntWidth bits)) init(0), blockCount)
  private val inputScaleExtPackets = Vec(Reg(UInt(packetCntWidth bits)) init(0), blockCount)
  private val weightDataPackets = Vec(Reg(UInt(packetCntWidth bits)) init(0), blockCount)

  private val commitPtr = Reg(UInt(blockIdxWidth bits)) init(0)
  private val outputPacketCounter = Reg(UInt(outputPktWidth bits)) init(0)
  private val maxReadyOutputs = Reg(UInt(readyCntWidth bits)) init(0)
  private val overflowSeen = Reg(Bool()) init(False)

  private val packetOneHot = Bits(blockCount bits)
  packetOneHot := B(blockCount bits, default -> false)
  packetOneHot(io.packet_block_idx) := True
  private val packetTargets = Mux(io.packet_target_mask.orR, io.packet_target_mask, packetOneHot)

  private val outputReadyBits = (msaDone & dequantDone) & ~committed
  private var readyCountComb = U(0, readyCntWidth bits)
  for(i <- 0 until blockCount) {
    readyCountComb = readyCountComb + outputReadyBits(i).asUInt.resize(readyCntWidth)
  }

  io.current_ready_outputs := readyCountComb
  io.max_ready_outputs := maxReadyOutputs
  io.output_overflow := overflowSeen
  io.running := running
  io.done := running && committed.andR
  io.commit_ptr_dbg := commitPtr

  io.msa_issue_valid := False
  io.msa_issue_block_idx := 0
  io.msa_issue_refined := False
  io.vpu_issue_valid := False
  io.vpu_issue_block_idx := 0
  io.vpu_issue_refined := False
  io.vpu_issue_mask_valid := False
  io.vpu_issue_mask := 0
  io.output_pulse := False
  io.output_block_idx := commitPtr
  io.output_packet_idx := outputPacketCounter

  when(io.clear) {
    running := False
    headSeen := 0
    inputDataSeen := 0
    metadataReady := 0
    matrixDataReady := 0
    msaIssued := 0
    vpuIssued := 0
    msaDone := 0
    dequantDone := 0
    committed := 0
    commitPtr := 0
    outputPacketCounter := 0
    maxReadyOutputs := 0
    overflowSeen := False
    for(i <- 0 until blockCount) {
      weightScalePackets(i) := 0
      inputScalePackets(i) := 0
      weightScaleExtPackets(i) := 0
      inputScaleExtPackets(i) := 0
      weightDataPackets(i) := 0
    }
  } elsewhen(io.start) {
    running := True
    headSeen := 0
    inputDataSeen := 0
    metadataReady := 0
    matrixDataReady := 0
    msaIssued := 0
    vpuIssued := 0
    msaDone := 0
    dequantDone := 0
    committed := 0
    commitPtr := 0
    outputPacketCounter := 0
    maxReadyOutputs := 0
    overflowSeen := False
    for(i <- 0 until blockCount) {
      weightScalePackets(i) := 0
      inputScalePackets(i) := 0
      weightScaleExtPackets(i) := 0
      inputScaleExtPackets(i) := 0
      weightDataPackets(i) := 0
    }
  } otherwise {
    when(running) {
      when(io.packet_valid) {
        for(i <- 0 until blockCount) {
          val targeted = packetTargets(i)
          val refined = io.refined_mask(i)

          val headNext = headSeen(i) || (targeted && io.packet_type === PKT_HEAD)
          val inputSeenNext = inputDataSeen(i) || (targeted && io.packet_type === PKT_INPUT_DATA)

          val weightScaleInc = targeted && io.packet_type === PKT_WEIGHT_SCALE && weightScalePackets(i) < U(baseScalePackets, packetCntWidth bits)
          val inputScaleInc = targeted && io.packet_type === PKT_INPUT_SCALE && inputScalePackets(i) < U(baseScalePackets, packetCntWidth bits)
          val weightScaleExtInc = targeted && refined && io.packet_type === PKT_WEIGHT_SCALE_EXT && weightScaleExtPackets(i) < U(extScalePackets, packetCntWidth bits)
          val inputScaleExtInc = targeted && refined && io.packet_type === PKT_INPUT_SCALE_EXT && inputScaleExtPackets(i) < U(extScalePackets, packetCntWidth bits)
          val weightDataInc = targeted && io.packet_type === PKT_WEIGHT_DATA && weightDataPackets(i) < U(msaCount, packetCntWidth bits)

          val weightScaleNext = weightScalePackets(i) + weightScaleInc.asUInt.resize(packetCntWidth)
          val inputScaleNext = inputScalePackets(i) + inputScaleInc.asUInt.resize(packetCntWidth)
          val weightScaleExtNext = weightScaleExtPackets(i) + weightScaleExtInc.asUInt.resize(packetCntWidth)
          val inputScaleExtNext = inputScaleExtPackets(i) + inputScaleExtInc.asUInt.resize(packetCntWidth)
          val weightDataNext = weightDataPackets(i) + weightDataInc.asUInt.resize(packetCntWidth)

          when(targeted && io.packet_type === PKT_HEAD) {
            headSeen(i) := True
          }
          when(weightScaleInc) {
            weightScalePackets(i) := weightScaleNext
          }
          when(inputScaleInc) {
            inputScalePackets(i) := inputScaleNext
          }
          when(weightScaleExtInc) {
            weightScaleExtPackets(i) := weightScaleExtNext
          }
          when(inputScaleExtInc) {
            inputScaleExtPackets(i) := inputScaleExtNext
          }
          when(weightDataInc) {
            weightDataPackets(i) := weightDataNext
          }
          when(targeted && io.packet_type === PKT_INPUT_DATA) {
            inputDataSeen(i) := True
          }

          val baseMetadataReady =
            headNext &&
              weightScaleNext >= U(baseScalePackets, packetCntWidth bits) &&
              inputScaleNext >= U(baseScalePackets, packetCntWidth bits)
          val refinedMetadataReady =
            !refined ||
              (weightScaleExtNext >= U(extScalePackets, packetCntWidth bits) &&
                inputScaleExtNext >= U(extScalePackets, packetCntWidth bits))
          when(baseMetadataReady && refinedMetadataReady) {
            metadataReady(i) := True
          }
          when(weightDataNext >= U(msaCount, packetCntWidth bits) && inputSeenNext) {
            matrixDataReady(i) := True
          }
        }
      }

      when(io.msa_done_valid) {
        msaDone(io.msa_done_block_idx) := True
      }
      when(io.dequant_done_valid) {
        dequantDone(io.dequant_done_block_idx) := True
      }

      if(vpuIssueMaskEnable) {
        val vpuIssueCandidates = metadataReady & ~vpuIssued
        io.vpu_issue_mask_valid := vpuIssueCandidates.orR
        io.vpu_issue_mask := vpuIssueCandidates
        when(io.vpu_issue_mask_valid && io.vpu_issue_mask_ready) {
          vpuIssued := vpuIssued | vpuIssueCandidates
        }
      } else {
        var lowerVpuReady = False
        for(i <- 0 until blockCount) {
          val canIssue = metadataReady(i) && !vpuIssued(i)
          val pick = canIssue && !lowerVpuReady
          lowerVpuReady = lowerVpuReady || canIssue
          when(pick) {
            io.vpu_issue_valid := True
            io.vpu_issue_block_idx := U(i, blockIdxWidth bits)
            io.vpu_issue_refined := io.refined_mask(i)
            when(io.vpu_issue_ready) {
              vpuIssued(i) := True
            }
          }
        }
      }

      var lowerMsaReady = False
      for(i <- 0 until blockCount) {
        val canIssue = metadataReady(i) && matrixDataReady(i) && !msaIssued(i)
        val pick = canIssue && !lowerMsaReady
        lowerMsaReady = lowerMsaReady || canIssue
        when(pick) {
          io.msa_issue_valid := True
          io.msa_issue_block_idx := U(i, blockIdxWidth bits)
          io.msa_issue_refined := io.refined_mask(i)
          when(io.msa_issue_ready) {
            msaIssued(i) := True
          }
        }
      }

      when(readyCountComb > maxReadyOutputs) {
        maxReadyOutputs := readyCountComb
      }
      when(readyCountComb > U(outputBufferCapacity, readyCntWidth bits)) {
        overflowSeen := True
      }

      val commitReady = outputReadyBits(commitPtr)
      val commitRefined = io.refined_mask(commitPtr)
      val commitPacketTotal = Mux(
        commitRefined,
        U(refinedOutputPackets, outputPktWidth bits),
        U(normalOutputPackets, outputPktWidth bits)
      )

      when(commitReady && io.output_ready) {
        io.output_pulse := True
        io.output_block_idx := commitPtr
        io.output_packet_idx := outputPacketCounter
        when(outputPacketCounter === commitPacketTotal - 1) {
          committed(commitPtr) := True
          outputPacketCounter := 0
          when(commitPtr =/= U(blockCount - 1, blockIdxWidth bits)) {
            commitPtr := commitPtr + 1
          }
        } otherwise {
          outputPacketCounter := outputPacketCounter + 1
        }
      }
    }
  }
}

object MBPriorQPacketSchedulerGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl_packet_scheduler",
      bitVectorWidthMax = 131072
    ).generateVerilog(new MBPriorQPacketScheduler(128, 16, 16))
  }
}
