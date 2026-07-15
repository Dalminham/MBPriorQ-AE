package MBPriorQ

import spinal.core._

/**
 * Shared FPU-pool scheduler for MBPriorQ dequant-scale reconstruction.
 *
 * The pool accepts per-block VPU work issued by the packet scheduler and models
 * the paper-ratio shared FPU resource: refined blocks require 8 stage-1 scale
 * multiplications and 4 stage-2 scale multiplications; normal blocks require
 * 2 stage-1 multiplications and 1 stage-2 multiplication. Completion is
 * reported per block so it can be joined with MSA output readiness by block
 * index.
 *
 * This module schedules FPU work and completion timing. It intentionally does
 * not compute FP32 numeric results; the existing FPU datapath supplies the
 * arithmetic units.
 */
class MBPriorQSharedFpuPool(
  blockCount: Int = 128,
  fpuCount: Int = 64,
  fpuOpLatency: Int = 3
) extends Component {
  require(blockCount > 0, "Block count must be positive.")
  require(blockCount <= 256, "Packet block index is 8 bits.")
  require(fpuCount > 0, "FPU count must be positive.")
  require(fpuOpLatency > 0, "FPU operation latency must be positive.")

  private val blockIdxWidth = log2Up(blockCount)
  private val busyWidth = log2Up(fpuCount + 1)
  private val blockCountWidth = log2Up(blockCount + 1)
  private val opCountWidth = log2Up(fpuCount + 1)
  private val normalStage1Ops = 2
  private val normalStage2Ops = 1
  private val refinedStage1Ops = 8
  private val refinedStage2Ops = 4

  val io = new Bundle {
    val clear = in Bool()

    val issue_valid = in Bool()
    val issue_ready = out Bool()
    val issue_block_idx = in UInt(blockIdxWidth bits)
    val issue_refined = in Bool()
    val issue_mask_valid = in Bool()
    val issue_mask_ready = out Bool()
    val issue_mask = in Bits(blockCount bits)
    val issue_refined_mask = in Bits(blockCount bits)

    val done_valid = out Bool()
    val done_block_idx = out UInt(blockIdxWidth bits)

    val current_busy_fpus = out UInt(busyWidth bits)
    val max_busy_fpus = out UInt(busyWidth bits)
    val accepted_blocks = out UInt(blockCountWidth bits)
    val completed_blocks = out UInt(blockCountWidth bits)
  }

  private val active = Reg(Bits(blockCount bits)) init(0)
  private val refined = Reg(Bits(blockCount bits)) init(0)
  private val completed = Reg(Bits(blockCount bits)) init(0)
  private val donePending = Reg(Bits(blockCount bits)) init(0)

  private val stage1Launched = Vec(Reg(UInt(opCountWidth bits)) init(0), blockCount)
  private val stage1Done = Vec(Reg(UInt(opCountWidth bits)) init(0), blockCount)
  private val stage2Launched = Vec(Reg(UInt(opCountWidth bits)) init(0), blockCount)
  private val stage2Done = Vec(Reg(UInt(opCountWidth bits)) init(0), blockCount)
  private val stage1Due = Vec(Vec(Reg(UInt(opCountWidth bits)) init(0), fpuOpLatency), blockCount)
  private val stage2Due = Vec(Vec(Reg(UInt(opCountWidth bits)) init(0), fpuOpLatency), blockCount)

  private val maxBusy = Reg(UInt(busyWidth bits)) init(0)

  private def stage1Total(idx: Int): UInt =
    Mux(refined(idx), U(refinedStage1Ops, opCountWidth bits), U(normalStage1Ops, opCountWidth bits))

  private def stage2Total(idx: Int): UInt =
    Mux(refined(idx), U(refinedStage2Ops, opCountWidth bits), U(normalStage2Ops, opCountWidth bits))

  private var busyComb = U(0, busyWidth bits)
  for(i <- 0 until blockCount) {
    for(lat <- 0 until fpuOpLatency) {
      busyComb = busyComb + stage1Due(i)(lat).resize(busyWidth)
      busyComb = busyComb + stage2Due(i)(lat).resize(busyWidth)
    }
  }

  private var acceptedComb = U(0, blockCountWidth bits)
  private var completedComb = U(0, blockCountWidth bits)
  for(i <- 0 until blockCount) {
    acceptedComb = acceptedComb + active(i).asUInt.resize(blockCountWidth)
    completedComb = completedComb + completed(i).asUInt.resize(blockCountWidth)
  }

  io.issue_ready := !active(io.issue_block_idx)
  io.issue_mask_ready := True
  io.done_valid := False
  io.done_block_idx := 0
  io.current_busy_fpus := busyComb
  io.max_busy_fpus := maxBusy
  io.accepted_blocks := acceptedComb
  io.completed_blocks := completedComb

  private var lowerDonePending = False
  for(i <- 0 until blockCount) {
    val pick = donePending(i) && !lowerDonePending
    lowerDonePending = lowerDonePending || donePending(i)
    when(pick) {
      io.done_valid := True
      io.done_block_idx := U(i, blockIdxWidth bits)
    }
  }

  private val stage1Launch = Vec(UInt(opCountWidth bits), blockCount)
  private val stage2Launch = Vec(UInt(opCountWidth bits), blockCount)

  private var freeFpus = (U(fpuCount, busyWidth bits) - busyComb).resize(busyWidth)
  for(i <- 0 until blockCount) {
    val total = stage2Total(i)
    val need = (total - stage2Launched(i)).resize(busyWidth)
    val canLaunch = active(i) && !completed(i) && stage1Done(i) >= stage1Total(i) && stage2Launched(i) < total && freeFpus =/= 0
    val launch = Mux(canLaunch, Mux(need < freeFpus, need, freeFpus), U(0, busyWidth bits))
    stage2Launch(i) := launch.resize(opCountWidth)
    freeFpus = (freeFpus - launch).resize(busyWidth)
  }
  for(i <- 0 until blockCount) {
    val total = stage1Total(i)
    val need = (total - stage1Launched(i)).resize(busyWidth)
    val canLaunch = active(i) && !completed(i) && stage1Launched(i) < total && freeFpus =/= 0
    val launch = Mux(canLaunch, Mux(need < freeFpus, need, freeFpus), U(0, busyWidth bits))
    stage1Launch(i) := launch.resize(opCountWidth)
    freeFpus = (freeFpus - launch).resize(busyWidth)
  }

  private var launchComb = U(0, busyWidth bits)
  for(i <- 0 until blockCount) {
    launchComb = launchComb + stage1Launch(i).resize(busyWidth)
    launchComb = launchComb + stage2Launch(i).resize(busyWidth)
  }

  when(io.clear) {
    active := 0
    refined := 0
    completed := 0
    donePending := 0
    maxBusy := 0
    for(i <- 0 until blockCount) {
      stage1Launched(i) := 0
      stage1Done(i) := 0
      stage2Launched(i) := 0
      stage2Done(i) := 0
      for(lat <- 0 until fpuOpLatency) {
        stage1Due(i)(lat) := 0
        stage2Due(i)(lat) := 0
      }
    }
  } otherwise {
    when(io.issue_valid && io.issue_ready) {
      active(io.issue_block_idx) := True
      refined(io.issue_block_idx) := io.issue_refined
    }
    when(io.issue_mask_valid && io.issue_mask_ready) {
      for(i <- 0 until blockCount) {
        val accept = io.issue_mask(i) && !active(i)
        when(accept) {
          active(i) := True
          refined(i) := io.issue_refined_mask(i)
        }
      }
    }

    when(io.done_valid) {
      donePending(io.done_block_idx) := False
    }

    val busyWithLaunch = (busyComb + launchComb).resize(busyWidth)
    when(busyWithLaunch > maxBusy) {
      maxBusy := busyWithLaunch
    }

    for(i <- 0 until blockCount) {
      val stage1DoneNext = (stage1Done(i) + stage1Due(i)(0)).resize(opCountWidth)
      val stage2DoneNext = (stage2Done(i) + stage2Due(i)(0)).resize(opCountWidth)
      val totalStage2 = stage2Total(i)

      stage1Launched(i) := (stage1Launched(i) + stage1Launch(i)).resize(opCountWidth)
      stage2Launched(i) := (stage2Launched(i) + stage2Launch(i)).resize(opCountWidth)
      stage1Done(i) := stage1DoneNext
      stage2Done(i) := stage2DoneNext

      for(lat <- 0 until fpuOpLatency - 1) {
        stage1Due(i)(lat) := stage1Due(i)(lat + 1)
        stage2Due(i)(lat) := stage2Due(i)(lat + 1)
      }
      stage1Due(i)(fpuOpLatency - 1) := stage1Launch(i)
      stage2Due(i)(fpuOpLatency - 1) := stage2Launch(i)

      when(active(i) && !completed(i) && stage2DoneNext >= totalStage2) {
        completed(i) := True
        donePending(i) := True
      }
    }
  }
}

object MBPriorQSharedFpuPoolGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl_upgraded_scheduler",
      bitVectorWidthMax = 131072
    ).generateVerilog(new MBPriorQSharedFpuPool(128, 64, 3))
  }
}
