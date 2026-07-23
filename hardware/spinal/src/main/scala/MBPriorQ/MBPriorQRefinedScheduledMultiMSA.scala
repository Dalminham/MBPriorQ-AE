package MBPriorQ

import spinal.core._

/**
 * Refined-aware scheduled MultiMSA wrapper.
 *
 * The wrapper treats the 16 MSA instances as a physical pool. A regular logical
 * block consumes one physical lane and emits sub_block_idx 0. A refined logical
 * block consumes four physical lanes, one per 4-element K sub-block, and emits
 * four partial matrices with the same block_idx and sub_block_idx 0..3. The
 * output backend pairs each event with its matching dequant factor.
 */
class MBPriorQRefinedScheduledMultiMSA(
  weightNum: Int = 16,
  inputNum: Int = 16,
  blockCount: Int = 16,
  physicalLaneCount: Int = 16
) extends Component {
  require(weightNum == 16, "The refined K-slice contract is defined for 16 weight elements.")
  require(inputNum == 16, "The refined K-slice contract is defined for 16 input elements.")
  require(blockCount > 0, "blockCount must be positive.")
  require(blockCount <= 256, "block_index is 8 bits.")
  require(physicalLaneCount >= 4, "A refined block requires four physical MSA lanes.")
  require(physicalLaneCount <= 16, "The current paper-scale wrapper assumes up to 16 physical MSA lanes.")

  private val blockIdxWidth = log2Up(blockCount)
  private val countWidth = log2Up(physicalLaneCount + 1)
  private val subBlockCount = 4
  private val elementsPerSubBlock = 4
  private val weightBitsPerBlock = 4 * weightNum
  private val inputBits = 4 * inputNum
  private val outputBitsPerBlock = weightNum * inputNum * 16
  private val zeroSubMask = B(subBlockCount bits, default -> false)

  val io = new Bundle {
    val clear = in Bool()

    val issue_valid = in Bool()
    val issue_ready = out Bool()
    val issue_block_idx = in UInt(blockIdxWidth bits)
    val issue_refined = in Bool()

    val weight_data = in Bits(weightBitsPerBlock * blockCount bits)
    val input_data = in Bits(inputBits bits)

    val partial_valid = out Bool()
    val partial_ready = in Bool()
    val partial_block_idx = out UInt(blockIdxWidth bits)
    val partial_sub_block_idx = out UInt(2 bits)
    val partial_data = out Bits(outputBitsPerBlock bits)

    val block_done_valid = out Bool()
    val block_done_block_idx = out UInt(blockIdxWidth bits)

    val active_mask = out Bits(physicalLaneCount bits)
    val pending_mask = out Bits(physicalLaneCount bits)
    val block_done_pending_mask = out Bits(blockCount bits)
  }

  private def selectSubBlock(data: Bits, subIdx: Int, elementCount: Int): Bits = {
    val masked = Bits(4 * elementCount bits)
    masked := 0
    for(j <- 0 until elementsPerSubBlock) {
      val idx = subIdx * elementsPerSubBlock + j
      masked(4 * (idx + 1) - 1 downto 4 * idx) := data(4 * (idx + 1) - 1 downto 4 * idx)
    }
    masked
  }

  private val active = Reg(Bits(physicalLaneCount bits)) init(0)
  private val pending = Reg(Bits(physicalLaneCount bits)) init(0)
  private val laneRefined = Reg(Bits(physicalLaneCount bits)) init(0)
  private val laneOutputCaptured = Reg(Bits(physicalLaneCount bits)) init(0)
  private val laneBlockIdx = Vec(Reg(UInt(blockIdxWidth bits)) init(0), physicalLaneCount)
  private val laneSubBlockIdx = Vec(Reg(UInt(2 bits)) init(0), physicalLaneCount)
  private val laneWeight = Vec(Reg(Bits(weightBitsPerBlock bits)) init(0), physicalLaneCount)
  private val laneInput = Vec(Reg(Bits(inputBits bits)) init(0), physicalLaneCount)
  private val laneDoneData = Vec(Reg(Bits(outputBitsPerBlock bits)) init(0), physicalLaneCount)

  private val requiredSubMask = Vec(Reg(Bits(subBlockCount bits)) init(0), blockCount)
  private val emittedSubMask = Vec(Reg(Bits(subBlockCount bits)) init(0), blockCount)
  private val blockDonePending = Reg(Bits(blockCount bits)) init(0)

  private val weightByBlock = Vec(Bits(weightBitsPerBlock bits), blockCount)
  for(i <- 0 until blockCount) {
    weightByBlock(i) := io.weight_data(weightBitsPerBlock * (i + 1) - 1 downto weightBitsPerBlock * i)
  }
  private val selectedWeight = weightByBlock(io.issue_block_idx)
  private val selectedInput = io.input_data

  private val weightSub = Vec(Bits(weightBitsPerBlock bits), subBlockCount)
  private val inputSub = Vec(Bits(inputBits bits), subBlockCount)
  for(i <- 0 until subBlockCount) {
    weightSub(i) := selectSubBlock(selectedWeight, i, weightNum)
    inputSub(i) := selectSubBlock(selectedInput, i, inputNum)
  }

  private val freeMask = ~(active | pending)
  private var freeCount = U(0, countWidth bits)
  for(i <- 0 until physicalLaneCount) {
    freeCount = freeCount + freeMask(i).asUInt.resize(countWidth)
  }

  private def firstOne(mask: Bits): Bits = {
    val oneHot = Bits(physicalLaneCount bits)
    oneHot := 0
    var lowerSeen = False
    for(i <- 0 until physicalLaneCount) {
      val pick = mask(i) && !lowerSeen
      oneHot(i) := pick
      lowerSeen = lowerSeen || mask(i)
    }
    oneHot
  }

  private val normalAllocMask = firstOne(freeMask)
  private val refinedAllocMask = Vec(Bits(physicalLaneCount bits), subBlockCount)
  private var remainingFreeMask = freeMask
  for(i <- 0 until subBlockCount) {
    refinedAllocMask(i) := firstOne(remainingFreeMask)
    remainingFreeMask = remainingFreeMask & ~refinedAllocMask(i)
  }

  io.issue_ready := Mux(io.issue_refined, freeCount >= U(subBlockCount, countWidth bits), freeCount =/= 0)
  io.active_mask := active
  io.pending_mask := pending
  io.block_done_pending_mask := blockDonePending

  private val selectedPartialValid = pending.orR
  private val selectedPartialBlock = UInt(blockIdxWidth bits)
  private val selectedPartialSub = UInt(2 bits)
  private val selectedPartialData = Bits(outputBitsPerBlock bits)
  selectedPartialBlock := 0
  selectedPartialSub := 0
  selectedPartialData := 0

  private var lowerPending = False
  for(i <- 0 until physicalLaneCount) {
    val pick = pending(i) && !lowerPending
    lowerPending = lowerPending || pending(i)
    when(pick) {
      selectedPartialBlock := laneBlockIdx(i)
      selectedPartialSub := laneSubBlockIdx(i)
      selectedPartialData := laneDoneData(i)
    }
  }

  io.partial_valid := selectedPartialValid
  io.partial_block_idx := selectedPartialBlock
  io.partial_sub_block_idx := selectedPartialSub
  io.partial_data := selectedPartialData

  private val partialFire = selectedPartialValid && io.partial_ready
  private val partialClearMask = Bits(physicalLaneCount bits)
  partialClearMask := B(physicalLaneCount bits, default -> false)
  for(i <- 0 until physicalLaneCount) {
    partialClearMask(i) := partialFire &&
      pending(i) &&
      laneBlockIdx(i) === selectedPartialBlock &&
      laneSubBlockIdx(i) === selectedPartialSub
  }

  private val lanes = Array.fill(physicalLaneCount)(new MBPriorQClearableMSA(weightNum, inputNum))
  for(i <- 0 until physicalLaneCount) {
    lanes(i).io.clear := io.clear || partialClearMask(i)
    lanes(i).io.Valid := active(i)
    lanes(i).io.NeedSubCycle := laneRefined(i)
    lanes(i).io.Weight_Forward_Data := laneWeight(i)
    lanes(i).io.Input_Forward_Data := laneInput(i)
    lanes(i).io.OutputReady := !pending(i) || partialClearMask(i)
  }

  io.block_done_valid := blockDonePending.orR
  io.block_done_block_idx := 0
  private var lowerDone = False
  for(i <- 0 until blockCount) {
    val pick = blockDonePending(i) && !lowerDone
    lowerDone = lowerDone || blockDonePending(i)
    when(pick) {
      io.block_done_block_idx := U(i, blockIdxWidth bits)
    }
  }

  when(io.clear) {
    active := 0
    pending := 0
    laneRefined := 0
    laneOutputCaptured := 0
    blockDonePending := 0
    for(i <- 0 until physicalLaneCount) {
      laneBlockIdx(i) := 0
      laneSubBlockIdx(i) := 0
      laneWeight(i) := 0
      laneInput(i) := 0
      laneDoneData(i) := 0
    }
    for(i <- 0 until blockCount) {
      requiredSubMask(i) := 0
      emittedSubMask(i) := 0
    }
  } otherwise {
    when(io.issue_valid && io.issue_ready) {
      requiredSubMask(io.issue_block_idx) := Mux(io.issue_refined, B"4'b1111", B"4'b0001")
      emittedSubMask(io.issue_block_idx) := zeroSubMask

      for(i <- 0 until physicalLaneCount) {
        val allocNormal = !io.issue_refined && normalAllocMask(i)
        val allocRefined = io.issue_refined && (
          refinedAllocMask(0)(i) ||
            refinedAllocMask(1)(i) ||
            refinedAllocMask(2)(i) ||
            refinedAllocMask(3)(i)
        )
        val alloc = allocNormal || allocRefined

        val selectedSubIdx = UInt(2 bits)
        selectedSubIdx := 0
        val selectedSubWeight = Bits(weightBitsPerBlock bits)
        val selectedSubInput = Bits(inputBits bits)
        selectedSubWeight := weightSub(0)
        selectedSubInput := inputSub(0)
        when(refinedAllocMask(1)(i)) {
          selectedSubIdx := 1
          selectedSubWeight := weightSub(1)
          selectedSubInput := inputSub(1)
        }
        when(refinedAllocMask(2)(i)) {
          selectedSubIdx := 2
          selectedSubWeight := weightSub(2)
          selectedSubInput := inputSub(2)
        }
        when(refinedAllocMask(3)(i)) {
          selectedSubIdx := 3
          selectedSubWeight := weightSub(3)
          selectedSubInput := inputSub(3)
        }

        when(alloc) {
          active(i) := True
          laneRefined(i) := io.issue_refined
          laneBlockIdx(i) := io.issue_block_idx
          laneSubBlockIdx(i) := Mux(io.issue_refined, selectedSubIdx, U(0, 2 bits))
          laneWeight(i) := Mux(io.issue_refined, selectedSubWeight, selectedWeight)
          laneInput(i) := Mux(io.issue_refined, selectedSubInput, selectedInput)
          laneOutputCaptured(i) := False
        }
      }
    }

    when(partialFire) {
      val subOneHot = Bits(subBlockCount bits)
      subOneHot := B(subBlockCount bits, default -> false)
      subOneHot(selectedPartialSub) := True
      val blockIdx = selectedPartialBlock
      val emittedNext = emittedSubMask(blockIdx) | subOneHot
      emittedSubMask(blockIdx) := emittedNext
      when((emittedNext & requiredSubMask(blockIdx)) === requiredSubMask(blockIdx)) {
        blockDonePending(blockIdx) := True
      }

      for(i <- 0 until physicalLaneCount) {
        when(pending(i) &&
          laneBlockIdx(i) === selectedPartialBlock &&
          laneSubBlockIdx(i) === selectedPartialSub) {
          pending(i) := False
        }
      }
    }

    when(io.block_done_valid) {
      blockDonePending(io.block_done_block_idx) := False
      requiredSubMask(io.block_done_block_idx) := zeroSubMask
      emittedSubMask(io.block_done_block_idx) := zeroSubMask
    }

    for(i <- 0 until physicalLaneCount) {
      when(lanes(i).io.OutputValid && !pending(i) && !laneOutputCaptured(i)) {
        laneDoneData(i) := lanes(i).io.OutputData
        pending(i) := True
        active(i) := False
        laneOutputCaptured(i) := True
      }
    }
  }
}

object MBPriorQRefinedScheduledMultiMSAGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl_refined_scheduled_multimma",
      bitVectorWidthMax = 131072
    ).generateVerilog(new MBPriorQRefinedScheduledMultiMSA(16, 16, 16, 16))
  }
}
