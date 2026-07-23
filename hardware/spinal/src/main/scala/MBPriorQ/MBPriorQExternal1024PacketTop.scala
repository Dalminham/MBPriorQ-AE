package MBPriorQ

import spinal.core._

/**
 * Public 1024-bit input packet wrapper around the validated refined-output top.
 *
 * External input uses the MBPriorQ packet format:
 *   [1023:1016] packet type
 *   [1015:1008] packet block / scale-entry start index
 *   [1007:0]    payload
 *
 * Internally, this wrapper collects parser state and replays it into
 * MBPriorQIntegratedRefinedOutputTop's split-packet interface. It preserves the
 * scale-entry addressing rule for input packets and keeps grouped metadata
 * reuse hidden inside the wrapper replay.
 */
class MBPriorQExternal1024PacketTop(
  blockCount: Int = 16,
  weightNum: Int = 16,
  inputNum: Int = 16,
  laneCount: Int = 16,
  outputBufferCapacity: Int = 16,
  fpuCount: Int = 64,
  fpuOpLatency: Int = 3
) extends Component {
  require(blockCount == laneCount, "This wrapper maps one logical block to one internal MSA lane group.")
  require(blockCount <= 16, "The current validated refined-output top supports up to 16 logical blocks.")

  private val blockIdxWidth = log2Up(blockCount)
  private val weightBitsPerLane = 4 * weightNum
  private val inputBits = 4 * inputNum
  private val extraScalePerRefinedBlock = 3
  private val extEntryCount = blockCount * extraScalePerRefinedBlock
  private val baseScalePackets = (laneCount * laneCount + 125) / 126
  private val extScalePackets = (laneCount * laneCount * extraScalePerRefinedBlock + 125) / 126
  private val replayCountWidth = log2Up(List(baseScalePackets, extScalePackets, laneCount).max + 1)
  private val busyWidth = log2Up(fpuCount + 1)
  private val readyCntWidth = log2Up(blockCount + 1)

  val io = new Bundle {
    val MSA_EN = in Bool()
    val data_packet = in Bits(1024 bits)
    val data_valid = in Bool()

    val output_ready = in Bool()
    val output_pulse = out Bool()
    val output_packet = out Bits(1024 bits)

    val done = out Bool()
    val adapter_running = out Bool()
    val replay_done = out Bool()
    val output_overflow = out Bool()
    val pair_max_ready_count = out UInt(readyCntWidth bits)
    val max_busy_fpus = out UInt(busyWidth bits)
  }

  private val OLD_HEAD = B"8'h00"
  private val OLD_WEIGHT_DATA = B"8'h01"
  private val OLD_INPUT_DATA = B"8'h02"
  private val OLD_WEIGHT_SCALE = B"8'h03"
  private val OLD_WEIGHT_SCALE_EXT = B"8'h04"
  private val OLD_INPUT_SCALE = B"8'h05"
  private val OLD_INPUT_SCALE_EXT = B"8'h06"

  private val NEW_HEAD = U(0, 3 bits)
  private val NEW_WEIGHT_SCALE = U(1, 3 bits)
  private val NEW_INPUT_SCALE = U(2, 3 bits)
  private val NEW_WEIGHT_SCALE_EXT = U(3, 3 bits)
  private val NEW_INPUT_SCALE_EXT = U(4, 3 bits)
  private val NEW_WEIGHT_DATA = U(5, 3 bits)
  private val NEW_INPUT_DATA = U(6, 3 bits)

  private object State extends SpinalEnum {
    val IDLE, COLLECT, START_INTERNAL, SEND_HEAD, SEND_WEIGHT_SCALE, SEND_INPUT_SCALE,
      SEND_WEIGHT_EXT, SEND_INPUT_EXT, SEND_WEIGHT_DATA, SEND_INPUT_DATA, RUN = newElement()
  }

  private val state = Reg(State()) init(State.IDLE)
  private val headPayload = Reg(Bits(1008 bits)) init(0)
  private val weightMask = Reg(Bits(blockCount bits)) init(0)
  private val inputMask = Reg(Bits(blockCount bits)) init(0)
  private val weightScaleBase = Vec(Reg(Bits(8 bits)) init(0), blockCount)
  private val inputScaleBase = Vec(Reg(Bits(8 bits)) init(0), blockCount)
  private val weightScaleExt = Vec(Reg(Bits(8 bits)) init(0), extEntryCount)
  private val inputScaleExt = Vec(Reg(Bits(8 bits)) init(0), extEntryCount)
  private val weightData = Vec(Reg(Bits(weightBitsPerLane bits)) init(0), blockCount)
  private val inputData = Reg(Bits(inputBits bits)) init(0)
  private val replayCount = Reg(UInt(replayCountWidth bits)) init(0)
  private val replayBlock = Reg(UInt(blockIdxWidth bits)) init(0)
  private val replayRepeat = Reg(UInt(replayCountWidth bits)) init(0)

  private val pktType = io.data_packet(1023 downto 1016)
  private val pktBlockIdx = io.data_packet(1015 downto 1008).asUInt
  private val pktPayload = io.data_packet(1007 downto 0)
  private val refinedMask = weightMask | inputMask
  private val allMask = B(blockCount bits, default -> true)

  private def resetStoredState(): Unit = {
    headPayload := 0
    weightMask := 0
    inputMask := 0
    inputData := 0
    for(i <- 0 until blockCount) {
      weightScaleBase(i) := 0
      inputScaleBase(i) := 0
      weightData(i) := 0
    }
    for(i <- 0 until extEntryCount) {
      weightScaleExt(i) := 0
      inputScaleExt(i) := 0
    }
  }

  private val internalPayload = Bits(1008 bits)
  internalPayload := 0

  private val internalPacketValid = Bool()
  private val internalPacketType = UInt(3 bits)
  private val internalPacketBlockIdx = UInt(blockIdxWidth bits)
  private val internalPacketTargetMask = Bits(blockCount bits)
  internalPacketValid := False
  internalPacketType := 0
  internalPacketBlockIdx := 0
  internalPacketTargetMask := 0

  private val top = new MBPriorQIntegratedRefinedOutputTop(
    blockCount = blockCount,
    weightNum = weightNum,
    inputNum = inputNum,
    laneCount = laneCount,
    outputBufferCapacity = outputBufferCapacity,
    fpuCount = fpuCount,
    fpuOpLatency = fpuOpLatency
  )

  top.io.clear := state === State.IDLE
  top.io.start := state === State.START_INTERNAL
  top.io.packet_valid := internalPacketValid
  top.io.packet_type := internalPacketType
  top.io.packet_block_idx := internalPacketBlockIdx
  top.io.packet_target_mask := internalPacketTargetMask
  top.io.packet_payload := internalPayload
  top.io.output_ready := io.output_ready

  io.output_packet := top.io.output_packet
  io.output_pulse := top.io.output_valid && io.output_ready
  io.done := state === State.RUN && top.io.done
  io.adapter_running := state =/= State.IDLE
  io.replay_done := state === State.RUN
  io.output_overflow := top.io.output_overflow
  io.pair_max_ready_count := top.io.pair_max_ready_count
  io.max_busy_fpus := top.io.max_busy_fpus

  private def packBaseScales(scales: Vec[Bits]): Unit = {
    for(i <- 0 until blockCount) {
      internalPayload(8 * (i + 1) - 1 downto 8 * i) := scales(i)
    }
  }

  private def packExtScales(scales: Vec[Bits]): Unit = {
    for(i <- 0 until blockCount) {
      val dstStart = 8 * extraScalePerRefinedBlock * i
      val srcStart = extraScalePerRefinedBlock * i
      for(j <- 0 until extraScalePerRefinedBlock) {
        internalPayload(dstStart + 8 * (j + 1) - 1 downto dstStart + 8 * j) := scales(srcStart + j)
      }
    }
  }

  switch(state) {
    is(State.IDLE) {
      replayCount := 0
      replayBlock := 0
      replayRepeat := 0
      when(io.MSA_EN) {
        resetStoredState()
        state := State.COLLECT
      }
    }

    is(State.COLLECT) {
      when(io.data_valid) {
        switch(pktType) {
          is(OLD_HEAD) {
            headPayload := pktPayload
            weightMask := pktPayload(943 downto 688).resize(blockCount)
            inputMask := pktPayload(687 downto 432).resize(blockCount)
          }
          is(OLD_WEIGHT_DATA) {
            when(pktBlockIdx < U(blockCount, 8 bits)) {
              weightData(pktBlockIdx.resized) := pktPayload(weightBitsPerLane - 1 downto 0)
            }
          }
          is(OLD_INPUT_DATA) {
            inputData := pktPayload(inputBits - 1 downto 0)
            replayCount := 0
            replayBlock := 0
            replayRepeat := 0
            state := State.START_INTERNAL
          }
          is(OLD_WEIGHT_SCALE) {
            for(i <- 0 until 126) {
              val idx = pktBlockIdx.resize(11) + U(i, 11 bits)
              when(idx < U(blockCount, 11 bits)) {
                weightScaleBase(idx.resized) := pktPayload(8 * (i + 1) - 1 downto 8 * i)
              }
            }
          }
          is(OLD_INPUT_SCALE) {
            for(i <- 0 until 126) {
              val idx = pktBlockIdx.resize(11) + U(i, 11 bits)
              when(idx < U(blockCount, 11 bits)) {
                inputScaleBase(idx.resized) := pktPayload(8 * (i + 1) - 1 downto 8 * i)
              }
            }
          }
          is(OLD_WEIGHT_SCALE_EXT) {
            for(i <- 0 until 126) {
              val idx = pktBlockIdx.resize(11) + U(i, 11 bits)
              when(idx < U(extEntryCount, 11 bits)) {
                weightScaleExt(idx.resized) := pktPayload(8 * (i + 1) - 1 downto 8 * i)
              }
            }
          }
          is(OLD_INPUT_SCALE_EXT) {
            for(i <- 0 until 126) {
              val idx = pktBlockIdx.resize(11) + U(i, 11 bits)
              when(idx < U(extEntryCount, 11 bits)) {
                inputScaleExt(idx.resized) := pktPayload(8 * (i + 1) - 1 downto 8 * i)
              }
            }
          }
        }
      }
    }

    is(State.START_INTERNAL) {
      replayCount := 0
      state := State.SEND_HEAD
    }

    is(State.SEND_HEAD) {
      internalPacketValid := True
      internalPacketType := NEW_HEAD
      internalPacketBlockIdx := 0
      internalPacketTargetMask := allMask
      internalPayload := headPayload
      replayCount := 0
      state := State.SEND_WEIGHT_SCALE
    }

    is(State.SEND_WEIGHT_SCALE) {
      internalPacketValid := True
      internalPacketType := NEW_WEIGHT_SCALE
      internalPacketBlockIdx := 0
      internalPacketTargetMask := allMask
      packBaseScales(weightScaleBase)
      when(replayCount === U(baseScalePackets - 1, replayCountWidth bits)) {
        replayCount := 0
        state := State.SEND_INPUT_SCALE
      } otherwise {
        replayCount := replayCount + 1
      }
    }

    is(State.SEND_INPUT_SCALE) {
      internalPacketValid := True
      internalPacketType := NEW_INPUT_SCALE
      internalPacketBlockIdx := 0
      internalPacketTargetMask := allMask
      packBaseScales(inputScaleBase)
      when(replayCount === U(baseScalePackets - 1, replayCountWidth bits)) {
        replayCount := 0
        when(refinedMask.orR) {
          state := State.SEND_WEIGHT_EXT
        } otherwise {
          replayBlock := 0
          replayRepeat := 0
          state := State.SEND_WEIGHT_DATA
        }
      } otherwise {
        replayCount := replayCount + 1
      }
    }

    is(State.SEND_WEIGHT_EXT) {
      internalPacketValid := True
      internalPacketType := NEW_WEIGHT_SCALE_EXT
      internalPacketBlockIdx := 0
      internalPacketTargetMask := refinedMask
      packExtScales(weightScaleExt)
      when(replayCount === U(extScalePackets - 1, replayCountWidth bits)) {
        replayCount := 0
        state := State.SEND_INPUT_EXT
      } otherwise {
        replayCount := replayCount + 1
      }
    }

    is(State.SEND_INPUT_EXT) {
      internalPacketValid := True
      internalPacketType := NEW_INPUT_SCALE_EXT
      internalPacketBlockIdx := 0
      internalPacketTargetMask := refinedMask
      packExtScales(inputScaleExt)
      when(replayCount === U(extScalePackets - 1, replayCountWidth bits)) {
        replayCount := 0
        replayBlock := 0
        replayRepeat := 0
        state := State.SEND_WEIGHT_DATA
      } otherwise {
        replayCount := replayCount + 1
      }
    }

    is(State.SEND_WEIGHT_DATA) {
      internalPacketValid := True
      internalPacketType := NEW_WEIGHT_DATA
      internalPacketBlockIdx := replayBlock
      internalPacketTargetMask := 0
      internalPayload(weightBitsPerLane - 1 downto 0) := weightData(replayBlock)
      when(replayRepeat === U(laneCount - 1, replayCountWidth bits)) {
        replayRepeat := 0
        when(replayBlock === U(blockCount - 1, blockIdxWidth bits)) {
          replayBlock := 0
          state := State.SEND_INPUT_DATA
        } otherwise {
          replayBlock := replayBlock + 1
        }
      } otherwise {
        replayRepeat := replayRepeat + 1
      }
    }

    is(State.SEND_INPUT_DATA) {
      internalPacketValid := True
      internalPacketType := NEW_INPUT_DATA
      internalPacketBlockIdx := 0
      internalPacketTargetMask := allMask
      internalPayload(inputBits - 1 downto 0) := inputData
      state := State.RUN
    }

    is(State.RUN) {
      when(top.io.done) {
        state := State.IDLE
      }
    }
  }
}

object MBPriorQExternal1024PacketTopGen {
  def main(args: Array[String]): Unit = {
    SpinalConfig(
      targetDirectory = "rtl_external_1024_packet_top",
      bitVectorWidthMax = 131072
    ).generateVerilog(new MBPriorQExternal1024PacketTop())
  }
}
