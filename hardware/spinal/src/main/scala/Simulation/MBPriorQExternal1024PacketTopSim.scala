package Simulation

import MBPriorQ.{MBPriorQ, MBPriorQClearableMSA}
import spinal.core._
import spinal.core.sim._

import java.io.{File, PrintWriter}
import scala.collection.mutable

object MBPriorQExternal1024PacketTopSim {
  private case class OldPacket(kind: Int, blockIdx: Int, payload: BigInt)
  private case class ScaleSet(base: Int, ext: Seq[Int])
  private case class PhysicalPacket(cycle: Int, packetType: Int, block: Int, sub: Int, segment: Int, last: Boolean, payload: BigInt)
  private case class PairKey(block: Int, sub: Int)

  private val blockCount = 16
  private val weightNum = 16
  private val inputNum = 16
  private val matrixBits = 16 * 16 * 16
  private val payloadBits = 1000
  private val payloadMask = (BigInt(1) << payloadBits) - 1
  private val elementsPerSubBlock = 4

  private def fp8Power(exp: Int): Int = exp << 3
  private val fp8Half = fp8Power(6)
  private val fp8One = fp8Power(7)
  private val fp8Two = fp8Power(8)
  private val fp8Four = fp8Power(9)

  private def fp32(value: Float): BigInt =
    BigInt(java.lang.Float.floatToIntBits(value).toLong & 0xffffffffL)

  private def fp8Value(bits: Int): Float = bits match {
    case `fp8Half` => 0.5f
    case `fp8One` => 1.0f
    case `fp8Two` => 2.0f
    case `fp8Four` => 4.0f
    case other => throw new IllegalArgumentException(s"Unexpected FP8 test value: $other")
  }

  private def hex32(value: BigInt): String =
    f"${value.toLong & 0xffffffffL}%08x"

  private def simWorkspace(name: String): String = {
    val base = sys.env.getOrElse("MBPRIORQ_SIM_WORKSPACE", "./simWorkspace")
    s"$base/$name"
  }

  private def headPayload(weightMask: BigInt, inputMask: BigInt): BigInt =
    (fp32(1.0f) << 976) | (fp32(1.0f) << 944) | (weightMask << 688) | (inputMask << 432)

  private def packOldPacket(kind: Int, blockIdx: Int, payload: BigInt): BigInt =
    (BigInt(kind & 0xff) << 1016) | (BigInt(blockIdx & 0xff) << 1008) | (payload & ((BigInt(1) << 1008) - 1))

  private def packAddressedScaleEntries(values: Seq[Int], startIdx: Int): BigInt =
    (0 until 126).foldLeft(BigInt(0)) { case (acc, byteIdx) =>
      val entryIdx = startIdx + byteIdx
      if (entryIdx < values.length) acc | (BigInt(values(entryIdx) & 0xff) << (8 * byteIdx))
      else acc
    }

  private def flattenExt(scales: Seq[ScaleSet]): Seq[Int] =
    scales.flatMap(_.ext)

  private def weightPayload(block: Int): BigInt =
    (0 until weightNum).foldLeft(BigInt(0)) { case (acc, idx) =>
      val value = ((block + idx) % 15) + 1
      acc | (BigInt(value & 0xf) << (4 * idx))
    }

  private def inputPayload(): BigInt =
    (0 until inputNum).foldLeft(BigInt(0)) { case (acc, idx) =>
      val value = (idx % 15) + 1
      acc | (BigInt(value & 0xf) << (4 * idx))
    }

  private def oldPacketStream(
    weightMask: BigInt,
    inputMask: BigInt,
    weightScales: Seq[ScaleSet],
    inputScales: Seq[ScaleSet]
  ): Vector[OldPacket] = {
    val packets = Vector.newBuilder[OldPacket]
    val weightBase = weightScales.map(_.base)
    val inputBase = inputScales.map(_.base)
    val weightExt = flattenExt(weightScales)
    val inputExt = flattenExt(inputScales)

    packets += OldPacket(0x00, 0, headPayload(weightMask, inputMask))

    // Deliberately split and reorder base scale packets to verify public scale-entry addressing.
    packets += OldPacket(0x03, 8, packAddressedScaleEntries(weightBase, 8))
    packets += OldPacket(0x03, 0, packAddressedScaleEntries(weightBase, 0))
    packets += OldPacket(0x05, 0, packAddressedScaleEntries(inputBase, 0))
    packets += OldPacket(0x05, 8, packAddressedScaleEntries(inputBase, 8))

    // Extended scale entries use the old flat entry address: block * 3 + extra_idx.
    packets += OldPacket(0x04, 24, packAddressedScaleEntries(weightExt, 24))
    packets += OldPacket(0x04, 0, packAddressedScaleEntries(weightExt, 0))
    packets += OldPacket(0x06, 0, packAddressedScaleEntries(inputExt, 0))
    packets += OldPacket(0x06, 24, packAddressedScaleEntries(inputExt, 24))

    for(block <- 0 until blockCount) {
      packets += OldPacket(0x01, block, weightPayload(block))
    }
    packets += OldPacket(0x02, 0, inputPayload())
    packets.result()
  }

  private def maskSubBlock(data: BigInt, sub: Int, elementCount: Int): BigInt = {
    val start = sub * elementsPerSubBlock
    (0 until elementsPerSubBlock).foldLeft(BigInt(0)) { case (acc, offset) =>
      val idx = start + offset
      val value = (data >> (4 * idx)) & BigInt(0xf)
      acc | (value << (4 * idx))
    } & ((BigInt(1) << (4 * elementCount)) - 1)
  }

  private def expectedFactors(
    idx: Int,
    weightMask: BigInt,
    inputMask: BigInt,
    weightScales: Seq[ScaleSet],
    inputScales: Seq[ScaleSet]
  ): (Int, Seq[BigInt]) = {
    val weightVmb = ((weightMask >> idx) & 1) == 1
    val inputVmb = ((inputMask >> idx) & 1) == 1
    val validMask = if (weightVmb || inputVmb) 0xf else 0x1
    val weight = if (weightVmb) weightScales(idx).base +: weightScales(idx).ext else Seq.fill(4)(weightScales(idx).base)
    val input = if (inputVmb) inputScales(idx).base +: inputScales(idx).ext else Seq.fill(4)(inputScales(idx).base)
    val factors = weight.zip(input).map { case (w, i) => fp32(fp8Value(w) * fp8Value(i)) }
    (validMask, factors)
  }

  private def outputReady(cycle: Int): Boolean =
    cycle % 7 != 6

  private def decode(cycle: Int, value: BigInt): PhysicalPacket =
    PhysicalPacket(
      cycle = cycle,
      packetType = ((value >> 1016) & 0xff).toInt,
      block = ((value >> 1008) & 0xff).toInt,
      sub = ((value >> 1006) & 0x3).toInt,
      segment = ((value >> 1003) & 0x7).toInt,
      last = ((value >> 1002) & 0x1) == 1,
      payload = value & payloadMask
    )

  private def reconstruct(group: Seq[PhysicalPacket]): (BigInt, BigInt) = {
    require(group.map(_.segment) == Seq(0, 1, 2, 3, 4), s"bad segment order: ${group.map(_.segment).mkString(",")}")
    val low = group.take(4).zipWithIndex.foldLeft(BigInt(0)) { case (acc, (packet, idx)) =>
      acc | (packet.payload << (idx * payloadBits))
    }
    val tail = group(4).payload
    val high = tail & ((BigInt(1) << 96) - 1)
    val scale = (tail >> 96) & BigInt("ffffffff", 16)
    (low | (high << 4000), scale)
  }

  def main(args: Array[String]): Unit = {
    val weightMask = (0 until blockCount).filter(i => i % 2 == 0 || i == 15).foldLeft(BigInt(0))((acc, idx) => acc | (BigInt(1) << idx))
    val inputMask = (0 until blockCount).filter(i => i % 3 == 0 || i == 14).foldLeft(BigInt(0))((acc, idx) => acc | (BigInt(1) << idx))
    val refinedMask = weightMask | inputMask
    val weightScales = (0 until blockCount).map { idx =>
      val base = if (idx % 2 == 0) fp8One else fp8Two
      ScaleSet(base, Seq(fp8Two, fp8Four, fp8Half))
    }
    val inputScales = (0 until blockCount).map { idx =>
      val base = if (idx % 3 == 0) fp8Two else fp8One
      ScaleSet(base, Seq(fp8One, fp8Half, fp8Four))
    }
    val packets = oldPacketStream(weightMask, inputMask, weightScales, inputScales)
    val physicalPackets = mutable.ArrayBuffer[PhysicalPacket]()

    val resultPath = sys.env.getOrElse(
      "MBPRIORQ_EXTERNAL_1024_TOP_CSV",
      "target/ae-results/external_1024_top.csv"
    )
    val summaryPath = sys.env.getOrElse(
      "MBPRIORQ_EXTERNAL_1024_TOP_SUMMARY_CSV",
      "target/ae-results/system_summary.csv"
    )

    val compiledMsa = SimConfig
      .withConfig(SpinalConfig(bitVectorWidthMax = 131072))
      .workspacePath(simWorkspace("MBPriorQExternal1024PacketTopRefMSA"))
      .addSimulatorFlag("-CFLAGS").addSimulatorFlag("-std=c++14")
      .addSimulatorFlag("-LDFLAGS").addSimulatorFlag("-std=c++14")
      .compile(new MBPriorQClearableMSA(weightNum, inputNum))

    def runReferenceMsa(weight: BigInt, inputData: BigInt, refined: Boolean): BigInt = {
      var result = BigInt(0)
      compiledMsa.doSim { dut =>
        dut.clockDomain.forkStimulus(period = 10)
        dut.io.clear #= false
        dut.io.Valid #= false
        dut.io.NeedSubCycle #= refined
        dut.io.Weight_Forward_Data #= weight
        dut.io.Input_Forward_Data #= inputData
        dut.io.OutputReady #= false
        dut.clockDomain.waitSampling(3)
        dut.io.clear #= true
        dut.clockDomain.waitSampling(1)
        dut.io.clear #= false

        var cycle = 0
        val maxCycles = 96
        var found = false
        while (!found && cycle < maxCycles) {
          dut.io.Valid #= true
          dut.clockDomain.waitSampling(1)
          if (dut.io.OutputValid.toBoolean) {
            result = dut.io.OutputData.toBigInt
            found = true
          }
          cycle += 1
        }
        assert(found, s"reference MSA did not produce output within $maxCycles cycles")
      }
      result
    }

    val expectedMatrices = mutable.Map[PairKey, BigInt]()
    val expectedScales = mutable.Map[PairKey, BigInt]()
    for(block <- 0 until blockCount) {
      val refined = ((refinedMask >> block) & 1) == 1
      val (_, factors) = expectedFactors(block, weightMask, inputMask, weightScales, inputScales)
      if (refined) {
        for(sub <- 0 until 4) {
          expectedMatrices(PairKey(block, sub)) = runReferenceMsa(
            maskSubBlock(weightPayload(block), sub, weightNum),
            maskSubBlock(inputPayload(), sub, inputNum),
            refined = true
          )
          expectedScales(PairKey(block, sub)) = factors(sub)
        }
      } else {
        expectedMatrices(PairKey(block, 0)) = runReferenceMsa(weightPayload(block), inputPayload(), refined = false)
        expectedScales(PairKey(block, 0)) = factors.head
      }
    }

    SimConfig
      .withConfig(SpinalConfig(bitVectorWidthMax = 131072))
      .workspacePath(simWorkspace("MBPriorQExternal1024PacketTopSim"))
      .addSimulatorFlag("-CFLAGS").addSimulatorFlag("-std=c++14")
      .addSimulatorFlag("-LDFLAGS").addSimulatorFlag("-std=c++14")
      .compile(new MBPriorQ())
      .doSim { dut =>
        dut.clockDomain.forkStimulus(period = 10)

        dut.io.MSA_EN #= false
        dut.io.data_packet #= 0
        dut.io.data_valid #= false
        dut.io.output_ready #= false
        dut.clockDomain.waitSampling(3)

        dut.io.MSA_EN #= true
        dut.clockDomain.waitSampling(1)
        dut.io.MSA_EN #= false

        var packetPtr = 0
        var cycle = 0
        val expectedPacketCount = expectedMatrices.size * 5
        val maxCycles = packets.size + expectedPacketCount * 32 + blockCount * 512
        while (physicalPackets.size < expectedPacketCount && cycle < maxCycles) {
          if (packetPtr < packets.length) {
            val pkt = packets(packetPtr)
            dut.io.data_valid #= true
            dut.io.data_packet #= packOldPacket(pkt.kind, pkt.blockIdx, pkt.payload)
            packetPtr += 1
          } else {
            dut.io.data_valid #= false
            dut.io.data_packet #= 0
          }

          val ready = outputReady(cycle)
          dut.io.output_ready #= ready
          dut.clockDomain.waitSampling(1)

          if (dut.io.output_pulse.toBoolean) {
            physicalPackets += decode(cycle, dut.io.output_packet.toBigInt)
          }
          cycle += 1
        }

        assert(physicalPackets.size == expectedPacketCount, s"expected $expectedPacketCount physical packets, got ${physicalPackets.size}")
      }

    val groups = physicalPackets.grouped(5).toVector
    val expectedKeys = (0 until blockCount).flatMap { block =>
      if (((refinedMask >> block) & 1) == 1) (0 until 4).map(sub => PairKey(block, sub))
      else Seq(PairKey(block, 0))
    }
    assert(groups.size == expectedKeys.size, s"expected ${expectedKeys.size} matrix-scale groups, got ${groups.size}")
    assert(groups.map(g => PairKey(g.head.block, g.head.sub)) == expectedKeys,
      s"bad packet group order: ${groups.map(g => s"${g.head.block}:${g.head.sub}").mkString(",")}")

    groups.foreach { group =>
      val key = PairKey(group.head.block, group.head.sub)
      group.foreach { packet =>
        assert(packet.packetType == 0x80, s"packet type mismatch")
        assert(packet.block == key.block, s"block changed within packet group")
        assert(packet.sub == key.sub, s"sub-block changed within packet group")
      }
      assert(group.last.last, s"last flag missing for $key")
      assert(group.dropRight(1).forall(!_.last), s"early last flag for $key")
      val (matrix, scale) = reconstruct(group)
      assert(matrix == expectedMatrices(key), s"matrix mismatch for $key")
      assert(scale == expectedScales(key), s"scale mismatch for $key: got 0x${hex32(scale)} expected 0x${hex32(expectedScales(key))}")
    }

    val file = new File(resultPath)
    Option(file.getParentFile).foreach(_.mkdirs())
    val out = new PrintWriter(file)
    try {
      out.println("cycle,packet_type,block_index,sub_block_idx,segment_idx,last")
      physicalPackets.foreach { p =>
        out.println(Seq(p.cycle, f"0x${p.packetType}%02x", p.block, p.sub, p.segment, p.last).mkString(","))
      }
    } finally {
      out.close()
    }

    val summaryFile = new File(summaryPath)
    Option(summaryFile.getParentFile).foreach(_.mkdirs())
    val summaryOut = new PrintWriter(summaryFile)
    try {
      summaryOut.println("block_index,path,weight_mask,activation_mask,sub_block_indices,matrix_scale_pairs,matrix_matches,scale_matches,output_packets,first_output_cycle,last_output_cycle,status")
      for(block <- 0 until blockCount) {
        val blockGroups = groups.filter(_.head.block == block)
        val refined = ((refinedMask >> block) & 1) == 1
        val weightVmb = ((weightMask >> block) & 1) == 1
        val inputVmb = ((inputMask >> block) & 1) == 1
        summaryOut.println(Seq(
          block,
          if(refined) "refined" else "regular",
          if(weightVmb) 1 else 0,
          if(inputVmb) 1 else 0,
          "\"" + blockGroups.map(_.head.sub).mkString("[", ",", "]") + "\"",
          blockGroups.size,
          s"${blockGroups.size}/${blockGroups.size}",
          s"${blockGroups.size}/${blockGroups.size}",
          blockGroups.map(_.size).sum,
          blockGroups.head.head.cycle,
          blockGroups.last.last.cycle,
          "PASS"
        ).mkString(","))
      }
      summaryOut.println(Seq(
        "TOTAL",
        "all",
        "-",
        "-",
        "-",
        groups.size,
        s"${groups.size}/${groups.size}",
        s"${groups.size}/${groups.size}",
        physicalPackets.size,
        physicalPackets.head.cycle,
        physicalPackets.last.cycle,
        "PASS"
      ).mkString(","))
    } finally {
      summaryOut.close()
    }

    println(s"MBPriorQ public 1024-bit packet top simulation passed. Packet trace: $resultPath")
    println(s"Human-readable system summary: $summaryPath")
  }
}
