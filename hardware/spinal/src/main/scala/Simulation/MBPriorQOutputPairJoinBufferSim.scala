package Simulation

import MBPriorQ.MBPriorQOutputPairJoinBuffer
import spinal.core._
import spinal.core.sim._

import java.io.{File, PrintWriter}
import scala.collection.mutable

object MBPriorQOutputPairJoinBufferSim {
  private case class MsaEvent(cycle: Int, block: Int, sub: Int)
  private case class FactorEvent(cycle: Int, block: Int, mask: Int)
  private case class PairOut(cycle: Int, block: Int, sub: Int)

  private val blockCount = 3
  private val matrixBits = 16 * 16 * 16

  private def simWorkspace(name: String): String = {
    val base = sys.env.getOrElse("MBPRIORQ_SIM_WORKSPACE", "./simWorkspace")
    s"$base/$name"
  }

  private def matrixPayload(block: Int, sub: Int): BigInt =
    (BigInt(block & 0xff) << 4088) | (BigInt(sub & 0x3) << 4080) | BigInt("123456789abcdef", 16)

  private def scalePayload(block: Int, sub: Int): BigInt =
    BigInt(((block + 1) * 0x10000000L + sub).toLong & 0xffffffffL)

  private def factorPayload(block: Int): BigInt =
    (0 until 4).foldLeft(BigInt(0)) { case (acc, sub) =>
      acc | (scalePayload(block, sub) << (32 * sub))
    }

  private def outputReady(cycle: Int): Boolean =
    cycle % 5 != 3

  def main(args: Array[String]): Unit = {
    val resultPath = sys.env.getOrElse(
      "MBPRIORQ_OUTPUT_PAIR_JOIN_CSV",
      "target/ae-results/output_pair_join.csv"
    )

    val msaEvents = Seq(
      MsaEvent(1, 1, 2),
      MsaEvent(2, 1, 0),
      MsaEvent(3, 1, 3),
      MsaEvent(4, 1, 1),
      MsaEvent(5, 0, 0),
      MsaEvent(8, 2, 0)
    )
    val factorEvents = Seq(
      FactorEvent(0, 1, 0xf),
      FactorEvent(6, 0, 0x1),
      FactorEvent(7, 2, 0x1)
    )
    val outputs = mutable.ArrayBuffer[PairOut]()

    SimConfig
      .withConfig(SpinalConfig(bitVectorWidthMax = 131072))
      .workspacePath(simWorkspace("MBPriorQOutputPairJoinBufferSim"))
      .addSimulatorFlag("-CFLAGS").addSimulatorFlag("-std=c++14")
      .addSimulatorFlag("-LDFLAGS").addSimulatorFlag("-std=c++14")
      .compile(new MBPriorQOutputPairJoinBuffer(blockCount, matrixBits))
      .doSim { dut =>
        dut.clockDomain.forkStimulus(period = 10)

        dut.io.clear #= false
        dut.io.msa_valid #= false
        dut.io.msa_block_idx #= 0
        dut.io.msa_sub_block_idx #= 0
        dut.io.msa_matrix #= 0
        dut.io.factor_valid #= false
        dut.io.factor_block_idx #= 0
        dut.io.factor_valid_mask #= 0
        dut.io.factor_data #= 0
        dut.io.output_ready #= false

        dut.clockDomain.waitSampling(3)
        dut.io.clear #= true
        dut.clockDomain.waitSampling(1)
        dut.io.clear #= false

        var cycle = 0
        val maxCycles = 96
        while (!dut.io.done.toBoolean && cycle < maxCycles) {
          msaEvents.find(_.cycle == cycle) match {
            case Some(event) =>
              dut.io.msa_valid #= true
              dut.io.msa_block_idx #= event.block
              dut.io.msa_sub_block_idx #= event.sub
              dut.io.msa_matrix #= matrixPayload(event.block, event.sub)
            case None =>
              dut.io.msa_valid #= false
              dut.io.msa_block_idx #= 0
              dut.io.msa_sub_block_idx #= 0
              dut.io.msa_matrix #= 0
          }

          factorEvents.find(_.cycle == cycle) match {
            case Some(event) =>
              dut.io.factor_valid #= true
              dut.io.factor_block_idx #= event.block
              dut.io.factor_valid_mask #= event.mask
              dut.io.factor_data #= factorPayload(event.block)
            case None =>
              dut.io.factor_valid #= false
              dut.io.factor_block_idx #= 0
              dut.io.factor_valid_mask #= 0
              dut.io.factor_data #= 0
          }

          val ready = outputReady(cycle)
          dut.io.output_ready #= ready
          dut.clockDomain.waitSampling(1)

          if (dut.io.output_valid.toBoolean && ready) {
            val block = dut.io.output_block_idx.toInt
            val sub = dut.io.output_sub_block_idx.toInt
            assert(dut.io.output_matrix.toBigInt == matrixPayload(block, sub), s"matrix mismatch for block=$block sub=$sub")
            assert(dut.io.output_dequant_scale.toBigInt == scalePayload(block, sub), s"scale mismatch for block=$block sub=$sub")
            outputs += PairOut(cycle, block, sub)
          }

          cycle += 1
        }

        assert(dut.io.done.toBoolean, s"output pair join did not finish by $maxCycles cycles")
      }

    val expected = Seq(
      PairOut(0, 0, 0),
      PairOut(0, 1, 0),
      PairOut(0, 1, 1),
      PairOut(0, 1, 2),
      PairOut(0, 1, 3),
      PairOut(0, 2, 0)
    )
    assert(outputs.map(o => (o.block, o.sub)) == expected.map(o => (o.block, o.sub)),
      s"unexpected output order: ${outputs.map(o => s"${o.block}:${o.sub}").mkString(",")}")

    val file = new File(resultPath)
    Option(file.getParentFile).foreach(_.mkdirs())
    val out = new PrintWriter(file)
    try {
      out.println("block_index,sub_block_idx,matrix_arrival_cycle,factor_arrival_cycle,both_inputs_ready_cycle,output_cycle,expected_output_rank,actual_output_rank,matrix_match,scale_match,status")
      outputs.zipWithIndex.foreach { case (item, rank) =>
        val matrixCycle = msaEvents.find(event => event.block == item.block && event.sub == item.sub).get.cycle
        val factorCycle = factorEvents.find(_.block == item.block).get.cycle
        val expectedRank = expected.indexWhere(event => event.block == item.block && event.sub == item.sub) + 1
        out.println(Seq(
          item.block,
          item.sub,
          matrixCycle,
          factorCycle,
          math.max(matrixCycle, factorCycle),
          item.cycle,
          expectedRank,
          rank + 1,
          true,
          true,
          "PASS"
        ).mkString(","))
      }
    } finally {
      out.close()
    }

    println(s"MBPriorQ output pair join buffer simulation passed. Results: $resultPath")
  }
}
