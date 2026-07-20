package Simulation

import MBPriorQ.MBPriorQRefinedScheduledMultiMSA
import spinal.core._
import spinal.core.sim._

import java.io.{File, PrintWriter}
import scala.collection.mutable

object MBPriorQRefinedScheduledMultiMSASim {
  private case class CaseResult(
    name: String,
    refined: Boolean,
    firstPartialCycle: Int,
    doneCycle: Int,
    subBlocks: Seq[Int]
  )

  private val blockCount = 2
  private val physicalLanes = 4

  private def simWorkspace(name: String): String = {
    val base = sys.env.getOrElse("MBPRIORQ_SIM_WORKSPACE", "./simWorkspace")
    s"$base/$name"
  }

  private def fp4Vector(offset: Int): BigInt =
    (0 until 16).foldLeft(BigInt(0)) { case (acc, idx) =>
      val value = ((idx + offset) % 15) + 1
      acc | (BigInt(value & 0xf) << (4 * idx))
    }

  private val packedWeights = fp4Vector(0) | (fp4Vector(3) << 64)
  private val packedActivation = fp4Vector(1)

  private def runCase(
    compiled: SimCompiled[MBPriorQRefinedScheduledMultiMSA],
    name: String,
    block: Int,
    refined: Boolean
  ): CaseResult = {
    val subBlocks = mutable.ArrayBuffer[Int]()
    var firstPartial = -1
    var done = -1

    compiled.doSim { dut =>
      dut.clockDomain.forkStimulus(period = 10)
      dut.io.clear #= false
      dut.io.issue_valid #= false
      dut.io.issue_block_idx #= 0
      dut.io.issue_refined #= false
      dut.io.weight_data #= packedWeights
      dut.io.input_data #= packedActivation
      dut.io.partial_ready #= true

      dut.clockDomain.waitSampling(3)
      dut.io.clear #= true
      dut.clockDomain.waitSampling(1)
      dut.io.clear #= false
      assert(dut.io.issue_ready.toBoolean, s"$name was not ready at issue")

      dut.io.issue_valid #= true
      dut.io.issue_block_idx #= block
      dut.io.issue_refined #= refined
      dut.clockDomain.waitSampling(1)
      dut.io.issue_valid #= false

      var cycle = 0
      val maxCycles = 96
      while (done < 0 && cycle < maxCycles) {
        dut.clockDomain.waitSampling(1)
        cycle += 1
        if (dut.io.partial_valid.toBoolean) {
          if(firstPartial < 0) firstPartial = cycle
          assert(dut.io.partial_block_idx.toInt == block,
            s"$name returned block ${dut.io.partial_block_idx.toInt}, expected $block")
          assert(dut.io.partial_data.toBigInt != 0, s"$name returned an all-zero partial matrix")
          subBlocks += dut.io.partial_sub_block_idx.toInt
        }
        if (dut.io.block_done_valid.toBoolean) {
          assert(dut.io.block_done_block_idx.toInt == block,
            s"$name completed block ${dut.io.block_done_block_idx.toInt}, expected $block")
          done = cycle
        }
      }
    }

    assert(firstPartial > 0, s"$name produced no partial matrix")
    assert(done > 0, s"$name did not complete")
    val expectedSubs = if(refined) Seq(0, 1, 2, 3) else Seq(0)
    assert(subBlocks == expectedSubs, s"$name sub-blocks $subBlocks != $expectedSubs")
    CaseResult(name, refined, firstPartial, done, subBlocks.toSeq)
  }

  def main(args: Array[String]): Unit = {
    val compiled = SimConfig
      .withConfig(SpinalConfig(bitVectorWidthMax = 131072))
      .workspacePath(simWorkspace("MBPriorQRefinedScheduledMultiMSASim"))
      .addSimulatorFlag("-CFLAGS").addSimulatorFlag("-std=c++14")
      .addSimulatorFlag("-LDFLAGS").addSimulatorFlag("-std=c++14")
      .compile(new MBPriorQRefinedScheduledMultiMSA(16, 16, blockCount, physicalLanes))

    val regular = runCase(compiled, "regular", block = 0, refined = false)
    val refined = runCase(compiled, "refined", block = 1, refined = true)
    assert(refined.firstPartialCycle < regular.firstPartialCycle,
      s"refined first partial ${refined.firstPartialCycle} should precede regular ${regular.firstPartialCycle}")

    val resultPath = sys.env.getOrElse(
      "MBPRIORQ_MULTIMSA_CSV",
      "target/ae-results/modules/multimsa_paths.csv"
    )
    val file = new File(resultPath)
    Option(file.getParentFile).foreach(_.mkdirs())
    val out = new PrintWriter(file)
    try {
      out.println("case,path,fp4_values_per_micro_block,physical_msa_lanes,expected_sub_block_indices,actual_sub_block_indices,expected_partial_count,actual_partial_count,first_partial_cycle,done_cycle,status")
      Seq(regular, refined).foreach { result =>
        val expectedSubs = if(result.refined) Seq(0, 1, 2, 3) else Seq(0)
        out.println(Seq(
          result.name,
          if(result.refined) "refined" else "regular",
          16,
          physicalLanes,
          "\"" + expectedSubs.mkString("[", ",", "]") + "\"",
          "\"" + result.subBlocks.mkString("[", ",", "]") + "\"",
          expectedSubs.size,
          result.subBlocks.size,
          result.firstPartialCycle,
          result.doneCycle,
          "PASS"
        ).mkString(","))
      }
    } finally {
      out.close()
    }

    println(s"MBPriorQ MultiMSA path simulation passed. Results: $resultPath")
  }
}
