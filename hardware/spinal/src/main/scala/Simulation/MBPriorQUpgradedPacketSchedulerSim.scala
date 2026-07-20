package Simulation

import MBPriorQ.MBPriorQUpgradedPacketScheduler
import spinal.core._
import spinal.core.sim._

import java.io.{File, PrintWriter}
import scala.collection.mutable

object MBPriorQUpgradedPacketSchedulerSim {
  private case class MaskIssue(cycle: Int, mask: Int)
  private case class BlockIssue(cycle: Int, block: Int, refined: Boolean)
  private case class OutputEvent(cycle: Int, block: Int, packet: Int)

  private val blockCount = 4
  private val msaCount = 4
  private val refinedMask = 0xa
  private val allMask = (1 << blockCount) - 1

  private def simWorkspace(name: String): String = {
    val base = sys.env.getOrElse("MBPRIORQ_SIM_WORKSPACE", "./simWorkspace")
    s"$base/$name"
  }

  def main(args: Array[String]): Unit = {
    val vpuIssues = mutable.ArrayBuffer[MaskIssue]()
    val msaIssues = mutable.ArrayBuffer[BlockIssue]()
    val outputs = mutable.ArrayBuffer[OutputEvent]()
    var finalDone = false

    SimConfig
      .withConfig(SpinalConfig(bitVectorWidthMax = 4096))
      .workspacePath(simWorkspace("MBPriorQUpgradedPacketSchedulerSim"))
      .addSimulatorFlag("-CFLAGS").addSimulatorFlag("-std=c++14")
      .addSimulatorFlag("-LDFLAGS").addSimulatorFlag("-std=c++14")
      .compile(new MBPriorQUpgradedPacketScheduler(
        blockCount = blockCount,
        msaCount = msaCount,
        outputBufferCapacity = blockCount,
        vpuIssueMaskEnable = true
      ))
      .doSim { dut =>
        dut.clockDomain.forkStimulus(period = 10)
        var cycle = 0

        def driveDefaults(): Unit = {
          dut.io.start #= false
          dut.io.clear #= false
          dut.io.refined_mask #= refinedMask
          dut.io.packet_valid #= false
          dut.io.packet_type #= 0
          dut.io.packet_block_idx #= 0
          dut.io.packet_target_mask #= 0
          dut.io.msa_issue_ready #= true
          dut.io.vpu_issue_ready #= true
          dut.io.vpu_issue_mask_ready #= true
          dut.io.msa_done_valid #= false
          dut.io.msa_done_block_idx #= 0
          dut.io.dequant_done_valid #= false
          dut.io.dequant_done_block_idx #= 0
          dut.io.output_ready #= true
        }

        def sample(): Unit = {
          dut.clockDomain.waitSampling(1)
          cycle += 1
          if(dut.io.vpu_issue_mask_valid.toBoolean) {
            vpuIssues += MaskIssue(cycle, dut.io.vpu_issue_mask.toInt)
          }
          if(dut.io.msa_issue_valid.toBoolean) {
            msaIssues += BlockIssue(
              cycle,
              dut.io.msa_issue_block_idx.toInt,
              dut.io.msa_issue_refined.toBoolean
            )
          }
          if(dut.io.output_pulse.toBoolean) {
            outputs += OutputEvent(cycle, dut.io.output_block_idx.toInt, dut.io.output_packet_idx.toInt)
          }
          finalDone = dut.io.done.toBoolean
        }

        def idle(): Unit = {
          driveDefaults()
          sample()
        }

        def packet(kind: Int, targetMask: Int): Unit = {
          driveDefaults()
          dut.io.packet_valid #= true
          dut.io.packet_type #= kind
          dut.io.packet_target_mask #= targetMask
          sample()
        }

        def complete(block: Int): Unit = {
          driveDefaults()
          dut.io.msa_done_valid #= true
          dut.io.msa_done_block_idx #= block
          dut.io.dequant_done_valid #= true
          dut.io.dequant_done_block_idx #= block
          sample()
        }

        driveDefaults()
        dut.clockDomain.waitSampling(3)
        dut.io.clear #= true
        sample()
        driveDefaults()
        dut.io.start #= true
        sample()

        // Metadata arrives before matrix data. Regular blocks become VPU-ready
        // after base scales; refined blocks wait for the three extra scales.
        packet(kind = 0, targetMask = allMask)
        packet(kind = 1, targetMask = allMask)
        packet(kind = 2, targetMask = allMask)
        packet(kind = 3, targetMask = refinedMask)
        packet(kind = 4, targetMask = refinedMask)
        idle()

        assert(msaIssues.isEmpty, "MultiMSA issued before weight/activation data was ready")
        assert(vpuIssues.map(_.mask) == Seq(allMask ^ refinedMask, refinedMask),
          s"unexpected VPU issue masks: ${vpuIssues.map(issue => f"0x${issue.mask}%x").mkString(",")}")

        for(_ <- 0 until msaCount) packet(kind = 5, targetMask = allMask)
        packet(kind = 6, targetMask = allMask)
        while(msaIssues.map(_.block).distinct.size < blockCount && cycle < 64) idle()

        assert(msaIssues.map(_.block) == (0 until blockCount),
          s"unexpected MultiMSA issue order: ${msaIssues.map(_.block).mkString(",")}")
        assert(vpuIssues.map(_.cycle).max < msaIssues.map(_.cycle).min,
          s"VPU did not issue before MultiMSA: VPU=$vpuIssues MultiMSA=$msaIssues")
        msaIssues.foreach { issue =>
          assert(issue.refined == (((refinedMask >> issue.block) & 1) == 1),
            s"block ${issue.block} path classification mismatch")
        }

        // Completion is intentionally out of order. Commit must still follow
        // block_idx and use 4/16 logical output beats for regular/refined blocks.
        Seq(3, 1, 2, 0).foreach(complete)
        while(!finalDone && cycle < 160) idle()
        assert(finalDone, "packet scheduler did not commit all blocks")

        val transitionOrder = outputs.foldLeft(Vector.empty[Int]) { (order, event) =>
          if(order.lastOption.contains(event.block)) order else order :+ event.block
        }
        assert(transitionOrder == (0 until blockCount),
          s"output block order mismatch: ${transitionOrder.mkString(",")}")
        val counts = outputs.groupBy(_.block).map { case (block, events) => block -> events.size }
        (0 until blockCount).foreach { block =>
          val expected = if(((refinedMask >> block) & 1) == 1) 16 else 4
          assert(counts.getOrElse(block, 0) == expected,
            s"block $block emitted ${counts.getOrElse(block, 0)} beats, expected $expected")
        }
      }

    val resultPath = sys.env.getOrElse(
      "MBPRIORQ_PACKET_SCHEDULER_CSV",
      "target/ae-results/modules/packet_scheduler.csv"
    )
    val file = new File(resultPath)
    Option(file.getParentFile).foreach(_.mkdirs())
    val out = new PrintWriter(file)
    try {
      val outputOrder = outputs.foldLeft(Vector.empty[Int]) { (order, event) =>
        if(order.lastOption.contains(event.block)) order else order :+ event.block
      }
      val outputCounts = (0 until blockCount).map(block => outputs.count(_.block == block))
      val completionOrder = Seq(3, 1, 2, 0)
      out.println("block_index,path,vpu_issue_cycle,multimsa_issue_cycle,vpu_precedes_multimsa,injected_completion_rank,committed_output_rank,expected_output_beats,actual_output_beats,status")
      (0 until blockCount).foreach { block =>
        val issue = msaIssues.find(_.block == block).get
        val vpuCycle = vpuIssues.find(maskIssue => ((maskIssue.mask >> block) & 1) == 1).get.cycle
        val expectedBeats = if(issue.refined) 16 else 4
        out.println(Seq(
          block,
          if(issue.refined) "refined" else "regular",
          vpuCycle,
          issue.cycle,
          vpuCycle < issue.cycle,
          completionOrder.indexOf(block) + 1,
          outputOrder.indexOf(block) + 1,
          expectedBeats,
          outputCounts(block),
          "PASS"
        ).mkString(","))
      }
    } finally {
      out.close()
    }

    println(s"MBPriorQ packet scheduler simulation passed. Results: $resultPath")
  }
}
