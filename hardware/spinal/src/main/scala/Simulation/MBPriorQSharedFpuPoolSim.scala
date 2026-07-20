package Simulation

import MBPriorQ.MBPriorQSharedFpuPool
import spinal.core._
import spinal.core.sim._

import java.io.{File, PrintWriter}
import scala.collection.mutable

object MBPriorQSharedFpuPoolSim {
  private case class DoneEvent(cycle: Int, block: Int, refined: Boolean)

  private val blockCount = 8
  private val fpuCount = 16
  private val fpuLatency = 3
  private val refinedMask = BigInt("ac", 16)

  private def simWorkspace(name: String): String = {
    val base = sys.env.getOrElse("MBPRIORQ_SIM_WORKSPACE", "./simWorkspace")
    s"$base/$name"
  }

  def main(args: Array[String]): Unit = {
    val events = mutable.ArrayBuffer[DoneEvent]()
    var observedMaxBusy = 0
    var acceptedAtEnd = 0
    var completedAtEnd = 0

    SimConfig
      .withConfig(SpinalConfig(bitVectorWidthMax = 4096))
      .workspacePath(simWorkspace("MBPriorQSharedFpuPoolSim"))
      .addSimulatorFlag("-CFLAGS").addSimulatorFlag("-std=c++14")
      .addSimulatorFlag("-LDFLAGS").addSimulatorFlag("-std=c++14")
      .compile(new MBPriorQSharedFpuPool(blockCount, fpuCount, fpuLatency))
      .doSim { dut =>
        dut.clockDomain.forkStimulus(period = 10)
        dut.io.clear #= false
        dut.io.issue_valid #= false
        dut.io.issue_block_idx #= 0
        dut.io.issue_refined #= false
        dut.io.issue_mask_valid #= false
        dut.io.issue_mask #= 0
        dut.io.issue_refined_mask #= refinedMask

        dut.clockDomain.waitSampling(3)
        dut.io.clear #= true
        dut.clockDomain.waitSampling(1)
        dut.io.clear #= false

        assert(dut.io.issue_mask_ready.toBoolean, "shared FPU pool did not accept a mask issue")
        dut.io.issue_mask_valid #= true
        dut.io.issue_mask #= (BigInt(1) << blockCount) - 1
        dut.clockDomain.waitSampling(1)
        dut.io.issue_mask_valid #= false
        dut.io.issue_mask #= 0

        var cycle = 0
        val maxCycles = 256
        while (events.map(_.block).distinct.size < blockCount && cycle < maxCycles) {
          dut.clockDomain.waitSampling(1)
          cycle += 1
          observedMaxBusy = math.max(observedMaxBusy, dut.io.max_busy_fpus.toInt)
          if (dut.io.done_valid.toBoolean) {
            val block = dut.io.done_block_idx.toInt
            events += DoneEvent(cycle, block, ((refinedMask >> block) & 1) == 1)
          }
        }

        acceptedAtEnd = dut.io.accepted_blocks.toInt
        completedAtEnd = dut.io.completed_blocks.toInt
        assert(events.map(_.block).distinct.sorted == (0 until blockCount),
          s"shared FPU pool completion set mismatch: ${events.map(_.block).mkString(",")}")
        assert(events.size == blockCount, s"duplicate completion event: $events")
        assert(acceptedAtEnd == blockCount, s"accepted $acceptedAtEnd/$blockCount blocks")
        assert(completedAtEnd == blockCount, s"completed $completedAtEnd/$blockCount blocks")
        assert(observedMaxBusy > 0 && observedMaxBusy <= fpuCount,
          s"invalid maximum FPU occupancy: $observedMaxBusy")
      }

    val resultPath = sys.env.getOrElse(
      "MBPRIORQ_FPU_POOL_CSV",
      "target/ae-results/modules/shared_fpu_pool.csv"
    )
    val file = new File(resultPath)
    Option(file.getParentFile).foreach(_.mkdirs())
    val out = new PrintWriter(file)
    try {
      out.println("block_index,path,completion_cycle,pool_size,max_busy_fpus,total_accepted_blocks,total_completed_blocks,status")
      events.foreach { event =>
        out.println(Seq(
          event.block,
          if(event.refined) "refined" else "regular",
          event.cycle,
          fpuCount,
          observedMaxBusy,
          acceptedAtEnd,
          completedAtEnd,
          "PASS"
        ).mkString(","))
      }
    } finally {
      out.close()
    }

    println(s"MBPriorQ shared FPU-pool simulation passed. Results: $resultPath")
  }
}
