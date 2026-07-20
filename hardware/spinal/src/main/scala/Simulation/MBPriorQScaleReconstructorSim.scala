package Simulation

import MBPriorQ.MBPriorQScaleReconstructor
import spinal.core._
import spinal.core.sim._

import java.io.{File, PrintWriter}
import scala.collection.mutable

object MBPriorQScaleReconstructorSim {
  private case class TestCase(
    name: String,
    weightVmb: Boolean,
    activationVmb: Boolean,
    validMask: Int,
    expected: Seq[Float]
  )

  private case class Result(test: TestCase, validMask: Int, factors: Seq[BigInt])

  private def simWorkspace(name: String): String = {
    val base = sys.env.getOrElse("MBPRIORQ_SIM_WORKSPACE", "./simWorkspace")
    s"$base/$name"
  }

  private def fp8Power(exp: Int): Int = exp << 3
  private val fp8Half = fp8Power(6)
  private val fp8One = fp8Power(7)
  private val fp8Two = fp8Power(8)
  private val fp8Four = fp8Power(9)

  private def fp32(value: Float): BigInt =
    BigInt(java.lang.Float.floatToIntBits(value).toLong & 0xffffffffL)

  private def packFp8(values: Seq[Int]): BigInt =
    values.zipWithIndex.foldLeft(BigInt(0)) { case (acc, (value, idx)) =>
      acc | (BigInt(value & 0xff) << (8 * idx))
    }

  private def factors(value: BigInt): Seq[BigInt] =
    (0 until 4).map(idx => (value >> (32 * idx)) & BigInt("ffffffff", 16))

  private def hex32(value: BigInt): String =
    f"${value.toLong & 0xffffffffL}%08x"

  private def fp32Value(value: BigInt): Float =
    java.lang.Float.intBitsToFloat((value & BigInt("ffffffff", 16)).toInt)

  private def csvList(values: Seq[Any]): String =
    "\"" + values.mkString("[", ",", "]") + "\""

  def main(args: Array[String]): Unit = {
    val cases = Seq(
      TestCase("regular", weightVmb = false, activationVmb = false, 0x1, Seq(2.0f, 2.0f, 2.0f, 2.0f)),
      TestCase("weight_refined", weightVmb = true, activationVmb = false, 0xf, Seq(2.0f, 4.0f, 8.0f, 1.0f)),
      TestCase("activation_refined", weightVmb = false, activationVmb = true, 0xf, Seq(2.0f, 1.0f, 0.5f, 4.0f)),
      TestCase("both_refined", weightVmb = true, activationVmb = true, 0xf, Seq(2.0f, 2.0f, 2.0f, 2.0f))
    )

    val results = mutable.ArrayBuffer[Result]()
    SimConfig
      .withConfig(SpinalConfig(bitVectorWidthMax = 4096))
      .workspacePath(simWorkspace("MBPriorQScaleReconstructorSim"))
      .addSimulatorFlag("-CFLAGS").addSimulatorFlag("-std=c++14")
      .addSimulatorFlag("-LDFLAGS").addSimulatorFlag("-std=c++14")
      .compile(new MBPriorQScaleReconstructor)
      .doSim { dut =>
        cases.foreach { test =>
          dut.io.global_scale_w #= fp32(1.0f)
          dut.io.global_scale_i #= fp32(1.0f)
          dut.io.weight_scale_base #= fp8One
          dut.io.input_scale_base #= fp8Two
          dut.io.weight_scale_ext #= packFp8(Seq(fp8Two, fp8Four, fp8Half))
          dut.io.input_scale_ext #= packFp8(Seq(fp8One, fp8Half, fp8Four))
          dut.io.weight_vmb #= test.weightVmb
          dut.io.input_vmb #= test.activationVmb
          sleep(1)

          val actualMask = dut.io.factor_valid_mask.toInt
          val actualFactors = factors(dut.io.dequant_factors.toBigInt)
          val expectedFactors = test.expected.map(fp32)
          assert(actualMask == test.validMask,
            s"${test.name}: valid mask 0x${actualMask.toHexString}, expected 0x${test.validMask.toHexString}")
          assert(actualFactors == expectedFactors,
            s"${test.name}: factors ${actualFactors.map(hex32)} != ${expectedFactors.map(hex32)}")
          results += Result(test, actualMask, actualFactors)
        }
      }

    val resultPath = sys.env.getOrElse(
      "MBPRIORQ_SCALE_RECONSTRUCTOR_CSV",
      "target/ae-results/modules/scale_reconstructor.csv"
    )
    val file = new File(resultPath)
    Option(file.getParentFile).foreach(_.mkdirs())
    val out = new PrintWriter(file)
    try {
      out.println("case,weight_mask,activation_mask,weight_scales_fp8,activation_scales_fp8,expected_valid_sub_blocks,actual_valid_sub_blocks,expected_dequant_factors_fp32,actual_dequant_factors_fp32,status")
      results.foreach { result =>
        val expectedSubBlocks = if(result.test.validMask == 0x1) Seq(0) else Seq(0, 1, 2, 3)
        val actualSubBlocks = (0 until 4).filter(idx => ((result.validMask >> idx) & 1) == 1)
        out.println(Seq(
          result.test.name,
          if(result.test.weightVmb) 1 else 0,
          if(result.test.activationVmb) 1 else 0,
          csvList(Seq(1.0, 2.0, 4.0, 0.5)),
          csvList(Seq(2.0, 1.0, 0.5, 4.0)),
          csvList(expectedSubBlocks),
          csvList(actualSubBlocks),
          csvList(result.test.expected),
          csvList(result.factors.map(fp32Value)),
          "PASS"
        ).mkString(","))
      }
    } finally {
      out.close()
    }

    println(s"MBPriorQ scale reconstruction simulation passed. Results: $resultPath")
  }
}
