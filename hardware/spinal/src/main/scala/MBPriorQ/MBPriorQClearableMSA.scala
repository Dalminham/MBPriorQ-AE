package MBPriorQ

import spinal.core._

/**
 * Small FIFO variant with an explicit clear input.
 *
 * It is used by the refined-aware scheduled MultiMSA wrapper.
 */
case class MBPriorQClearableFIFO(dataWidth: Int, depth: Int) extends Component {
  val io = new Bundle {
    val clear = in Bool()
    val write = new Bundle {
      val valid = in Bool()
      val ready = out Bool()
      val payload = in Bits(dataWidth bits)
    }
    val read = new Bundle {
      val valid = out Bool()
      val ready = in Bool()
      val payload = out Bits(dataWidth bits)
    }
  }

  private val ptrWidth = log2Up(depth)
  private val mem = Mem(Bits(dataWidth bits), depth)
  private val writePtr = Reg(UInt(ptrWidth bits)) init(0)
  private val readPtr = Reg(UInt(ptrWidth bits)) init(0)

  private val empty = readPtr === writePtr
  private val full =
    (writePtr === (readPtr - 1).resize(ptrWidth)) ||
      (writePtr === U(depth - 1, ptrWidth bits) && readPtr === 0)

  io.write.ready := !full
  io.read.valid := !empty
  io.read.payload := mem(readPtr)

  when(io.clear) {
    writePtr := 0
    readPtr := 0
  } otherwise {
    when(io.write.valid && !full) {
      mem(writePtr) := io.write.payload
      when(writePtr === U(depth - 1, ptrWidth bits)) {
        writePtr := 0
      } otherwise {
        writePtr := writePtr + 1
      }
    }

    when(io.read.ready && !empty) {
      when(readPtr === U(depth - 1, ptrWidth bits)) {
        readPtr := 0
      } otherwise {
        readPtr := readPtr + 1
      }
    }
  }
}

/**
 * MSA with clearable input and output FIFOs.
 *
 * The arithmetic datapath is the PEArray-based systolic path. The added clear
 * lets a physical MSA lane be safely returned to the shared lane pool
 * after its output has been accepted.
 */
class MBPriorQClearableMSA(WeightNum: Int, InputNum: Int) extends Component {
  val io = new Bundle {
    val clear = in Bool()
    val Valid = in Bool()
    val NeedSubCycle = in Bool()
    val Weight_Forward_Data = in Bits(4 * WeightNum bits)
    val Input_Forward_Data = in Bits(4 * InputNum bits)
    val OutputReady = in Bool()
    val Weight_Write_ready = out Bool()
    val Input_Write_ready = out Bool()
    val OutputData = out Bits(WeightNum * InputNum * 16 bits)
    val OutputValid = out Bool()
  }

  private val weightFifo = MBPriorQClearableFIFO(WeightNum * 4, 4)
  private val outputProduced = Reg(Bool()) init(False)
  private val acceptInput = io.Valid && !outputProduced

  weightFifo.io.clear := io.clear
  weightFifo.io.write.payload := io.Weight_Forward_Data
  weightFifo.io.write.valid := acceptInput
  weightFifo.io.read.ready := acceptInput
  io.Weight_Write_ready := weightFifo.io.write.ready

  private val inputFifo = MBPriorQClearableFIFO(InputNum * 4, 4)
  inputFifo.io.clear := io.clear
  inputFifo.io.write.payload := io.Input_Forward_Data
  inputFifo.io.write.valid := acceptInput
  inputFifo.io.read.ready := acceptInput
  io.Input_Write_ready := inputFifo.io.write.ready

  private val pes = new PEArray(WeightNum, InputNum)
  pes.io.WeightRow := weightFifo.io.read.payload
  pes.io.InputRow := inputFifo.io.read.payload
  pes.io.Valid := acceptInput && weightFifo.io.read.valid && inputFifo.io.read.valid
  pes.io.acc_cycle_sel := ~io.NeedSubCycle

  private val outputBuffer = MBPriorQClearableFIFO(16 * WeightNum * InputNum, 4)
  outputBuffer.io.clear := io.clear
  outputBuffer.io.write.payload := pes.io.Result
  outputBuffer.io.write.valid := pes.io.output_pulse
  pes.io.OutputReady := outputBuffer.io.write.ready
  io.OutputValid := outputBuffer.io.read.valid
  outputBuffer.io.read.ready := io.OutputReady
  io.OutputData := outputBuffer.io.read.payload

  when(io.clear) {
    outputProduced := False
  } otherwise {
    when(pes.io.output_pulse) {
      outputProduced := True
    }
  }
}
