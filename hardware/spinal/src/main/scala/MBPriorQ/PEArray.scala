package MBPriorQ
import spinal.core._
import spinal.lib._
import General_IP.{BFloat16Adder, NVFP4Multiplier}

class PE extends Component {
  val io = new Bundle {
        //
        val Valid = in Bool()
        val OutputEn = in Bool()
        val Weight  = in Bits(4 bits)
        val Input   = in Bits(4 bits)
        //
        val Forwarded_Weight  = out Bits(4 bits)
        val Forwarded_Input   = out Bits(4 bits)
        val Accumulator_Result = out Bits(16 bits)
      }
  // 1st level NVFP4 Multiplier
  private val multiplier = new NVFP4Multiplier
  multiplier.io.a := io.Input
  multiplier.io.b := io.Weight
  private val product = multiplier.io.result

  // 2nd level BF16 Adder
  private val Accumulator = Reg(Bits(16 bits)) init(0)
  private val adder = new BFloat16Adder
  adder.io.a := product
  adder.io.b := Accumulator

  when(io.Valid){
    when(io.OutputEn){
      Accumulator := B"16'd0"
    }otherwise{
      Accumulator := adder.io.result
    }
  }

  io.Accumulator_Result := adder.io.result
  io.Forwarded_Weight := io.Weight
  io.Forwarded_Input := io.Input
}
case class PE_Row(PE_Num:Int) extends Component{
  val io = new Bundle{
    val Valid = in Bool()
    val OutputEn = in Bool()
    val InputRow = in Bits (4 * PE_Num bits)
    val Weight    = in Bits(4 bits)

    val ForwardedInputRow  = out Bits(4*PE_Num bits)
    val ResRow             = out Bits(16*PE_Num bits)
  }

  // Initialize PE_Num NVFP4 MAC uints
//  println("PE_Num",PE_Num)
  private val PEs = Array.fill(PE_Num)(new PE)

  // connect the input of each PE
  PEs(0).io.Valid    := io.Valid
  PEs(0).io.OutputEn := io.OutputEn
  PEs(0).io.Input    := io.InputRow(3 downto 0)
  PEs(0).io.Weight   := io.Weight
  for(i <- 1 until PE_Num) {
    PEs(i).io.Input    := io.InputRow(4*i+3 downto 4*i)
    PEs(i).io.Weight   := PEs(i - 1).io.Forwarded_Weight
    PEs(i).io.Valid    := io.Valid
    PEs(i).io.OutputEn := io.OutputEn
  }

  // Connect the output of each PE
  for (i <- 0 until PE_Num) {
      io.ForwardedInputRow(4*i+3 downto 4*i) := PEs(i).io.Forwarded_Input
      io.ResRow(16*i+15 downto 16*i) := PEs(i).io.Accumulator_Result
    }
}
class PEArray(WeightNum:Int, InputNum:Int) extends Component {
  val io = new Bundle{
    // Input
    val Valid = in Bool()
    val OutputReady = in Bool()
    val acc_cycle_sel = in Bool() // False: emit after 4 MACs; true: emit after 16 MACs.
    val WeightRow = in Bits(4*WeightNum bits)
    val InputRow = in Bits(4*InputNum bits)
    // Output
    val output_pulse = out Bool()
    val Result = out Bits(16*WeightNum*InputNum bits)
  }
  // Emit one output pulse after the selected accumulation interval.
  private val cycle_counter = Reg(UInt(4 bits)) init (0)
  private val target_value: UInt = Mux(io.acc_cycle_sel, U"4'd15", U"4'd3")
  private val cycle_counter_next: UInt = cycle_counter + U"4'd1"
  private val output_enable: Bool = cycle_counter === target_value
  private val output_fire: Bool = io.Valid & output_enable & io.OutputReady
  io.output_pulse := output_fire

  // Initialize WeightNum PE_Rows
  private val PE_Rows = Array.fill(WeightNum)(PE_Row(InputNum))
  // Connect the input of the first PE_Row
  PE_Rows(0).io.Valid    := io.Valid
  PE_Rows(0).io.OutputEn := output_enable
  PE_Rows(0).io.InputRow := io.InputRow
  PE_Rows(0).io.Weight   := io.WeightRow(3 downto 0)
  // Connect the input of the rest of PE_Rows
  for(i <- 1 until WeightNum) {
    PE_Rows(i).io.Valid    := io.Valid
    PE_Rows(i).io.OutputEn := output_enable
    PE_Rows(i).io.InputRow := PE_Rows(i-1).io.ForwardedInputRow
    PE_Rows(i).io.Weight   := io.WeightRow(4*i+3 downto 4*i)
  }
  //Connect the output of each PE_Row to ResArray
  for (i <- 0 until WeightNum) {
    io.Result(i*(16*InputNum)+(16*InputNum-1) downto i*(16*InputNum)) := PE_Rows(i).io.ResRow
  }

  when(io.Valid){
    when(output_enable){
      when(io.OutputReady) {
      cycle_counter := U"4'd0"
      }
    }otherwise {
      cycle_counter := cycle_counter_next
    }
  }

}
