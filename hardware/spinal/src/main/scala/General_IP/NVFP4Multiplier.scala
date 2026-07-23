package General_IP
import spinal.core._
import spinal.lib._

class NVFP4Multiplier extends Component {
  val io = new Bundle {
    val a = in Bits(4 bits)
    val b = in Bits(4 bits)
    val result = out Bits(16 bits)
  }

  // Split sign and magnitude fields.
  private val signA: Bool = io.a(3)
  private val signB: Bool = io.b(3)
  private val absA: UInt  = io.a(2 downto 0).asUInt
  private val absB: UInt  = io.b(2 downto 0).asUInt

  private val aIsZero: Bool = (absA === 0)
  private val bIsZero: Bool = (absB === 0)
  private val resultIsZero: Bool = aIsZero || bIsZero

  // Sort magnitudes so commutative products share one LUT entry.
  private val AisLarger: Bool = absA >= absB
  private val larger: Bits = Mux(AisLarger, absA, absB).asBits
  private val smaller: Bits = Mux(AisLarger, absB, absA).asBits
  private val highIdx:Bits = (larger.asUInt - 1).asBits
  private val lowIdx: Bits = (smaller.asUInt - 1).asBits

  // Compact the triangular set of 28 non-zero magnitude pairs into five bits.
  private val MSIdx: Bits = Mux(lowIdx===B"100",
    B"0"##highIdx(1 downto 0),
    Mux(lowIdx === B"101",
      B"00"##highIdx(0),
      Mux(lowIdx === B"110",
        B"000",
        highIdx)
    ))
  private val LSIdx: Bits = Mux(lowIdx===B"100",
    B"11",
    Mux(lowIdx === B"101",
      B"10",
      Mux(lowIdx === B"110",
        B"01",
        lowIdx(1 downto 0))
    ))
  private val lutAddress: UInt = (MSIdx ## LSIdx).resize(5 bits).asUInt

  // Each LUT entry is the BF16 magnitude for one unique product.
  private val absValueLUT = Vec(
    B"16'h3E80", // 0.5 * 0.5 = 0.25
    B"16'h4210", // 6.0 * 6.0 = 36.0
    B"16'h41C0", // 6.0 * 4.0 = 24.0
    B"16'h4110", // 3.0 * 3.0 = 9.0

    B"16'h3F00", // 1.0 * 0.5 = 0.5
    B"16'h3F80", // 1.0 * 1.0 = 1.0
    B"16'h4180", // 4.0 * 4.0 = 16.0
    B"16'h4140", // 4.0 * 3.0 = 12.0

    B"16'h3F40", // 1.5 * 0.5 = 0.75
    B"16'h3FC0", // 1.5 * 1.0 = 1.5
    B"16'h4010", // 1.5 * 1.5 = 2.25
    B"16'h4190", // 6.0 * 3.0 = 18.0

    B"16'h3F80", // 2.0 * 0.5 = 1.0
    B"16'h4000", // 2.0 * 1.0 = 2.0
    B"16'h4040", // 2.0 * 1.5 = 3.0
    B"16'h4080", // 2.0 * 2.0 = 4.0

    B"16'h3FC0", // 3.0 * 0.5 = 1.5
    B"16'h4040", // 3.0 * 1.0 = 3.0
    B"16'h4090", // 3.0 * 1.5 = 4.5
    B"16'h40C0", // 3.0 * 2.0 = 6.0

    B"16'h4000", // 4.0 * 0.5 = 2.0
    B"16'h4080", // 4.0 * 1.0 = 4.0
    B"16'h40C0", // 4.0 * 1.5 = 6.0
    B"16'h4100", // 4.0 * 2.0 = 8.0

    B"16'h4040", // 6.0 * 0.5 = 3.0
    B"16'h40C0", // 6.0 * 1.0 = 6.0
    B"16'h4110", // 6.0 * 1.5 = 9.0
    B"16'h4140", // 6.0 * 2.0 = 12.0
  )

  private val resultSign:Bool = signA ^ signB

  private val absResult:Bits = absValueLUT(lutAddress)

  private val resultWithSign: Bits = Mux(resultSign,
    (B"1" ## absResult(14 downto 0)),
    absResult
  )

  io.result := Mux(resultIsZero, B(0, 16 bits), resultWithSign)
}
