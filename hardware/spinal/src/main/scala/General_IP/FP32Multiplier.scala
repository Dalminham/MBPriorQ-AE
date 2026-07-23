package General_IP
import spinal.core._
import spinal.lib._

class FP32Multiplier extends Component {
  val io = new Bundle {
    val a      = in Bits(32 bits)
    val b      = in Bits(32 bits)
    val result = out Bits(32 bits)
  }

  // Decode operands and classify special values.
  private val aSign: Bool = io.a(31)
  private val aExp: UInt = io.a(30 downto 23).asUInt
  private val aMant: UInt = io.a(22 downto 0).asUInt
  private val bSign: Bool = io.b(31)
  private val bExp: UInt = io.b(30 downto 23).asUInt
  private val bMant: UInt = io.b(22 downto 0).asUInt

  private val aIsNaN: Bool = aExp === 0xFF && aMant =/= 0
  private val aIsInf: Bool = aExp === 0xFF && aMant === 0
  private val bIsNaN: Bool = bExp === 0xFF && bMant =/= 0
  private val bIsInf: Bool = bExp === 0xFF && bMant === 0
  private val aIsZero: Bool = aExp === 0 && aMant === 0
  private val bIsZero: Bool = bExp === 0 && bMant === 0
  private val aIsDenorm: Bool = aExp === 0 && aMant =/= 0
  private val bIsDenorm: Bool = bExp === 0 && bMant =/= 0

  private val resultSign: Bool = aSign ^ bSign

  private val aHidden: UInt = Mux(aExp === 0, U(0), U(1))
  private val bHidden: UInt = Mux(bExp === 0, U(0), U(1))

  private val aMant24: UInt = Cat(aHidden, aMant).asUInt
  private val bMant24: UInt = Cat(bHidden, bMant).asUInt

  private val mantProduct48: UInt = aMant24 * bMant24

  // Multiply significands and combine unbiased exponents.
  private val aActualExp: SInt = Mux(aIsDenorm, S(-126), (aExp.asSInt - 127)).resize(12 bits)
  private val bActualExp: SInt = Mux(bIsDenorm, S(-126), (bExp.asSInt - 127)).resize(12 bits)
  private val sumActualExp: SInt = aActualExp + bActualExp
  private val expSumWithBias: SInt = sumActualExp + 127
  private val expSumWillUnderflowBeforeNorm: Bool = expSumWithBias < 0
  private val expSumFull: UInt = Mux(expSumWithBias < 0, U(0), expSumWithBias.asUInt.resize(12 bits))
  private val expSum: UInt = Mux(expSumWithBias < 0, U(0), expSumWithBias.asUInt.resize(9 bits))
  private val expSumWillOverflowBeforeNorm: Bool = expSumFull >= 255

  // Normalize the 48-bit significand product.
  private val hasCarry: Bool = mantProduct48(47)
  private val normalizedMant: UInt = UInt(48 bits)
  private val finalExp: UInt = UInt(8 bits)
  private val lzCount = UInt(7 bits)
  private val expAfterShift = SInt(10 bits)

  private val expSumWillOverflow: Bool = expSumWillOverflowBeforeNorm || (expSum >= 254)
  private val expSumWillUnderflow: Bool = expSum.asSInt < 0

  when(hasCarry) {
    normalizedMant := mantProduct48 |>> 1
    finalExp := Mux(expSumWillOverflow, U(0xFF), (expSum.resize(8 bits) + 1))
    lzCount := 0
    expAfterShift := Mux(expSumWillOverflow, S(255), (expSum.asSInt + 1).resize(10 bits))
  } otherwise {
    when(mantProduct48(46)) {
      lzCount := 0
    } elsewhen(mantProduct48(45)) {
      lzCount := 1
    } elsewhen(mantProduct48(44)) {
      lzCount := 2
    } elsewhen(mantProduct48(43)) {
      lzCount := 3
    } elsewhen(mantProduct48(42)) {
      lzCount := 4
    } elsewhen(mantProduct48(41)) {
      lzCount := 5
    } elsewhen(mantProduct48(40)) {
      lzCount := 6
    } elsewhen(mantProduct48(39)) {
      lzCount := 7
    } elsewhen(mantProduct48(38)) {
      lzCount := 8
    } elsewhen(mantProduct48(37)) {
      lzCount := 9
    } elsewhen(mantProduct48(36)) {
      lzCount := 10
    } elsewhen(mantProduct48(35)) {
      lzCount := 11
    } elsewhen(mantProduct48(34)) {
      lzCount := 12
    } elsewhen(mantProduct48(33)) {
      lzCount := 13
    } elsewhen(mantProduct48(32)) {
      lzCount := 14
    } elsewhen(mantProduct48(31)) {
      lzCount := 15
    } elsewhen(mantProduct48(30)) {
      lzCount := 16
    } elsewhen(mantProduct48(29)) {
      lzCount := 17
    } elsewhen(mantProduct48(28)) {
      lzCount := 18
    } elsewhen(mantProduct48(27)) {
      lzCount := 19
    } elsewhen(mantProduct48(26)) {
      lzCount := 20
    } elsewhen(mantProduct48(25)) {
      lzCount := 21
    } elsewhen(mantProduct48(24)) {
      lzCount := 22
    } elsewhen(mantProduct48(23)) {
      lzCount := 23
    } elsewhen(mantProduct48(22)) {
      lzCount := 24
    } elsewhen(mantProduct48(21)) {
      lzCount := 25
    } elsewhen(mantProduct48(20)) {
      lzCount := 26
    } elsewhen(mantProduct48(19)) {
      lzCount := 27
    } elsewhen(mantProduct48(18)) {
      lzCount := 28
    } elsewhen(mantProduct48(17)) {
      lzCount := 29
    } elsewhen(mantProduct48(16)) {
      lzCount := 30
    } elsewhen(mantProduct48(15)) {
      lzCount := 31
    } elsewhen(mantProduct48(14)) {
      lzCount := 32
    } elsewhen(mantProduct48(13)) {
      lzCount := 33
    } elsewhen(mantProduct48(12)) {
      lzCount := 34
    } elsewhen(mantProduct48(11)) {
      lzCount := 35
    } elsewhen(mantProduct48(10)) {
      lzCount := 36
    } elsewhen(mantProduct48(9)) {
      lzCount := 37
    } elsewhen(mantProduct48(8)) {
      lzCount := 38
    } elsewhen(mantProduct48(7)) {
      lzCount := 39
    } elsewhen(mantProduct48(6)) {
      lzCount := 40
    } elsewhen(mantProduct48(5)) {
      lzCount := 41
    } elsewhen(mantProduct48(4)) {
      lzCount := 42
    } elsewhen(mantProduct48(3)) {
      lzCount := 43
    } elsewhen(mantProduct48(2)) {
      lzCount := 44
    } elsewhen(mantProduct48(1)) {
      lzCount := 45
    } elsewhen(mantProduct48(0)) {
      lzCount := 46
    } otherwise {
      lzCount := 47
    }

    normalizedMant := mantProduct48 |<< lzCount
    expAfterShift := (expSum.asSInt - lzCount.resize(9 bits).asSInt).resize(10 bits)
    finalExp := Mux(expAfterShift < 0, U(0), expAfterShift.asUInt.resize(8 bits))

    when(expAfterShift < 0) {
      normalizedMant := U(0)
    }
  }

  // Apply IEEE 754 round-to-nearest-even.
  private val mantissaBits: UInt = normalizedMant(45 downto 23)
  private val roundBit: Bool = normalizedMant(22)
  private val stickyBits: UInt = normalizedMant(21 downto 0)
  private val sticky: Bool = stickyBits =/= 0
  private val lsb: Bool = mantissaBits(0)

  private val shouldRoundUp: Bool = roundBit && (sticky || lsb)
  private val mantissaRounded: UInt = (mantissaBits + shouldRoundUp.asUInt).resize(24 bits)

  private val mantissaCarry: Bool = mantissaRounded(23)
  private val resultMantissa: UInt = Mux(mantissaCarry,
    U(0),
    mantissaRounded(22 downto 0)
  )

  private val finalExpWithRound: UInt = Mux(mantissaCarry, finalExp + 1, finalExp)

  private val resultExpIsZero: Bool = finalExp === 0
  private val resultIsUnderflow: Bool = resultExpIsZero && (resultMantissa === 0)

  // Select IEEE special cases or the normalized product.
  when(aIsNaN || bIsNaN) {
    io.result := B"32'h7FC00000" // NaN
  } elsewhen((aIsInf || bIsInf) && (aIsZero || bIsZero)) {
    io.result := B"32'h7FC00000" // Inf × 0 = NaN
  } elsewhen(aIsInf || bIsInf) {
    io.result := Cat(resultSign, B"8'hFF", B"23'h000000")
  } elsewhen(aIsZero || bIsZero) {
    io.result := Cat(resultSign, B"8'h00", B"23'h000000")
  } otherwise {
    when(expSumWillUnderflowBeforeNorm) {
      io.result := Cat(resultSign, B"8'h00", B"23'h000000")
    } elsewhen(expSumWillOverflowBeforeNorm || finalExpWithRound >= 0xFF || (mantissaCarry && finalExp >= 0xFE)) {
      io.result := Cat(resultSign, B"8'hFF", B"23'h000000")
    } elsewhen(resultIsUnderflow || (finalExp === 0 && resultMantissa === 0)) {
      io.result := Cat(resultSign, B"8'h00", B"23'h000000")
    } elsewhen(finalExpWithRound === 0) {
      io.result := Cat(resultSign, B"8'h00", resultMantissa)
    } otherwise {
      io.result := Cat(resultSign, finalExpWithRound, resultMantissa)
    }
  }
}
