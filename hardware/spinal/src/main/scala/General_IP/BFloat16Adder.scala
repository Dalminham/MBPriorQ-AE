package General_IP
import spinal.core._
import spinal.lib._

class BFloat16Adder extends Component {
  val io = new Bundle {
    val a      = in Bits(16 bits)
    val b      = in Bits(16 bits)
    val result = out Bits(16 bits)
  }

  // Decode operands and classify special values.
  private val aSign: Bool = io.a(15)
  private val aExp: UInt = io.a(14 downto 7).asUInt
  private val aMant: UInt = io.a(6 downto 0).asUInt
  private val bSign: Bool = io.b(15)
  private val bExp: UInt = io.b(14 downto 7).asUInt
  private val bMant: UInt = io.b(6 downto 0).asUInt

  private val aIsNaN: Bool = aExp === 0xFF && aMant =/= 0
  private val aIsInf: Bool = aExp === 0xFF && aMant === 0
  private val bIsNaN: Bool = bExp === 0xFF && bMant =/= 0
  private val bIsInf: Bool = bExp === 0xFF && bMant === 0
  private val aIsZero: Bool = aExp === 0 && aMant === 0
  private val bIsZero: Bool = bExp === 0 && bMant === 0

  // Align the smaller operand before add/subtract.
  private val aGtB: Bool = (aExp > bExp) || (aExp === bExp && aMant > bMant)

  private val largerExp: UInt = Mux(aGtB, aExp, bExp)
  private val largerMant: UInt = Mux(aGtB, aMant, bMant)
  private val largerSign: Bool = Mux(aGtB, aSign, bSign)
  private val smallerExp: UInt = Mux(aGtB, bExp, aExp)
  private val smallerMant: UInt = Mux(aGtB, bMant, aMant)
  private val smallerSign: Bool = Mux(aGtB, bSign, aSign)

  private val expDiff: UInt = largerExp - smallerExp
  private val DiffIsHuge: Bool = expDiff > 15

  private val largerHidden: UInt = Mux(largerExp === 0, U(0), U(1))
  private val smallerHidden:UInt = Mux(smallerExp === 0, U(0), U(1))

  private val largerMantExtended: UInt = Cat(largerHidden, largerMant, U(0, 8 bits)).asUInt
  private val smallerMantExtended: UInt = Cat(smallerHidden, smallerMant, U(0, 8 bits)).asUInt

  private val shiftAmount: UInt = Mux(DiffIsHuge, U(15), expDiff).resize(4 bits)
  private val smallerMantAligned: UInt = (smallerMantExtended |>> shiftAmount)

  private val sameSign: Bool = largerSign === smallerSign
  private val mantissaSum: UInt = UInt(17 bits)
  private val largerMantissa: UInt = (U"1'b0"##largerMantExtended).asUInt
  private val smallerMantissa: UInt = Mux(DiffIsHuge, U(0), (U"1'b0"##smallerMantAligned).asUInt)
  private val resultSign: Bool = Bool()

  when(sameSign) {
    mantissaSum := largerMantissa + smallerMantissa
    resultSign := largerSign
  } otherwise {
    mantissaSum := largerMantissa - smallerMantissa
    resultSign := largerSign
  }

  // Count leading zeros for post-add normalization.
  private val lzCount = UInt(5 bits)

  when(mantissaSum(16)) {
    lzCount := 0
  } elsewhen(mantissaSum(15)) {
    lzCount := 0
  } elsewhen(mantissaSum(14)) {
    lzCount := 1
  } elsewhen(mantissaSum(13)) {
    lzCount := 2
  } elsewhen(mantissaSum(12)) {
    lzCount := 3
  } elsewhen(mantissaSum(11)) {
    lzCount := 4
  } elsewhen(mantissaSum(10)) {
    lzCount := 5
  } elsewhen(mantissaSum(9)) {
    lzCount := 6
  } elsewhen(mantissaSum(8)) {
    lzCount := 7
  } elsewhen(mantissaSum(7)) {
    lzCount := 8
  } elsewhen(mantissaSum(6)) {
    lzCount := 9
  } elsewhen(mantissaSum(5)) {
    lzCount := 10
  } elsewhen(mantissaSum(4)) {
    lzCount := 11
  } elsewhen(mantissaSum(3)) {
    lzCount := 12
  } elsewhen(mantissaSum(2)) {
    lzCount := 13
  } elsewhen(mantissaSum(1)) {
    lzCount := 14
  } elsewhen(mantissaSum(0)) {
    lzCount := 15
  } otherwise {
    lzCount := 16
  }

  private val carry: Bool = mantissaSum(16)
  private val UnNorm: Bool = ~mantissaSum(15)

  // Normalize the result and update its exponent.
  private val normalizedMantissa: UInt = UInt(17 bits)
  private val newExponent: UInt = UInt(8 bits)

  when(carry) {
    normalizedMantissa := mantissaSum |>> 1
    newExponent := largerExp + 1
  } elsewhen(UnNorm){
    when(mantissaSum === 0){
      normalizedMantissa := U(0)
      newExponent := U(0)
    } elsewhen(largerExp === U(0)){
      normalizedMantissa := mantissaSum
      newExponent := U(0)
    }otherwise  {
      normalizedMantissa := (mantissaSum |<< lzCount)
      newExponent := (largerExp - lzCount)
    }
  } otherwise{
    normalizedMantissa := (mantissaSum)
    newExponent := largerExp
  }

  private val resultMantissa: UInt = normalizedMantissa(14 downto 8)

  private val finalSign: Bool = Mux(mantissaSum===0, False, resultSign)

  // Select IEEE special cases or the normalized result.
  when(aIsNaN || bIsNaN) {
    io.result := B"16'h7FC0" // NaN
  } elsewhen(aIsInf || bIsInf) {
    when(aIsInf && bIsInf && aSign =/= bSign) {
      io.result := B"16'h7FC0"
    } elsewhen(aIsInf) {
      io.result := io.a
    } otherwise {
      io.result := io.b
    }
  } elsewhen(aIsZero && bIsZero) {
    when(aSign && bSign) {
      io.result := B"16'h8000" // -0 + -0 = -0
    } otherwise {
      io.result := B"16'h0000"
    }
  } elsewhen(aIsZero) {
    io.result := io.b
  } elsewhen(bIsZero) {
    io.result := io.a
  } otherwise {
    when(newExponent === 0xFF) {
      io.result := Cat(finalSign, B"8'hFF", B"7'h00")
    } otherwise {
      io.result := Cat(finalSign, newExponent, resultMantissa)
    }
  }
}
