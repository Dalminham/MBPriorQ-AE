package General_IP
import spinal.core._
import spinal.lib._

// BFloat16加法器实现
class BFloat16Adder extends Component {
  val io = new Bundle {
    val a      = in Bits(16 bits)
    val b      = in Bits(16 bits)
    val result = out Bits(16 bits)
  }

  // 解析BFloat16
  private val aSign: Bool = io.a(15)
  private val aExp: UInt = io.a(14 downto 7).asUInt
  private val aMant: UInt = io.a(6 downto 0).asUInt
  private val bSign: Bool = io.b(15)
  private val bExp: UInt = io.b(14 downto 7).asUInt
  private val bMant: UInt = io.b(6 downto 0).asUInt

  // 特殊值检测
  private val aIsNaN: Bool = aExp === 0xFF && aMant =/= 0
  private val aIsInf: Bool = aExp === 0xFF && aMant === 0
  private val bIsNaN: Bool = bExp === 0xFF && bMant =/= 0
  private val bIsInf: Bool = bExp === 0xFF && bMant === 0
  private val aIsZero: Bool = aExp === 0 && aMant === 0
  private val bIsZero: Bool = bExp === 0 && bMant === 0

  // 选择较大的操作数（基于指数和尾数）
  private val aGtB: Bool = (aExp > bExp) || (aExp === bExp && aMant > bMant)

  private val largerExp: UInt = Mux(aGtB, aExp, bExp)
  private val largerMant: UInt = Mux(aGtB, aMant, bMant)
  private val largerSign: Bool = Mux(aGtB, aSign, bSign)
  private val smallerExp: UInt = Mux(aGtB, bExp, aExp)
  private val smallerMant: UInt = Mux(aGtB, bMant, aMant)
  private val smallerSign: Bool = Mux(aGtB, bSign, aSign)

  // 计算指数差
  private val expDiff: UInt = largerExp - smallerExp
  private val DiffIsHuge: Bool = expDiff > 15

  // 添加隐藏位（指数为0时是非规格化数，隐藏位为0）
  private val largerHidden: UInt = Mux(largerExp === 0, U(0), U(1))
  private val smallerHidden:UInt = Mux(smallerExp === 0, U(0), U(1))

  // 扩展尾数为16位（1位隐藏位 + 7位尾数 + 8位保护位）
  private val largerMantExtended: UInt = Cat(largerHidden, largerMant, U(0, 8 bits)).asUInt // 16位
  private val smallerMantExtended: UInt = Cat(smallerHidden, smallerMant, U(0, 8 bits)).asUInt // 16位

  // 对齐尾数（限制移位量避免溢出）
  private val shiftAmount: UInt = Mux(DiffIsHuge, U(15), expDiff).resize(4 bits)
  private val smallerMantAligned: UInt = (smallerMantExtended |>> shiftAmount)

  // 尾数加减法（使用17位以保留可能的进位）
  private val sameSign: Bool = largerSign === smallerSign
  private val mantissaSum: UInt = UInt(17 bits)
  private val largerMantissa: UInt = (U"1'b0"##largerMantExtended).asUInt
  // 当阶数差距巨大时，小尾数直接为0
  private val smallerMantissa: UInt = Mux(DiffIsHuge, U(0), (U"1'b0"##smallerMantAligned).asUInt)
  private val resultSign: Bool = Bool()

  when(sameSign) {
    // 同号相加
    mantissaSum := largerMantissa + smallerMantissa
    resultSign := largerSign
  } otherwise {
    // 异号相减
    mantissaSum := largerMantissa - smallerMantissa
    resultSign := largerSign
  }

  // 前导零计数（计算需要左移多少位才能让最高位为1），扫描bit16..0
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
    lzCount := 16 // 全0情况
  }

  // 检测进位（第16位为1表示加法有进位）
  private val carry: Bool = mantissaSum(16)
  // 检测非规格化
  private val UnNorm: Bool = ~mantissaSum(15)

  // 归一化处理
  private val normalizedMantissa: UInt = UInt(17 bits)
  private val newExponent: UInt = UInt(8 bits)

  when(carry) {
    // 有进位，右移1位；隐藏位对齐到bit16
    normalizedMantissa := mantissaSum |>> 1
    newExponent := largerExp + 1
  } elsewhen(UnNorm){
    when(mantissaSum === 0){
      //同数相减
      normalizedMantissa := U(0)
      newExponent := U(0)
    } elsewhen(largerExp === U(0)){
      //非规格数输入
      normalizedMantissa := mantissaSum
      newExponent := U(0)
    }otherwise  {
      // 尾数非规格化，需要左移归一化
      // 左移前导零数量进行归一化（将最高1移到bit16）
      normalizedMantissa := (mantissaSum |<< lzCount)
      newExponent := (largerExp - lzCount)
    }
  } otherwise{
    normalizedMantissa := (mantissaSum)
    newExponent := largerExp
  }

  // 提取7位尾数结果（去掉隐藏位）
  // 归一化后，最高位（第16位）是进位，第15位是隐藏位，提取14:8位作为尾数
  private val resultMantissa: UInt = normalizedMantissa(14 downto 8)

  // 结果为零时，强制符号为0
  private val finalSign: Bool = Mux(mantissaSum===0, False, resultSign)

  // 特殊值处理
  when(aIsNaN || bIsNaN) {
    io.result := B"16'h7FC0" // NaN
  } elsewhen(aIsInf || bIsInf) {
    when(aIsInf && bIsInf && aSign =/= bSign) {
      io.result := B"16'h7FC0" // 无穷大相减 = NaN
    } elsewhen(aIsInf) {
      io.result := io.a
    } otherwise {
      io.result := io.b
    }
  } elsewhen(aIsZero && bIsZero) {
    // 两个零相加
    when(aSign && bSign) {
      io.result := B"16'h8000" // -0 + -0 = -0
    } otherwise {
      io.result := B"16'h0000" // 其他情况返回+0
    }
  } elsewhen(aIsZero) {
    io.result := io.b
  } elsewhen(bIsZero) {
    io.result := io.a
  } otherwise {
    // 正常情况
    // 检查是否溢出到无穷大
    when(newExponent === 0xFF) {
      io.result := Cat(finalSign, B"8'hFF", B"7'h00") // 溢出到无穷大
    } otherwise {
      io.result := Cat(finalSign, newExponent, resultMantissa)
    }
  }
}