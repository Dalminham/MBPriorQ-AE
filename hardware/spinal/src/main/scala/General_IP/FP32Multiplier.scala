package General_IP
import spinal.core._
import spinal.lib._

// FP32单精度浮点乘法器实现
class FP32Multiplier extends Component {
  val io = new Bundle {
    val a      = in Bits(32 bits)
    val b      = in Bits(32 bits)
    val result = out Bits(32 bits)
  }

  // 解析FP32格式：1位符号 + 8位指数 + 23位尾数
  private val aSign: Bool = io.a(31)
  private val aExp: UInt = io.a(30 downto 23).asUInt
  private val aMant: UInt = io.a(22 downto 0).asUInt
  private val bSign: Bool = io.b(31)
  private val bExp: UInt = io.b(30 downto 23).asUInt
  private val bMant: UInt = io.b(22 downto 0).asUInt

  // 特殊值检测
  private val aIsNaN: Bool = aExp === 0xFF && aMant =/= 0
  private val aIsInf: Bool = aExp === 0xFF && aMant === 0
  private val bIsNaN: Bool = bExp === 0xFF && bMant =/= 0
  private val bIsInf: Bool = bExp === 0xFF && bMant === 0
  private val aIsZero: Bool = aExp === 0 && aMant === 0
  private val bIsZero: Bool = bExp === 0 && bMant === 0
  private val aIsDenorm: Bool = aExp === 0 && aMant =/= 0  // 非规格化数
  private val bIsDenorm: Bool = bExp === 0 && bMant =/= 0  // 非规格化数

  // 结果符号位：符号异或
  private val resultSign: Bool = aSign ^ bSign

  // 添加隐含位（指数为0时是非规格化数，隐含位为0；否则为1）
  private val aHidden: UInt = Mux(aExp === 0, U(0), U(1))
  private val bHidden: UInt = Mux(bExp === 0, U(0), U(1))

  // 构建24位有效尾数：1位隐含位 + 23位尾数
  private val aMant24: UInt = Cat(aHidden, aMant).asUInt // 24位
  private val bMant24: UInt = Cat(bHidden, bMant).asUInt // 24位

  // 尾数相乘（24位 × 24位 = 48位）
  private val mantProduct48: UInt = aMant24 * bMant24 // 48位结果

  // 计算新指数：非规格化数的实际指数是-126（对应指数字段0）
  // 对于非规格化数，实际指数 = -126；对于规格化数，实际指数 = exp - 127
  // 乘法：实际指数 = (实际指数_a) + (实际指数_b)
  // 所以：新指数字段 = (实际指数) + 127
  // 如果输入是非规格化数，实际指数是-126，否则是 exp - 127
  private val aActualExp: SInt = Mux(aIsDenorm, S(-126), (aExp.asSInt - 127)).resize(12 bits)
  private val bActualExp: SInt = Mux(bIsDenorm, S(-126), (bExp.asSInt - 127)).resize(12 bits)
  private val sumActualExp: SInt = aActualExp + bActualExp // 实际指数和（12位，范围约-4096到4095）
  private val expSumWithBias: SInt = sumActualExp + 127 // 加上偏移后的指数（仍为有符号数）
  // 检查是否下溢：如果 expSumWithBias < 0，则下溢（即使归一化后也会下溢）
  private val expSumWillUnderflowBeforeNorm: Bool = expSumWithBias < 0
  // 转换为无符号数用于溢出检测（只有当非负时才转换）
  private val expSumFull: UInt = Mux(expSumWithBias < 0, U(0), expSumWithBias.asUInt.resize(12 bits))
  private val expSum: UInt = Mux(expSumWithBias < 0, U(0), expSumWithBias.asUInt.resize(9 bits)) // 9位用于归一化计算
  // 检查指数溢出：如果expSumFull >= 255，或者在有进位时 >= 254（加1后会>=255）
  private val expSumWillOverflowBeforeNorm: Bool = expSumFull >= 255

  // 检测尾数乘积的归一化：检查最高位（第47位）
  private val hasCarry: Bool = mantProduct48(47) // 如果bit47=1，说明结果>=2，需要右移
  private val normalizedMant: UInt = UInt(48 bits)
  private val finalExp: UInt = UInt(8 bits)
  private val lzCount = UInt(7 bits) // 前导零计数（用于归一化，最大值47）
  private val expAfterShift = SInt(10 bits) // 归一化后的指数（用于检测下溢）

  // 检查指数是否会在归一化后溢出（在归一化之前检查）
  private val expSumWillOverflow: Bool = expSumWillOverflowBeforeNorm || (expSum >= 254) // expSum + 1 >= 255 会导致溢出
  private val expSumWillUnderflow: Bool = expSum.asSInt < 0 // 指数下溢（9位时检查）

  when(hasCarry) {
    // 结果 >= 2，右移1位（bit47移到bit46，作为新的隐含位）
    normalizedMant := mantProduct48 |>> 1
    // 检查溢出：如果expSum >= 254，加1后会>=255（溢出）
    finalExp := Mux(expSumWillOverflow, U(0xFF), (expSum.resize(8 bits) + 1))
    lzCount := 0 // 有进位时不需要前导零计数，赋值避免锁存器
    expAfterShift := Mux(expSumWillOverflow, S(255), (expSum.asSInt + 1).resize(10 bits))
  } otherwise {
    // 结果 < 2，需要左移归一化（查找最高位1的位置，使其成为bit46）
    // 查找从bit46开始向下的最高位1
    when(mantProduct48(46)) {
      lzCount := 0  // 已经在正确位置
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
      lzCount := 47 // 全0情况
    }

    normalizedMant := mantProduct48 |<< lzCount
    // 计算最终指数：如果左移太多，可能导致指数下溢
    expAfterShift := (expSum.asSInt - lzCount.resize(9 bits).asSInt).resize(10 bits)
    finalExp := Mux(expAfterShift < 0, U(0), expAfterShift.asUInt.resize(8 bits))

    // 如果指数下溢，尾数也应该置0
    when(expAfterShift < 0) {
      normalizedMant := U(0)
    }
  }

  // 提取尾数并进行舍入（IEEE 754 round to nearest even）
  // bit 46: 隐含位（不包含在结果中）
  // bit 45:23: 23位尾数（结果的前23位）
  // bit 22: 舍入位（guard bit）
  // bit 21:0: 粘滞位（sticky bits）
  private val mantissaBits: UInt = normalizedMant(45 downto 23) // 23位尾数
  private val roundBit: Bool = normalizedMant(22) // 舍入位
  private val stickyBits: UInt = normalizedMant(21 downto 0) // 粘滞位
  private val sticky: Bool = stickyBits =/= 0 // 如果有任何粘滞位为1，则sticky=true
  private val lsb: Bool = mantissaBits(0) // 尾数最低位（用于舍入到最近偶数）

  // IEEE 754 round to nearest even (RNE)
  // 规则：如果 roundBit=1 且 (sticky=1 或 lsb=1)，则向上舍入（加1）
  // 注意：如果 roundBit=1 且 sticky=0 且 lsb=0，则舍入到偶数（向下舍入，即不加1）
  private val shouldRoundUp: Bool = roundBit && (sticky || lsb)
  private val mantissaRounded: UInt = (mantissaBits + shouldRoundUp.asUInt).resize(24 bits) // 扩展到24位以检测进位

  // 检查舍入后是否会产生进位（溢出到隐含位）
  // 如果mantissaRounded的第23位（隐含位位置）为1，说明有进位
  private val mantissaCarry: Bool = mantissaRounded(23)
  private val resultMantissa: UInt = Mux(mantissaCarry,
    U(0), // 有进位，尾数置为0（隐含位为1，指数会加1）
    mantissaRounded(22 downto 0)  // 无进位，取bit 22:0（23位尾数）
  )

  // 如果舍入产生进位，指数需要加1
  private val finalExpWithRound: UInt = Mux(mantissaCarry, finalExp + 1, finalExp)

  // 检查结果是否下溢：如果归一化后的指数为0且尾数太小，应该返回0
  private val resultExpIsZero: Bool = finalExp === 0
  private val resultIsUnderflow: Bool = resultExpIsZero && (resultMantissa === 0)

  // 特殊值处理
  when(aIsNaN || bIsNaN) {
    io.result := B"32'h7FC00000" // NaN
  } elsewhen((aIsInf || bIsInf) && (aIsZero || bIsZero)) {
    io.result := B"32'h7FC00000" // Inf × 0 = NaN
  } elsewhen(aIsInf || bIsInf) {
    io.result := Cat(resultSign, B"8'hFF", B"23'h000000") // 无穷大
  } elsewhen(aIsZero || bIsZero) {
    io.result := Cat(resultSign, B"8'h00", B"23'h000000") // 零
  } otherwise {
    // 正常情况
    // 检查溢出和下溢
    // 首先检查归一化前的下溢（expSumWithBias < 0）
    when(expSumWillUnderflowBeforeNorm) {
      // 指数在下溢范围内，即使归一化也会下溢到0
      io.result := Cat(resultSign, B"8'h00", B"23'h000000") // 下溢到零
    } elsewhen(expSumWillOverflowBeforeNorm || finalExpWithRound >= 0xFF || (mantissaCarry && finalExp >= 0xFE)) {
      // 然后检查归一化前的溢出（expSumFull >= 255 或 finalExp >= 255，或舍入导致溢出）
      io.result := Cat(resultSign, B"8'hFF", B"23'h000000") // 溢出到无穷大
    } elsewhen(resultIsUnderflow || (finalExp === 0 && resultMantissa === 0)) {
      // 下溢到零：指数为0且尾数也为0，或者归一化后指数下溢
      io.result := Cat(resultSign, B"8'h00", B"23'h000000") // 下溢到零
    } elsewhen(finalExpWithRound === 0) {
      // 非规格化数：指数为0但尾数非0
      io.result := Cat(resultSign, B"8'h00", resultMantissa)
    } otherwise {
      io.result := Cat(resultSign, finalExpWithRound, resultMantissa)
    }
  }
}
