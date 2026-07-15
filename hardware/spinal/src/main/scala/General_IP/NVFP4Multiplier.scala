package General_IP
import spinal.core._
import spinal.lib._

class NVFP4Multiplier extends Component {
  val io = new Bundle {
    val a = in Bits(4 bits)
    val b = in Bits(4 bits)
    val result = out Bits(16 bits)
  }

  // 提取符号位和数值部分
  private val signA: Bool = io.a(3)
  private val signB: Bool = io.b(3)
  private val absA: UInt  = io.a(2 downto 0).asUInt  // 3位数值部分
  private val absB: UInt  = io.b(2 downto 0).asUInt  // 3位数值部分

  // 检查是否为零输入
  private val aIsZero: Bool = (absA === 0)
  private val bIsZero: Bool = (absB === 0)
  private val resultIsZero: Bool = aIsZero || bIsZero

  // 比较并排序：larger ## smaller
  private val AisLarger: Bool = absA >= absB
  private val larger: Bits = Mux(AisLarger, absA, absB).asBits
  private val smaller: Bits = Mux(AisLarger, absB, absA).asBits
  private val highIdx:Bits = (larger.asUInt - 1).asBits
  private val lowIdx: Bits = (smaller.asUInt - 1).asBits

  // 基于插队逻辑生成LUT地址
  // - 输入幅值按绝对值排序并-1映射：highIdx = larger-1；lowIdx = smaller-1（范围 000..110）
  // - LUT地址为5位：lutAddress = MSIdx(3) ## LSIdx(2)
  // - 默认：MSIdx = highIdx；LSIdx = lowIdx[1:0]
  // - 高效插队（当 lowIdx[2] == 1，即 lowIdx ∈ {100,101,110}）：
  //     lowIdx==100: MSIdx = 0 ## highIdx[1:0]，LSIdx = 11
  //     lowIdx==101: MSIdx = 00 ## highIdx[0]， LSIdx = 10
  //     lowIdx==110: MSIdx = 000，               LSIdx = 01
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

  // 优化后的LUT：只存储唯一乘法组合的绝对值结果
  // 由于乘法满足交换律，a*b和b*a结果相同，LUT只需要存储C(2,7)+7=21+7=28种结果
  private val absValueLUT = Vec(
    B"16'h3E80", // 0.5 * 0.5 = 0.25 (索引: 000_00)
    B"16'h4210", // 6.0 * 6.0 = 36.0  (索引: 000_01) 01插队，原lowIdx=110
    B"16'h41C0", // 6.0 * 4.0 = 24.0  (索引: 000_10) 10插队，原101
    B"16'h4110", // 3.0 * 3.0 = 9.0   (索引: 000_11) 11插队，原100

    B"16'h3F00", // 1.0 * 0.5 = 0.5   (索引: 001_00)
    B"16'h3F80", // 1.0 * 1.0 = 1.0   (索引: 001_01)
    B"16'h4180", // 4.0 * 4.0 = 16.0 (索引：001_10) 10插队，原101
    B"16'h4140", // 4.0 * 3.0 = 12.0  (索引: 001_11) 11插队，原100

    B"16'h3F40", // 1.5 * 0.5 = 0.75  (索引: 010_00)
    B"16'h3FC0", // 1.5 * 1.0 = 1.5   (索引: 010_01)
    B"16'h4010", // 1.5 * 1.5 = 2.25  (索引: 010_10)
    B"16'h4190", // 6.0 * 3.0 = 18.0  (索引: 010_11) 11插队，原100

    B"16'h3F80", // 2.0 * 0.5 = 1.0   (索引: 011_00)
    B"16'h4000", // 2.0 * 1.0 = 2.0   (索引: 011_01)
    B"16'h4040", // 2.0 * 1.5 = 3.0   (索引: 011_10)
    B"16'h4080", // 2.0 * 2.0 = 4.0   (索引: 011_11)

    B"16'h3FC0", // 3.0 * 0.5 = 1.5   (索引: 100_00)
    B"16'h4040", // 3.0 * 1.0 = 3.0   (索引: 100_01)
    B"16'h4090", // 3.0 * 1.5 = 4.5   (索引: 100_10)
    B"16'h40C0", // 3.0 * 2.0 = 6.0   (索引: 100_11)

    B"16'h4000", // 4.0 * 0.5 = 2.0   (索引: 101_00)
    B"16'h4080", // 4.0 * 1.0 = 4.0   (索引: 101_01)
    B"16'h40C0", // 4.0 * 1.5 = 6.0   (索引: 101_10)
    B"16'h4100", // 4.0 * 2.0 = 8.0   (索引: 101_11)

    B"16'h4040", // 6.0 * 0.5 = 3.0   (索引: 110_00)
    B"16'h40C0", // 6.0 * 1.0 = 6.0   (索引: 110_01)
    B"16'h4110", // 6.0 * 1.5 = 9.0   (索引: 110_10)
    B"16'h4140", // 6.0 * 2.0 = 12.0  (索引: 110_11)
  )

  // 计算结果的符号
  private val resultSign:Bool = signA ^ signB

  // 从LUT读取绝对值结果
  private val absResult:Bits = absValueLUT(lutAddress)

  // 符号处理：如果结果为负，对绝对值取反符号位
  private val resultWithSign: Bits = Mux(resultSign,
    (B"1" ## absResult(14 downto 0)), // 负数，设置符号位为1
    absResult // 正数，保持原符号位（应该是0）
  )

  // 最终结果：零输入直接返回0，否则返回带符号的结果
  io.result := Mux(resultIsZero, B(0, 16 bits), resultWithSign)
}