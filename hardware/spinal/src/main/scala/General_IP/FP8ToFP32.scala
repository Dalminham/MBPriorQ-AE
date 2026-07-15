package General_IP
import spinal.core._

class FP8ToFP32 extends Component {
    val io = new Bundle {
        val fp8_in = in Bits(8 bits) // S1E4M3
        val fp32_out = out Bits(32 bits) // S1E8M23
    }
    // Sign
    private val sign: Bool = io.fp8_in(7)
    // Exponent
    private val fp8_exponent: UInt = io.fp8_in(6 downto 3).asUInt
    // Mantissa
    private val mantissa = io.fp8_in(2 downto 0)

    // 特殊值检测
    // FP8 E4M3格式规则：
    // - 指数=0且尾数=0: 零
    // - 指数=0且尾数!=0: 非规格化数
    // - 指数=15且尾数=7: 无穷大（符号决定正负）
    // - 指数=15且尾数!=7: 仍然是有效数字（不遵循IEEE 754标准）
    private val isZero: Bool = (fp8_exponent === 0) && (mantissa === 0)
    private val isInf: Bool = (fp8_exponent === 15) && (mantissa === 7)
    private val isDenorm: Bool = (fp8_exponent === 0) && (mantissa =/= 0)

    // FP32指数计算
    // 对于非规格化数（指数=0，尾数!=0），应该使用指数120（0+120）
    // 对于规格化数（指数!=0且指数!=15，或指数=15但尾数!=7），使用指数+120
    // 对于零（指数=0，尾数=0），指数为0
    // 对于无穷大（指数=15，尾数=7），指数为255
    private val unnormalized_component: UInt = fp8_exponent.resize(8)
    private val normalized_component: UInt = Mux(
        isZero,
        U(0, 8 bits),  // 零：指数为0
        Mux(
            isInf,
            U(255, 8 bits),  // 无穷大：指数为255
            Mux(
                isDenorm,
                U(120, 8 bits),  // 非规格化数：指数为120（0+120）
                unnormalized_component + 120  // 规格化数：指数+120（包括指数=15且尾数!=7的情况）
            )
        )
    )

    // FP32尾数
    // 对于无穷大，尾数必须为0（IEEE 754标准）
    // 对于其他情况，将3位尾数放在高3位，低位补0
    private val fp32_mantissa = Mux(
        isInf,
        B"23'h000000",  // 无穷大：尾数为0
        mantissa##B"20'h00000"  // 其他情况：尾数正常处理
    )

    // 最终输出
    private val fp32_out = sign ## normalized_component ## fp32_mantissa
    io.fp32_out := fp32_out
}
