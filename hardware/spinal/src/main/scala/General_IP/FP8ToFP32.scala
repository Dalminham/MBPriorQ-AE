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

    // E4M3 reserves exponent 15, mantissa 7 for infinity. Other exponent-15
    // encodings remain finite values.
    private val isZero: Bool = (fp8_exponent === 0) && (mantissa === 0)
    private val isInf: Bool = (fp8_exponent === 15) && (mantissa === 7)
    private val isDenorm: Bool = (fp8_exponent === 0) && (mantissa =/= 0)

    // Re-bias the E4M3 exponent from 7 to the FP32 bias of 127.
    private val unnormalized_component: UInt = fp8_exponent.resize(8)
    private val normalized_component: UInt = Mux(
        isZero,
        U(0, 8 bits),
        Mux(
            isInf,
            U(255, 8 bits),
            Mux(
                isDenorm,
                U(120, 8 bits),
                unnormalized_component + 120
            )
        )
    )

    // Place the three E4M3 mantissa bits at the top of the FP32 mantissa.
    private val fp32_mantissa = Mux(
        isInf,
        B"23'h000000",
        mantissa##B"20'h00000"
    )

    private val fp32_out = sign ## normalized_component ## fp32_mantissa
    io.fp32_out := fp32_out
}
