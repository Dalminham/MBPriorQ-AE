name := "LLM_Accelerator"

version := "0.1"

scalaVersion := "2.12.19"

val spinalVersion = "1.8.0"

libraryDependencies ++= Seq(
  "com.github.spinalhdl" % "spinalhdl-core_2.12" % spinalVersion,
  "com.github.spinalhdl" % "spinalhdl-lib_2.12" % spinalVersion,
  compilerPlugin("com.github.spinalhdl" % "spinalhdl-idsl-plugin_2.12" % spinalVersion)
)

// Required by SpinalHDL/Verilator simulations when running via sbt runMain
Compile / run / fork := true
