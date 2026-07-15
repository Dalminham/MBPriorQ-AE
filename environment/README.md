# Reproduction Environments

The software and hardware workflows are intentionally separated. A reviewer
who only runs the Functional software smoke does not need Java or Verilator.

## Software

Create the Python environment and install the local package:

```bash
conda env create -f environment/software.yml
conda activate mbpriorq-ae
python -m pip install -e software
```

The recorded results were produced with Python 3.12.12, PyTorch 2.10.0+cu129,
Transformers 4.57.6, Datasets 4.7.0, Safetensors 0.7.0, and NumPy 2.2.6. The
environment file pins the public package versions; the CUDA-enabled PyTorch
wheel may be selected according to the reviewer's driver using the official
PyTorch installation instructions.

## Open Hardware Simulation

Create and activate the hardware environment:

```bash
conda env create -f environment/hardware.yml
conda activate mbpriorq-ae-hw
```

Install sbt 1.10.2 with Coursier if it is not already available:

```bash
curl -fL https://github.com/coursier/launchers/raw/master/cs-x86_64-pc-linux.gz \
  | gzip -d > /tmp/cs
chmod +x /tmp/cs
/tmp/cs install sbt:1.10.2
```

The validated host used OpenJDK 17 for the hardware environment, sbt 1.10.2,
Scala 2.12.19, SpinalHDL 1.8.0, GNU Make 4.4.1, GCC/G++ 15.2.0, and Verilator
4.034. Java 17 is recommended because it avoids compatibility ambiguity with
the older SpinalHDL release.
