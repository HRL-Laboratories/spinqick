# spinQICK

<p align="center">
 <img src="graphics/SpinQICK_logo.svg" alt="spinQICK logo" width=40% height=auto>
</p>

## Description
Welcome to spinQICK, an open-source extension to the [Quantum Instrumentation Control Kit](https://github.com/openquantumhardware/qick) (QICK) designed to control electrostatically confined solid-state spin-qubits! SpinQICK enables researchers to use low-cost off-the-shelf Xilinx Radio Frequency System-on-Chip (RFSoC) Field Programmable Gate Arrays (FPGAs) to rapidly develop novel application specific experimental hardware and software for controlling spin-qubits.

This package utilizes both the standard QICK API and modified low-level QICK assembly to implement standard measurement and control methods that are unique to electrostatically confined spin-qubit systems. These methods currently accomodate single-spin (Loss-DiVincenzo) qubits and include charge-stability and electrostatic tune-up, initialization and parity readout, single-spin coherent control and characterization (T1, T2*, T2-Echo, Ramsey, All-XY), exchange calibration, and two-qubit gates. In addition to these facilities, this package also provides features for parameter-management, plotting, and demonstrations to help get started with spinQICK.

## Hardware
- [ZCU216 RFSoC](https://www.xilinx.com/products/boards-and-kits/zcu216.html)
- [LMH5401](https://www.ti.com/tool/LMH5401EVM) DC Coupled Differential Amplifiers
- [PMOD level shifter](https://digilent.com/shop/pmod-lvlshft-logic-level-shifter/?srsltid=AfmBOoqZodUKJkK6xvxAk7vgOS6NISjlLeNHoWDSeB-TueM1wp54cUVR) to buffer the trigger pulses for improved stability
- Precision DC bias (QDevil QDAC, Basel LNHR, etc)
- Support hardware including DC supplies for amplifiers, control computer, etc.

## Software
- Python 3.10, specific packages can be found in the _dev_environment.yml_ file
- [QICK](https://github.com/openquantumhardware/qick) version 0.2.260 (later versions will be checked for compatibility)
- Pynq 2.7 bootloader, QICK standard boot can be found [here](https://github.com/sarafs1926/ZCU216-PYNQ/issues/1)
- Pyro4 must also be installed on the ZCU216 board, see [QICK demo](https://github.com/openquantumhardware/qick/blob/main/pyro4/00_nameserver.ipynb) for information on running Pyro server on the RFSoC

## Package Structure
All code for the API can be found in `src/spinqick` with demo notebooks illustrating the use of the API and resources for getting started found in `demo_notebooks`. The spinQICK API is organized into four folders as follows.

### Experiments
High-level experimental code, including the general `dot_experiment` class inherited by subsequent experiments. Experiments are organized into exchange only (`eo_single_qubit`) and ld single (`ld_single_qubit`) and two qubit (`ld_2_qubit`), electrostatic tune-up (`tune_electrostatic`), and calibration experiments (`system_calibrations`). The readout functionality for Pauli-spin blockade can be found in `psb_setup`, and routines for taking noise using QICK's DSO functionality can be found in `measure_noise`.

### Helper functionality
Inside the `helper_function` directory is functionality to help manage data files (`file_manager`), hardware maps (`hardware_manager`), and plotting (`plot_tools`), as well as routines for generating common dac pulse shapes (`dac_pulses`).

### Models
The models directory contains PyDantic models for various experimental config types. Using PyDantic allows for efficient type checking of experimental parameters among many other features. To see how to use these models and experimental configs see `demo_notebooks\00_getting_started.ipynb`.

### QICK code
The `qick_code` directory represents the collection of underlying QICK API code used by experiments. This code is written in the native QICK API or in QICK assembly, and represents the custom functions required for spin-qubit experiments.

### Settings
In addition to the above four directories, the `settings.py` file contains a user-defined pydantic object with directories for data and config files. Default directories are all located in `C:/Data/QICK/` path.

## Updates
SpinQICK is under active development. As such there may be changes that break existing code built on this package. SpinQICK follows [Semantic Versioning](https://semver.org/), with patches and minor versions being backwards compatible, and major version revisions representing changes that break existing API implementation.

### Near-term Updates
Current implementation of the spinQICK API is built on the QICK [tProc 1 V4](https://github.com/openquantumhardware/qick/blob/main/firmware/tProcessor_64_and_Signal_Generator_V4.pdf) instruction set. Later revisions will move to [tProc 2](https://github.com/meeg/qick_demos_sho/blob/main/tprocv2/qick_processor_TRM.pdf), released in QICK version 0.2.285.

## Authors
Abigail Wessels<sup>[1](#HRL)</sup>, Andrew E. Oriani<sup>[1](#HRL)</sup>

<a name="HRL">1</a>: HRL Laboratories, LLC, 3011 Malibu Canyon Road, Malibu, California 90265, USA

## Acknowledgements
This software was created by HRL under Army Research Office (ARO) Award Number: W911NF‐24‐1‐0020. ARO, as the Federal awarding agency, reserves a royalty‐ free, nonexclusive and irrevocable right to reproduce, publish, or otherwise use this software for Federal purposes, and to authorize others to do so in accordance with 2 CFR 200.315(b). We would like to thank Sho Uemura, Sara Sussman and the QICK team on their continued support in developing this package. We would like to acknowledge the help of Joe Broz, Edwin Acuna, and support of Jason Petta for the creation of this package.
