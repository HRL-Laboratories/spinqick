.. spinQICK documentation master file, created by
   sphinx-quickstart on Thu Nov  6 10:21:22 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

spinQICK documentation
======================

Welcome to the spinQICK docs!

spinQICK is an extension of the `Quantum Instrumentation Control Kit <https://github.com/openquantumhardware/qick>`__ (QICK) designed to control
electrostatically confined solid-state spin-qubits! SpinQICK enables researchers to use low-cost off-the-shelf Xilinx Radio Frequency System-on-Chip (RFSoC)
Field Programmable Gate Arrays (FPGAs) to rapidly develop novel application specific experimental hardware and software for controlling spin-qubits.

The repository contains code for spinQICK V2, which updates spinQICK for compatibility with the updated `tProcV2 <https://github.com/meeg/qick_demos_sho/blob/main/tprocv2/qick_processor_TRM.pdf>`__
core processor from QICK. This expands the number of DACs from 8 to 16, allowing for six single-qubits or two exchange-only qubits.
In addition to this, the latest firmware allows for upto five digitization channels.


Check out our March Meeting talk `here <https://youtu.be/lcLnXZpfxwc?si=JAEB6xYO9OmrOPjL>`__

And our github `here <https://github.com/HRL-Laboratories/spinqick>`__

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   getting_started
   qickquack
