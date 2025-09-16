# Setup Instructions

This directory contains the problem setup, inputs files, and analysis tools (in the `util` subfolder)
for running hydrodynamics simulations of supernovae with a central energy source. To run one of the
models listed in the table below, compile with the listed additional compilation flags (by using
make -j <flag> or editing the makefile), and then run the executable with the listed inputs file:

| Simulation ID | Additional Compilation Flags | Inputs File |
| ------------- | ------------- | -------------|
| `1/4ms` |   | inputs.2d.1_4ms.16384 |
| `7/16ms` |   | inputs.2d.7_16ms.16384 |
| `1ms` |   | inputs.2d.1ms.16384 |
| `2ms` |   | inputs.2d.2ms.16384 |
| `3ms` |   | inputs.2d.3ms.16384 |
| `10ms` |   | inputs.2d.10ms.16384 |
| `1ms_lowB` |   | inputs.2d.1ms.lowB |
| `1ms_sin` | `USE_SIN_DEP=TRUE`  | inputs.2d.1ms.16384 |
| `1ms_eq` | `USE_EQ_DEP=TRUE` | inputs.2d.1ms.eq |
| `polar_5/3` | `USE_COS_DEP=TRUE`  | inputs.2d.polar_5_3 |
