v2.0.2
======
- readthedocs.yaml added
- docs/requirements.txt added

v2.0.1
======
- docs added
- demo notebooks added and working
- default config location moved to location in package to streamline setup
- all exchange only experiments are set up to use the crosstalk compensation firmware
- added pyproject.toml for configuration of dev tools
- added general environment.yml for environment creation without all the packages for developers

v2.0.0
======
- utilizes tprocv2 firmware and API from qick
- new experiment config file structure and parameter management using pydantic models
- code reorganization - core and models modules
- SpinqickData objects implemented for data handling and saving in most experiments
- new exchange-only experiment including calibration routines, nosc and free induction decay experiments
- changed version to v2, to avoid confusion

v0.0.3
======
- filled in more of the docstrings
- fixing general formatting errors
- removed some unused variables

v0.0.2
======
- renamed variables that weren't in snake_case
- renamed dict keys so they are in snake_case

v0.0.1
======
- Added a more comprehensive gateset for ALLXY phase control demo
- Split the idle cell scan functionality and added a new class to handle IdleCell experiment setup in the psb_setup_program.py
- Added a routine for electron temperature sweeps (and more generally 1D cuts in the PvP) in the electrostatic tuning set of functionality (based on TODO listed in initial version)
- Split and added some low-level functionality to conduct voltage compensation
- Added fields for LD qubit RF drive parameters and a new HemtBias model for parameter tracking
- general bug fixes for file saving, configs etc
- debugging of the PSB spin averager

v0.0.0
======
- First version
