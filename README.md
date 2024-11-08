# tttsa

[![License](https://img.shields.io/pypi/l/tttsa.svg?color=green)](https://github.com/McHaillet/tttsa/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tttsa.svg?color=green)](https://pypi.org/project/tttsa)
[![Python Version](https://img.shields.io/pypi/pyversions/tttsa.svg?color=green)](https://python.org)
[![CI](https://github.com/McHaillet/tttsa/actions/workflows/ci.yml/badge.svg)](https://github.com/McHaillet/tttsa/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/McHaillet/tttsa/branch/main/graph/badge.svg)](https://codecov.io/gh/McHaillet/tttsa)

Automated tilt-series alignment for cryo-ET.

## Under development

Not yet stable and the API might change.

## Give it a whirl?

Developer install into a conda environment (`conda create -n tttsa python=3.12`) 
after locally cloning the repository:

```
python -m pip install .[dev]
```

This will run the example. It will take some time to initially download the data from Zenodo.

```
cd examples
python usage.py
```

In `usage.py` you can also adjust the DEVICE to run on cpu (default is cuda:0).
