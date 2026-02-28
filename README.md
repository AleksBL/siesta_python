# siesta_python
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://www.mozilla.org/en-US/MPL/2.0/)

Version 1.0

# Introduction
This is a code meant for easy handling of SIESTA, TranSIESTA and TBtrans calculations. This code allows to call these DFT programs directly in python, thereby allowing for better scripting using these programs. 

## Features 
- The SiP object allows for automatisation of many tasks, such as writing fdf files, rearranging geometries and finding particular indices to be passed to SIESTA. 
- wrapper functions around many features of SIESTA, TranSIESTA and sisl such as doping
- Handling of electrode and device objects and combining them inside the python script. 

## Installation
To install siesta_python, download the code as a zip file, unpack it and navigate to the siesta_python folder containing the setup.py file in a terminal. Now execute
```console
    python3 -m pip install -e .
```
and you will have an editable install of the code. 

## Contact 
Users can get in contact with the developer by submitting an issue on the Zandpack Github page. You also direct messages to aleksander.bl@proton.me.

