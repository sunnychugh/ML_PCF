# ML-PCF


## This repo is used to obtain results for a Journal paper "Machine learning approach for computing optical properties of a photonic crystal fiber"


* Multilayer perceptron has been implemented using PyTorch framework to compute various optical properties (effective index, effective mode area, dispersion, and confinement loss) of a photonic crsytal fiber (PCF). 

* Python3 packages given in start of file "pcf_modeSoln_pytorch_1.py" can be installed after installing [Anaconda version > 3](https://www.anaconda.com/distribution/) OR may be installed from "requirements.txt". You can try requirements.txt, bit it might create issue when installing "Torch" or other packages depending on the PC. I recommend to install individual packages, as required.  

* Data file "pcf_modeSoln_data_1.xlsx" is for training the model and "pcf_modeSoln_data_manual_1.xlsx" is for testing the model.

* One previously saved model weights file "checkpoint_5000.pth" is also provided. Main code file (pcf_modeSoln_pytorch_1.py) needs to be changed at respective position, if you donot want to use this already saved weights file.
