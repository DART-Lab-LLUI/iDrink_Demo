# iDrink_Demo


## Installation

### OpenSim

To install Opensim use the command: `conda install opensim-org::opensim`

### pytorch

To install Pytorch you can use: `conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia`

If you tried to install using conda and it fails, you can use: `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121`

The conda isntallation needs to be performed to install CUDA dependencies, that are not added using pip.

### onnxruntime-gpu

[source](https://onnxruntime.ai/docs/install/)

```
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```
