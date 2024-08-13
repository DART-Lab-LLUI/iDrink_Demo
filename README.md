# iDrink_Demo


## Installation

### OpenSim

To install Opensim use the command: `conda install opensim-org::opensim`

### pytorch

To install Pytorch you can use: `conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia`

If you tried to install using conda and it fails, you can use: `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121`

The conda isntallation needs to be performed to install CUDA dependencies, that are not added using pip.

### onnxruntime-gpu
To install the correct onnxruntime-gpu you can use one of the follwing commands depending on your OS.

**Win**: `https://aiinfra.pkgs.visualstudio.com/PublicPackages/_apis/packaging/feeds/9387c3aa-d9ad-4513-968c-383f6f7f53b8/pypi/packages/onnxruntime-gpu/versions/1.18/onnxruntime_gpu-1.18.0-cp311-cp311-win_amd64.whl/content`

**Linux**: `https://aiinfra.pkgs.visualstudio.com/PublicPackages/_apis/packaging/feeds/9387c3aa-d9ad-4513-968c-383f6f7f53b8/pypi/packages/onnxruntime-gpu/versions/1.18/onnxruntime_gpu-1.18.0-cp311-cp311-manylinux_2_28_x86_64.whl/content`

The Linux Version has not yet been tested.

Alternatively you can donwload the file from
`https://aiinfra.visualstudio.com/PublicPackages/_artifacts/feed/onnxruntime-cuda-12/PyPI/onnxruntime-gpu/overview/1.18.0`
and install it loccally
