# idrink_backend


## Installation

- Install CUDA and CudNN:
```
https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html
```

- Create a new conda environment form the specified yaml:
```
conda env create -f env.yml
conda activate idrink_backend
```

## Running
```
python src/idrink_app.py --dir_root /home/arashsm79/Playground/iDrink_Demo --bids_root /home/arashsm79/amyproject --id_p 4a2 --id_s rest --task drink   
```

or use the launch file at `.vscode/launch.json`