# NHR Summer School 2023 - Data-Parallel Neural Networks in `PyTorch`
This repository contains the materials for the workshop on "Data-Parallel Neural Networks in `PyTorch`" at the NHR Summer School 2023. 
You can find the complete program [here](https://moodle.nhr.fau.de/course/view.php?id=117).

### Program for Thursday, June 15, 2023

| W H E N           | W H A T                                                 |
| :-----------------| :------------------------------------------------------ |
| **09:00 - 10:15** | **Introduction to Neural Networks**                     |  
|                   | Backpropagation and Stochastic Gradient Descent (SGD)   |  
|                   | Layer Architectures                                     |  
|                   | Training a Neural Network                               |  
| 10:15 - 10:30     | *It's coffee o'clock!*                                  |
| **10:30 - 12:00** | **Hands-on Session: Neural Networks with `PyTorch`**    |  
|                   | Exercise notebook in this repository: `notebook/lueckentext_serial.ipynb`  |
|                   | Solutions: `notebook/solution_serial.ipynb`     |
| 12:00 - 13:00     | *Enjoy your lunch break!*                               |  
| **13:00 - 14:15** | **Data-Parallel Neural Networks**                       |  
|                   | Parallelization Strategies for Neural Networks          |  
|                   | Distributed SGD                                         |  
|                   | IID and Large Minibatch Effects                         |  
|   14:15 - 14:30   | *It's coffee o'clock!*                                  |
| **14:30 - 16:00** | **Hands-on Session: `PyTorch DistributedDataParallel`** |
|                   | Exercise notebook in this repository: `notebook/lueckentext_ddp.ipynb`  |
|                   | Solutions: `notebook/solution_ddp.ipynb`  |

## Hands-on Session: Neural Networks with `PyTorch`
In the first hands-on session, you will learn how to train a neural network in `PyTorch`. This exercise serves as a prerequisite for training the network in a data-parallel fashion later on. 
As a first step, you need to log into the PC2 JupyterHub to access a compute node of the HPC system Noctua2 interactively: [https://jh.pc2.uni-paderborn.de](https://jh.pc2.uni-paderborn.de)  
Choose the Jupyter preset "NHR Courseweek Data Parallel Neural Networks (32 cores, 1 GPU)".  
Alternatively, enter the following in the "Expert" field:  
`#SBATCH --job-name=alexnet`  
`#SBATCH --partition=gpu`  
`#SBATCH --account=hpc-prf-nhrgs`  
`#SBATCH --time=02:00:00`  
`#SBATCH --mem=10G`  
`#SBATCH --cpus-per-task=32`  
`#SBATCH --gres=gpu:a100:1`  

Once logged in, please load the following module: `JupyterKernel-Python/3.10.4-torchvision-0.13.1-CUDA-11.7.0`  
To start working on the exercises, clone this repository into your personal folder and open the notebook `notebook/lueckentext_serial.ipynb`.  
You can find the corresponding solution in the notebook `notebook/solution_serial.ipynb`.

## Hands-on Session: `Pytorch DistributedDataParallel`
In this hands-on tutorial, you will learn how to train a data-parallel neural network using `PyTorch`'s `DistributedDataParallel` package. 
You can find the corresponding exercises and solutions in `notebook/lueckentext_ddp.ipynb` and `notebook/solution_ddp.ipynb`, respectively. 
You can work on and develop your code in the `lueckentext_ddp.ipynb` notebook using the same setup as before. However, truly parallel runs are inconvenient in Jupyter notebooks. This is why you need to create Python scripts from the code snippets as well as a job script to test and actually run your code in parallel as a batch job on Noctua2. You can find step-by-step instructions for how to do this in the exercises notebook. The actual `Python` and SLURM job scripts are in `py/` and `sh/`, respectively.
