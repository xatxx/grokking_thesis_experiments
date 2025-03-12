<br/>
This is the copy implementation of the paper <a href="https://arxiv.org/abs/2501.04697" target="_blank"><i>Grokking at the Edge of Numerical Stability</i></a> (arXiv: 2501.04697). 

The aim is to conduct experiments for the grokking phenomenon for a master thesis.
<br/>

## Replicating Our Results

To replicate the main figures from our paper, use the `run_main_experiments.sh` script. This script generates and logs all necessary metrics for the primary figures (Figures **1**, **2**, **4**, and **6**), and uses `cuda:0` by default. Once the relevant metrics have been saved, you can generate the figures from the paper using [paper_plots.ipynb](https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability/blob/main/paper_plots.ipynb).

### Usage

```bash
./run_main_experiments.sh
```
- **On a MacBook**: Use the CPU for running the experiment.  
- **On a system with an NVIDIA GPU**: You can run the experiment with CUDA.  

#### Configuration

You can modify the execution settings in the `run_main_experiments.sh` file to switch between CPU and CUDA.
## Requirements

1. **Install PyTorch:**
    Visit the [PyTorch Get Started](https://pytorch.org/get-started/locally/) page to choose the appropriate installation command for your system.

2. **Install Python Packages:**
   Install the remaining dependencies using `pip`:
   ```
   pip install pandas==2.2.3
   pip install matplotlib==3.10.0
   ```

## Citation

If you found this code or paper useful, please consider citing:

```shell
@article{prieto2025grokking,
  title={Grokking at the Edge of Numerical Stability},
  author={Prieto, Lucas and Barsbey, Melih and Mediano, Pedro and Birdal, Tolga},
  year = {2025},
  eprint={2501.04697},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
