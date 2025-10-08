# BEAR (Beam-Search-Aware Regularization)

This is the official implementation of our **WWW 2026 Submission** paper:

> **BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models**

## Environment Setup

We provide the environment configuration in [`environment.yaml`](environment.yaml). You can create the environment using:

```bash
conda env create -f environment.yaml
conda activate bear
```

You will install the required packages and set up the environment `bear`. Note that you may need to install the [`genre`](https://github.com/facebookresearch/GENRE) package manually.


## Running the Code

We provide a sample script [`cmd/run.sh`](cmd/run.sh) to run the experiments. You need to modify the paths in the script first, including the code path and the LLM model path in [`cmd/run.sh`](cmd/run.sh) and the code path in [`cmd/run.py`](cmd/run.py). Then you can run the script:

```bash
tmux set mouse on
conda activate bear
cd /path/to/your/code

MODEL_PATH="/path/to/your/LLM/Llama-3.2-3B"
CUDA="0,1,2,3,4,5,6,7"

# -------------------------------------------------------------------
# EXAMPLE SCRIPT TO RUN EXPERIMENTS (TOY DATASET)
# -------------------------------------------------------------------

# LML
python cmd/run.py Toy --loss lml --model_path ${MODEL_PATH} --cuda ${CUDA} --num_epochs 10

# MSL
python cmd/run.py Toy --loss msl --model_path ${MODEL_PATH} --cuda ${CUDA} --num_epochs 10

# BEAR
python cmd/run.py Toy --loss bear --model_path ${MODEL_PATH} --cuda ${CUDA} --num_epochs 10 --tau 1.0 --topk_tau 2.75 --topk_weight 0.25
```

Here for our BEAR method, the `topk_tau` and `topk_weight` refer to the hyperparameter $\xi$ and $\lambda$ in our paper, respectively.
