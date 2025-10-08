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
