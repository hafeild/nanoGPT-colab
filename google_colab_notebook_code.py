# Paste this into a Google Colab notebook cell and run it. If you want to make it a bit more
# modular, paste each chunk into its own cell.

# Install necessary packages.
!pip install torch numpy transformers datasets tiktoken wandb tqdm

# Clone the this repo; update if necessary (if re-running cell after a repo change).
!git clone https://github.com/hafeild/nanoGPT-colab.git
!cd nanoGPT-colab && git pull 

# Get and prepare the data for training.
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'nanoGPT-colab/data/google_colab_char')
import prepare
prepare.prepareGoogleColab()

# Train.
!cd nanoGPT-colab/ && python train.py config/train_google_colab_char.py \
    --device=cpu --compile=False --eval_iters=20 --log_interval=1 \
    --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 \
    --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0


# @title Text generation settings
# @markdown Specify each of the following before running this cell.

start_of_output = 'If I could have three wishes'  # @param {type: "string"}
temperature = 0.8 # @param {type: "slider", min: 0, max: 2, step: 0.05}
numberOfPassages = 29  # @param {type: "slider", max: 100, min: 1}


# Generate new text.
sys.path.insert(1, 'nanoGPT-colab')
import sample
sample.sample(out_dir='out-google-colab-char', device='cpu', start=start_of_output, num_samples=numberOfPassages, temperature=temperature)
