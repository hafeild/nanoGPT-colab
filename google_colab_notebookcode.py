# Paste this into a Google Colab notebook cell and run it. If you want to make it a bit more
# modular, paste each chunk into its own cell.

# Install necessary packages.
!pip install torch numpy transformers datasets tiktoken wandb tqdm

# Clone the this repo.
!git clone https://github.com/hafeild/nanoGPT-colab.git

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

# Generate new text.
!cd nanoGPT-colab/ && python sample.py --out_dir=out-google-colab-char --device=cpu