# LMAO

Three Forward, One Backward: Memory-Efficient Full-Rank Fine-Tuning of Large Models via Extra Forward Passes

## Abstract

Fine-tuning large language models (LLMs) has achieved significant success in downstream tasks.
However, as the model size continues to grow, traditional fine-tuning methods have become increasingly impractical due to their high computational and memory costs.
This has motivated researchers to explore parameter-efficient and memory-friendly fine-tuning strategies to enable scalable approaches, with Low-Rank Adaptation (LoRA) standing out as a representative work.
However, the LoRA update is restricted to a low-rank subspace, which results in suboptimal performance compared to the full-parameter update.
Recent research has also explored memory-efficient fine-tuning LLMs using just forward passes while suffer from high variance in gradient estimation and low convergence speed.
To address the issues above, we propose a new alternating optimization framework called LMAO (**L**ow-rank and **M**emory-efficient Zeroth-Order **A**lternating **O**ptimization), which combines the advantages of LoRA and MeZO.
This method alternately updates the low-rank components and zeroth-order directions during training.
By performing three forward propagations and one backward propagation, each update is full-rank, thereby reducing feature loss and enabling efficient fine-tuning under strict memory constraints.
We provide theoretical guarantees on the convergence and convergence rate of this method.
Empirical results demonstrate that, in experiments on multiple models (e.g., OPT, RoBERTa-large), LMAO achieves performance comparable to first-order methods.
This presents a practical and scalable solution for fine-tuning large-scale models.

## Usage

### build

```shell
sudo apt update && sudo apt install screen tree vim git htop gcc g++ make cmake colordiff python-is-python3 jq -y
# sudo apt upgrade
# sudo apt autoremove

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all

conda update -n base -c defaults conda

conda create -y -n lmao python=3.9.7
conda activate lmao
conda install pytorch==2.1.2 torchvision torchaudio numpy scikit-learn tqdm scipy pandas pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install transformers==4.28.1 loralib accelerate
pip install datasets wandb

# vim ~/.bashrc
```

```shell
alias ls="ls --color=auto"
alias la="ls --color=auto -al"
alias l="ls --color=auto -ahlF"
alias diff='colordiff'
alias grep='grep --color=auto'
alias egrep='egrep --colour=auto'
alias fgrep='fgrep --colour=auto'
alias dua="du -sh *"
alias vi='vim'
alias py="python3"

conda deactivate
conda activate lmao
cd ~/lmao/large_models
```

```shell
git clone https://github.com/workelaina/LMAO.git

cd ~/lmao/medium_models/data
bash download_dataset.sh
cp -r original ../../large_models/datasets
cd ..
bash tools/generate_k_shot_data.sh

wandb login
# https://wandb.ai/authorize
```

### run

#### Large

```shell
# cd ~/lmao/large_models
byobu

# USE_GPU=0 SEED=13 MODEL=facebook/opt-1.3b LR=1e-5 bash ft_all.sh
# USE_GPU=1 SEED=42 MODEL=facebook/opt-1.3b LR=1e-5 bash ft_all.sh
# USE_GPU=2 SEED=100 MODEL=facebook/opt-1.3b LR=1e-5 bash ft_all.sh
# USE_GPU=3 SEED=3407 MODEL=facebook/opt-1.3b LR=1e-5 bash ft_all.sh

# USE_GPU=0,1 SEED=13 MODE=lora bash lora_lr.sh
# USE_GPU=2,3 SEED=42 MODE=lora bash lora_lr.sh
# USE_GPU=4,5 SEED=3407 MODE=lora bash lora_lr.sh
# USE_GPU=6,7 SEED=100 MODE=lora bash lora_lr.sh

# USE_GPU=0,1 SEED=13 MODE=lora EPS=5e-3 bash mix_lr.sh
# USE_GPU=2,3 SEED=42 MODE=lora EPS=5e-3 bash mix_lr.sh
# USE_GPU=4,5 SEED=3407 MODE=lora EPS=5e-3 bash mix_lr.sh
# USE_GPU=6,7 SEED=100 MODE=lora EPS=5e-3 bash mix_lr.sh

# USE_GPU=0,1 SEED=13 MODE=lora LR=1e-6 LR_base=1e-7 EPS=5e-3 bash mix_all_13.sh
# USE_GPU=2,3 SEED=42 MODE=lora LR=1e-6 LR_base=1e-7 EPS=5e-3 bash mix_all_13.sh
# USE_GPU=4,5 SEED=3407 MODE=lora LR=1e-6 LR_base=1e-7 EPS=5e-3 bash mix_all_13.sh
# USE_GPU=6,7 SEED=100 MODE=lora LR=1e-6 LR_base=1e-7 EPS=5e-3 bash mix_all_13.sh

# USE_GPU=0 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_6.7_onegpu.sh

# USE_GPU=0 SEED=13 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_13.sh
# USE_GPU=1 SEED=21 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_13.sh
# USE_GPU=2 SEED=42 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_13.sh
# USE_GPU=3 SEED=87 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_13.sh
# USE_GPU=4 SEED=100 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_13.sh
# USE_GPU=5 SEED=0 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_13.sh
# USE_GPU=6 SEED=4242 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_13.sh
# USE_GPU=7 SEED=3407 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_13.sh

# USE_GPU=0 SEED=13 MODE=lora LR=1e-5 bash lora_all_13.sh
# USE_GPU=1 SEED=21 MODE=lora LR=1e-5 bash lora_all_13.sh
# USE_GPU=2 SEED=42 MODE=lora LR=1e-5 bash lora_all_13.sh
# USE_GPU=3 SEED=87 MODE=lora LR=1e-5 bash lora_all_13.sh
# USE_GPU=4 SEED=100 MODE=lora LR=1e-5 bash lora_all_13.sh
# USE_GPU=5 SEED=0 MODE=lora LR=1e-5 bash lora_all_13.sh
# USE_GPU=6 SEED=4242 MODE=lora LR=1e-5 bash lora_all_13.sh
# USE_GPU=7 SEED=3407 MODE=lora LR=1e-5 bash lora_all_13.sh

# USE_GPU=0 SEED=13 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_13.sh
# USE_GPU=1 SEED=21 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_13.sh
# USE_GPU=2 SEED=42 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_13.sh
# USE_GPU=3 SEED=87 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_13.sh
# USE_GPU=4 SEED=100 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_13.sh
# USE_GPU=5 SEED=0 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_13.sh
# USE_GPU=6 SEED=4242 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_13.sh
# USE_GPU=7 SEED=3407 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_13.sh

# USE_GPU=0 SEED=13 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_6.7.sh
# USE_GPU=1 SEED=21 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_6.7.sh
# USE_GPU=2 SEED=42 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_6.7.sh
# USE_GPU=3 SEED=87 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_6.7.sh
# USE_GPU=4 SEED=100 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_6.7.sh
# USE_GPU=5 SEED=0 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_6.7.sh
# USE_GPU=6 SEED=4242 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_6.7.sh
# USE_GPU=7 SEED=3407 MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_all_6.7.sh

# USE_GPU=0 SEED=13 MODE=lora LR=1e-5 bash lora_all_6.7.sh
# USE_GPU=1 SEED=21 MODE=lora LR=1e-5 bash lora_all_6.7.sh
# USE_GPU=2 SEED=42 MODE=lora LR=1e-5 bash lora_all_6.7.sh
# USE_GPU=3 SEED=87 MODE=lora LR=1e-5 bash lora_all_6.7.sh
# USE_GPU=4 SEED=100 MODE=lora LR=1e-5 bash lora_all_6.7.sh
# USE_GPU=5 SEED=0 MODE=lora LR=1e-5 bash lora_all_6.7.sh
# USE_GPU=6 SEED=4242 MODE=lora LR=1e-5 bash lora_all_6.7.sh
# USE_GPU=7 SEED=3407 MODE=lora LR=1e-5 bash lora_all_6.7.sh

# USE_GPU=0 SEED=13 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_6.7.sh
# USE_GPU=1 SEED=21 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_6.7.sh
# USE_GPU=2 SEED=42 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_6.7.sh
# USE_GPU=3 SEED=87 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_6.7.sh
# USE_GPU=4 SEED=100 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_6.7.sh
# USE_GPU=5 SEED=0 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_6.7.sh
# USE_GPU=6 SEED=4242 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_6.7.sh
# USE_GPU=7 SEED=3407 MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all_6.7.sh

# USE_GPU=0 SEED=13 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_miss.sh
# USE_GPU=1 SEED=21 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_miss.sh
# USE_GPU=2 SEED=42 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_miss.sh
# USE_GPU=3 SEED=87 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_miss.sh
# USE_GPU=4 SEED=100 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_miss.sh
# USE_GPU=5 SEED=0 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_miss.sh
# USE_GPU=6 SEED=4242 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_miss.sh
# USE_GPU=7 SEED=3407 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=5e-3 bash mix_miss.sh

# TASK=MultiRC MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_mem.sh
# TASK=MultiRC MODE=ft LR=1e-7 EPS=1e-3 bash mezo_mem.sh
# TASK=MultiRC MODE=lora LR=1e-5 bash lora_mem.sh
# TASK=MultiRC MODE=prefix LR=1e-2 bash ft_mem.sh
# TASK=MultiRC MODE=ft LR=1e-5 bash ft_mem.sh

# TASK=WSC MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_mem.sh
# TASK=WSC MODE=ft LR=1e-7 EPS=1e-3 bash mezo_mem.sh
# TASK=WSC MODE=lora LR=1e-5 bash lora_mem.sh
# TASK=WSC MODE=ft LR=1e-5 bash ft_mem.sh

# TASK=WIC MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_mem.sh
# TASK=WIC MODE=ft LR=1e-7 EPS=1e-3 bash mezo_mem.sh
# TASK=WIC MODE=lora LR=1e-5 bash lora_mem.sh
# TASK=WIC MODE=ft LR=1e-5 bash ft_mem.sh

# TASK=Copa MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_mem.sh
# TASK=Copa MODE=ft LR=1e-7 EPS=1e-3 bash mezo_mem.sh
# TASK=Copa MODE=lora LR=1e-5 bash lora_mem.sh
# TASK=Copa MODE=ft LR=1e-5 bash ft_mem.sh

# TASK=ReCoRD MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_mem.sh
# TASK=ReCoRD MODE=ft LR=1e-7 EPS=1e-3 bash mezo_mem.sh
# TASK=ReCoRD MODE=lora LR=1e-5 bash lora_mem.sh
# TASK=ReCoRD MODE=ft LR=1e-5 bash ft_mem.sh

# MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_miss.sh

# USE_GPU=0 SEED=13 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=1 SEED=21 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=2 SEED=42 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=3 SEED=87 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=4 SEED=100 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=5 SEED=0 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=6 SEED=4242 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=7 SEED=3407 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh

# USE_GPU=0 SEED=13 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=1 SEED=21 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=2 SEED=42 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=3 SEED=87 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=4 SEED=100 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=5 SEED=0 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=6 SEED=4242 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=7 SEED=3407 MODEL=facebook/opt-1.3b MODE=lora LR=1e-5 bash lora_all.sh

# USE_GPU=0 SEED=13 MODEL=facebook/opt-1.3b MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all.sh
# USE_GPU=1 SEED=21 MODEL=facebook/opt-1.3b MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all.sh
# USE_GPU=2 SEED=42 MODEL=facebook/opt-1.3b MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all.sh
# USE_GPU=3 SEED=87 MODEL=facebook/opt-1.3b MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all.sh
# USE_GPU=4 SEED=100 MODEL=facebook/opt-1.3b MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all.sh
# USE_GPU=5 SEED=0 MODEL=facebook/opt-1.3b MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all.sh
# USE_GPU=6 SEED=4242 MODEL=facebook/opt-1.3b MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all.sh
# USE_GPU=7 SEED=3407 MODEL=facebook/opt-1.3b MODE=ft LR=1e-7 EPS=1e-3 bash mezo_all.sh

# USE_GPU=0 SEED=13 MODEL=facebook/opt-1.3b bash icl_all.sh
# USE_GPU=1 SEED=21 MODEL=facebook/opt-1.3b bash icl_all.sh
# USE_GPU=2 SEED=42 MODEL=facebook/opt-1.3b bash icl_all.sh
# USE_GPU=3 SEED=87 MODEL=facebook/opt-1.3b bash icl_all.sh
# USE_GPU=4 SEED=100 MODEL=facebook/opt-1.3b bash icl_all.sh
# USE_GPU=5 SEED=0 MODEL=facebook/opt-1.3b bash icl_all.sh
# USE_GPU=6 SEED=4242 MODEL=facebook/opt-1.3b bash icl_all.sh
# USE_GPU=7 SEED=3407 MODEL=facebook/opt-1.3b bash icl_all.sh

# USE_GPU=0 SEED=13 MODEL=facebook/opt-1.3b bash zeroshot_all.sh
# USE_GPU=1 SEED=21 MODEL=facebook/opt-1.3b bash zeroshot_all.sh
# USE_GPU=2 SEED=42 MODEL=facebook/opt-1.3b bash zeroshot_all.sh
# USE_GPU=3 SEED=87 MODEL=facebook/opt-1.3b bash zeroshot_all.sh
# USE_GPU=4 SEED=100 MODEL=facebook/opt-1.3b bash zeroshot_all.sh
# USE_GPU=5 SEED=0 MODEL=facebook/opt-1.3b bash zeroshot_all.sh
# USE_GPU=6 SEED=4242 MODEL=facebook/opt-1.3b bash zeroshot_all.sh
# USE_GPU=7 SEED=3407 MODEL=facebook/opt-1.3b bash zeroshot_all.sh

# USE_GPU=0 SEED=13 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=1 SEED=21 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=2 SEED=42 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=3 SEED=87 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=4 SEED=100 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=5 SEED=0 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=6 SEED=4242 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh
# USE_GPU=7 SEED=3407 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix_all.sh

# USE_GPU=0 SEED=13 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=1 SEED=21 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=2 SEED=42 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=3 SEED=87 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=4 SEED=100 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=5 SEED=0 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=6 SEED=4242 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 bash lora_all.sh
# USE_GPU=7 SEED=3407 MODEL=facebook/opt-2.7b MODE=lora LR=1e-5 bash lora_all.sh

# USE_GPU=0 SEED=13 MODEL=facebook/opt-2.7b bash icl_all.sh
# USE_GPU=1 SEED=21 MODEL=facebook/opt-2.7b bash icl_all.sh
# USE_GPU=2 SEED=42 MODEL=facebook/opt-2.7b bash icl_all.sh
# USE_GPU=3 SEED=87 MODEL=facebook/opt-2.7b bash icl_all.sh
# USE_GPU=4 SEED=100 MODEL=facebook/opt-2.7b bash icl_all.sh
# USE_GPU=5 SEED=0 MODEL=facebook/opt-2.7b bash icl_all.sh
# USE_GPU=6 SEED=4242 MODEL=facebook/opt-2.7b bash icl_all.sh
# USE_GPU=7 SEED=3407 MODEL=facebook/opt-2.7b bash icl_all.sh

# USE_GPU=0 SEED=13 MODEL=facebook/opt-2.7b bash zeroshot_all.sh
# USE_GPU=1 SEED=21 MODEL=facebook/opt-2.7b bash zeroshot_all.sh
# USE_GPU=2 SEED=42 MODEL=facebook/opt-2.7b bash zeroshot_all.sh
# USE_GPU=3 SEED=87 MODEL=facebook/opt-2.7b bash zeroshot_all.sh
# USE_GPU=4 SEED=100 MODEL=facebook/opt-2.7b bash zeroshot_all.sh
# USE_GPU=5 SEED=0 MODEL=facebook/opt-2.7b bash zeroshot_all.sh
# USE_GPU=6 SEED=4242 MODEL=facebook/opt-2.7b bash zeroshot_all.sh
# USE_GPU=7 SEED=3407 MODEL=facebook/opt-2.7b bash zeroshot_all.sh
```

#### Medium

```shell
# cd lmao/medium_models
# byobu

# USE_GPU=0 SEED=13 K=512 BS=64 EPS=1e-3 bash mix_all.sh --apply_lora --lora_r 8 --lora_alpha 16
# USE_GPU=1 SEED=21 K=512 BS=64 EPS=1e-3 bash mix_all.sh --apply_lora --lora_r 8 --lora_alpha 16
# USE_GPU=2 SEED=42 K=512 BS=64 EPS=1e-3 bash mix_all.sh --apply_lora --lora_r 8 --lora_alpha 16
# USE_GPU=3 SEED=87 K=512 BS=64 EPS=1e-3 bash mix_all.sh --apply_lora --lora_r 8 --lora_alpha 16
# USE_GPU=4 SEED=100 K=512 BS=64 EPS=1e-3 bash mix_all.sh --apply_lora --lora_r 8 --lora_alpha 16
# USE_GPU=5 SEED=0 K=512 BS=64 EPS=1e-3 bash mix_all.sh --apply_lora --lora_r 8 --lora_alpha 16
# USE_GPU=6 SEED=4242 K=512 BS=64 EPS=1e-3 bash mix_all.sh --apply_lora --lora_r 8 --lora_alpha 16
# USE_GPU=7 SEED=3407 K=512 BS=64 EPS=1e-3 bash mix_all.sh --apply_lora --lora_r 8 --lora_alpha 16
```

### other knowledge

### err

#### 1

```txt
    model = torch.nn.parallel.DistributedDataParallel(
RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.
```

Check `transformers`' version: `transformers==4.28.1`.

#### 2

```txt
RuntimeError: Numpy is not available
```

Check Python version: `python==3.9.7`.

`python==3.9.x` is not enough.
