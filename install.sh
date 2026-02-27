conda create -n lmao python=3.9.7
conda activate lmao
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install transformers==4.28.1 -c conda-forge
# conda install accelerate==0.17.1 -c conda-forge
conda install accelerate
conda install scikit-learn
conda install numpy==1.24.3
conda install wandb
conda install loralib
# pip install loralib
conda install debugpy
