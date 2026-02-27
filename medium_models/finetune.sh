#!/bin/bash

# CUDA_VISIBLE_DEVICES=0

# Adam fine-tuning
# TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-5 MODEL=roberta-large bash finetune.sh
# TASK=SNLI K=16 SEED=42 BS=8 LR=1e-5 MODEL=roberta-large bash finetune.sh
# TASK=MNLI K=16 SEED=42 BS=8 LR=1e-5 MODEL=roberta-large bash finetune.sh
# TASK=RTE K=16 SEED=42 BS=8 LR=1e-5 MODEL=roberta-large bash finetune.sh

# Adam fine-tuning + prefix-tuning
# TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-2 MODEL=roberta-large EXTRA_TAG=prefix bash finetune.sh --prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act

# Adam fine-tuning + LoRA
# TASK=SST-2 K=16 SEED=42 BS=8 LR=1e-4 MODEL=roberta-large EXTRA_TAG=lora bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=SST-2 K=16 BS=8 MODEL=roberta-large EXTRA_TAG=lora bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=sst-5 K=16 SEED=42 BS=8 LR=1e-4 MODEL=roberta-large EXTRA_TAG=lora bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=SNLI K=16 SEED=100 BS=8 LR=1e-4 MODEL=roberta-large EXTRA_TAG=lora bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=MNLI K=16 BS=8 LR=3e-4 MODEL=roberta-large EXTRA_TAG=lora bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=RTE K=16 BS=8 LR=1e-4 MODEL=roberta-large EXTRA_TAG=lora bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=trec K=16 BS=8 LR=1e-4 MODEL=roberta-large EXTRA_TAG=lora bash finetune.sh --apply_lora --lora_r 8 --lora_alpha 16


SEEDS=(13 21 42 87 100)
LRS=(1e-4 3e-4 5e-4)

for SEED in "${SEEDS[@]}"; do
    for LR in "${LRS[@]}"; do
        TASK=${TASK:-SST-2}
        K=${K:-16}
        SEED=${SEED:-42}
        BS=${BS:-8}
        LR=${LR:-1e-5}
        STEP=${STEP:-1000}
        EVAL_STEP=${EVAL_STEP:-100}
        MODEL=${MODEL:-roberta-large}

        LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

        echo "TASK: $TASK"
        echo "K: $K"
        echo "Seed: $SEED"
        echo "BS: $BS"
        echo "LR: $LR"
        echo "Step: $STEP; Eval step: $EVAL_STEP"

        GR_TAG=seed$SEED-bs$BS-lr$LR-step$STEP-evalstep$EVAL_STEP
        EXTRA_TAG=${EXTRA_TAG:-ft}
        TAG=${TAG:-k${K}-${MODEL}-${EXTRA_TAG}}
        echo "Grid search tag: $GR_TAG"
        echo "Tag: $TAG"

        TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
            bash run_fewshot.sh --per_device_train_batch_size $BS --learning_rate $LR --eval_steps $EVAL_STEP \
            $@
    done
done