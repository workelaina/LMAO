#!/bin/bash

# CUDA_VISIBLE_DEVICES=1

# seeds 13, 21, 42, 87, 100
# Base (ZO) + LoRA (FO)
# TASK=SST-2 K=16 SEED=87 BS=64 LR_lora=1e-4 LR_base=1e-5 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=SST-2 K=16 SEED=100 BS=64 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=sst-5 K=16 SEED=87 BS=64 LR_lora=1e-4 LR_base=1e-5 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=sst-5 K=16 SEED=100 BS=64 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=SNLI K=16 SEED=100 BS=64 LR_lora=5e-4 LR_base=1e-5 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=SNLI K=16 SEED=100 BS=64 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=MNLI K=16 SEED=21 BS=64 LR_lora=1e-4 LR_base=1e-5 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=MNLI K=16 SEED=100 BS=64 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=RTE K=16 SEED=21 BS=64 LR_lora=1e-4 LR_base=1e-5 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=RTE K=16 SEED=21 BS=64 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=trec K=16 SEED=21 BS=64 LR_lora=1e-4 LR_base=1e-5 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16
# TASK=trec K=16 SEED=21 BS=64 EPS=1e-3 MODEL=roberta-large EXTRA_TAG=lora bash mix.sh --apply_lora --lora_r 8 --lora_alpha 16

# TASK=${TASK:-SST-2}
# K=${K:-16}
# SEED=${SEED:-42}
# BS=${BS:-64}
# LR_lora=${LR_lora:-1e-6}
# LR_base=${LR_base:1e-6}
# EPS=${EPS:-1e-3}
# WD=${WD:-0}
# STEP=${STEP:-1000}
# EVAL_STEP=${EVAL_STEP:-100}
# MODEL=${MODEL:-roberta-large}

# LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

# echo "TASK: $TASK"
# echo "K: $K"
# echo "Seed: $SEED"
# echo "BS: $BS"
# echo "LR_lora: $LR_lora"
# echo "LR_base: $LR_base"
# echo "EPS: $EPS"
# echo "Step: $STEP; Eval step: $EVAL_STEP"

# GR_TAG=seed$SEED-bs$BS-lr_lora$LR_lora-lr_base$LR_base-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
# EXTRA_TAG=${EXTRA_TAG:-ft}
# TAG=${TAG:-k${K}-${MODEL}-mix-${EXTRA_TAG}}
# echo "Grid search tag: $GR_TAG"
# echo "Tag: $TAG"

# TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
#     bash run_fewshot.sh --per_device_train_batch_size $BS --lora_learning_rate $LR_lora --base_learning_rate $LR_base --eval_steps $EVAL_STEP --weight_decay $WD --zero_order_eps $EPS \
#     --zero_order_optim --lr_scheduler_type "constant" --lora_optimizer "adam" --base_optimizer "sgd" --efficient_zero_order \
#     $@

SEEDS=(13 21 42 87 100)
lr_lora=(1e-4 3e-4 5e-4)
lr_base=(1e-7 1e-6 1e-5)

for SEED in "${SEEDS[@]}"; do
    for LR_lora in "${lr_lora[@]}"; do
        for LR_base in "${lr_base[@]}"; do
            TASK=${TASK:-SST-2}
            K=${K:-16}
            BS=${BS:-64}
            LR_lora=${LR_lora:-1e-6}
            LR_base=${LR_base:-1e-6}
            EPS=${EPS:-1e-3}
            WD=${WD:-0}
            STEP=${STEP:-1000}
            EVAL_STEP=${EVAL_STEP:-100}
            MODEL=${MODEL:-roberta-large}

            LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

            echo "Running with seed: $SEED"
            echo "TASK: $TASK"
            echo "K: $K"
            echo "Seed: $SEED"
            echo "BS: $BS"
            echo "LR_lora: $LR_lora"
            echo "LR_base: $LR_base"
            echo "EPS: $EPS"
            echo "Step: $STEP; Eval step: $EVAL_STEP"

            GR_TAG=seed$SEED-bs$BS-lr_lora$LR_lora-lr_base$LR_base-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
            EXTRA_TAG=${EXTRA_TAG:-ft}
            TAG=${TAG:-k${K}-${MODEL}-mix-${EXTRA_TAG}}
            
            TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
                bash run_fewshot.sh --per_device_train_batch_size $BS --lora_learning_rate $LR_lora --base_learning_rate $LR_base --eval_steps $EVAL_STEP --weight_decay $WD --zero_order_eps $EPS \
                --zero_order_optim --lr_scheduler_type "constant" --lora_optimizer "adam" --base_optimizer "sgd" --efficient_zero_order \
                $@
        done
    done
done
