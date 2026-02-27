#!/bin/bash
# USE_GPU=1 bash mix_all_sensitivity_32.sh --lora_r 32
export CUDA_VISIBLE_DEVICES=$USE_GPU

export HF_HOME=/cache/ely_hfcache/home
export HF_DATASETS_CACHE=/cache/ely_hfcache/dataset
export TRANSFORMERS_CACHE=/cache/ely_hfcache/tf

MODEL=${MODEL:-roberta-large}
EXTRA_TAG=${EXTRA_TAG:-lora}

SEEDS=(13 21 42 87 100)
# TASKS=(SST-2 sst-5 SNLI MNLI RTE trec)
TASKS=(sst-5 SNLI MNLI RTE trec)
# lr_lora=(1e-4 3e-4 5e-4)
# lr_base=(1e-7 1e-6 1e-5)
lr_lora=(1e-4)
lr_base=(1e-6)

for TASK in "${TASKS[@]}"; do
    for alpha in 4 8 16; do
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
                    echo "Model: $Model"
                    echo "EXTRA_TAG: $EXTRA_TAG"
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
                        --zero_order_optim --lr_scheduler_type "constant" --lora_optimizer "adam" --base_optimizer "sgd" --efficient_zero_order --apply_lora --lora_alpha $alpha \
                        $@
                done
            done
        done
    done
done