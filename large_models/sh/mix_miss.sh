#!/bin/bash

# export CUDA_VISIBLE_DEVICES=$USE_GPU

# export HF_HOME=/cache/ely_hfcache/home
# export HF_DATASETS_CACHE=/cache/ely_hfcache/dataset
# export TRANSFORMERS_CACHE=/cache/ely_hfcache/tf

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

# TASKS=(SST2 RTE WSC WIC CB Copa MultiRC ReCoRD DROP SQuAD BoolQ)
TASK=${TASK:-ReCoRD}
SEEDS=(13 21 42 87 100 0 4242 3407)

# for TASK in "${TASKS[@]}"; do
for SEED in "${SEEDS[@]}"; do

    BS=${BS:-2}
    LR_lora=${LR_lora:-1e-4}
    LR_base=${LR_base:-1e-7}
    EPS=${EPS:-1e-3}
    # SEED=${SEED:-0}
    TRAIN=${TRAIN:-1000}
    DEV=${DEV:-500}
    EVAL=${EVAL:-1000}
    STEPS=${STEPS:-1000}
    EVAL_STEPS=${EVAL_STEPS:-100}

    TASK_ARGS=""
    case $TASK in
        # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
        ReCoRD)
            # Can only fit real bsz = 2 on 80G A100
            GA=1
            echo "Gradient accumulation: $GA"
            TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
            ;;
    esac

    MODE=${MODE:-ft}
    EXTRA_ARGS=""
    if [ "$MODE" == "prefix" ]; then
        EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
    elif [ "$MODE" == "lora" ]; then
        EXTRA_ARGS="--lora"
    fi
    TAG=mix-$MODE-$STEPS-$BS-$LR-$EPS-$SEED

    echo $TAG
    echo "BS: $BS"
    echo "LR: $LR"
    echo "LR_base: $LR_base"
    echo "EPS: $EPS"
    echo "SEED: $SEED"
    echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
    echo "MODE: $MODE"
    echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
    date

    python run.py \
        --model_name $MODEL \
        --task_name $TASK \
        --output_dir result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
        --max_steps $STEPS \
        --trainer mix --fp16 \
        --learning_rate $LR --base_learning_rate $LR_base --zo_eps $EPS --per_device_train_batch_size $BS \
        --load_best_model_at_end --evaluation_strategy epoch --save_strategy epoch --save_total_limit 1 \
        --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
        --train_as_classification \
        $EXTRA_ARGS \
        $TASK_ARGS \
        "$@"
    date
done
