#!/bin/bash

export CUDA_VISIBLE_DEVICES=$USE_GPU

export HF_HOME=/cache/ely_hfcache/home
export HF_DATASETS_CACHE=/cache/ely_hfcache/dataset
export TRANSFORMERS_CACHE=/cache/ely_hfcache/tf

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

TASKS=(SST2 RTE WSC WIC CB Copa MultiRC ReCoRD DROP SQuAD BoolQ)

for TASK in "${TASKS[@]}"; do

    BS=${BS:-16}
    LR=${LR:-1e-5}
    EPS=${EPS:-1e-3}
    SEED=${SEED:-0}
    TRAIN=${TRAIN:-1000}
    DEV=${DEV:-500}
    EVAL=${EVAL:-1000}
    STEPS=${STEPS:-20000}
    EVAL_STEPS=${EVAL_STEPS:-4000}

    TASK_ARGS=""
    case $TASK in
        # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
        CB)
            # It has <1000 training examples. Only use 100 for dev
            DEV=100
            ;;
        Copa)
            # It has <1000 training examples. Only use 100 for dev
            DEV=100
            TASK_ARGS="--train_as_classification False"
            ;;
        ReCoRD)
            TASK_ARGS="--train_as_classification False"
            ;;
        DROP)
            TASK_ARGS="--train_as_classification False"
            ;;
        SQuAD)
            TASK_ARGS="--train_as_classification False"
            ;;
    esac

    MODE=${MODE:-ft}
    EXTRA_ARGS=""
    # if [ "$MODE" == "prefix" ]; then
    #     EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
    # elif [ "$MODE" == "lora" ]; then
    #     EXTRA_ARGS="--lora"
    # fi
    TAG=mezo-$MODE-$STEPS-$BS-$LR-$EPS-$SEED

    echo $TAG
    echo "BS: $BS"
    echo "LR: $LR"
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
        --trainer zo --fp16\
        --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "constant" \
        --load_best_model_at_end --evaluation_strategy steps --save_strategy steps --save_total_limit 1 \
        --eval_steps $EVAL_STEPS --save_steps $EVAL_STEPS \
        --train_as_classification \
        $EXTRA_ARGS \
        $TASK_ARGS \
        "$@"
    date
done
