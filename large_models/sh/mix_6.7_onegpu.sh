#!/bin/bash

export CUDA_VISIBLE_DEVICES=$USE_GPU

export HF_HOME=/ephemeral/home
export HF_DATASETS_CACHE=/ephemeral/dataset
export TRANSFORMERS_CACHE=/ephemeral/tf

MODEL=${MODEL:-facebook/opt-6.7b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

TASKS=(SST2 RTE WSC WIC CB Copa MultiRC ReCoRD DROP SQuAD BoolQ)
SEEDS=(13 21 42 87 100 0 4242 3407)

for SEED in "${SEEDS[@]}"; do
    for TASK in "${TASKS[@]}"; do
        BS=1
        LR_lora=${LR_lora:-1e-4}
        LR_base=${LR_base:-1e-7}
        EPS=${EPS:-1e-3}
        # SEED=${SEED:-0}
        TRAIN=${TRAIN:-1000}
        DEV=500
        # DEV=${DEV:-500}
        EVAL=${EVAL:-1000}
        STEPS=${STEPS:-1000}
        EVAL_STEPS=${EVAL_STEPS:-100}

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
            MultiRC)
                # Can only fit real bsz = 2 on 80G A100
                GA=1
                # BS=2
                echo "Gradient accumulation: $GA"
                TASK_ARGS="--gradient_accumulation_steps $GA"
                ;;
            ReCoRD)
                # Can only fit real bsz = 2 on 80G A100
                GA=1
                # BS=2
                echo "Gradient accumulation: $GA"
                TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
                ;;
            DROP)
                # Can only fit real bsz = 1 on 80G A100
                GA=1
                # BS=1
                echo "Gradient accumulation: $GA"
                TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
                ;;
            SQuAD)
                GA=1
                # BS=1
                echo "Gradient accumulation: $GA"
                TASK_ARGS="--gradient_accumulation_steps $GA --train_as_classification False"
                ;;
            BoolQ)
                GA=1
                # BS=2
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
            --output_dir /ephemeral/result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
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
done
