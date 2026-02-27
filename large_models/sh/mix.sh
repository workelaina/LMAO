#!/bin/bash
# MODEL=facebook/opt-1.3b TASK=SST2 MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix.sh
# MODEL=facebook/opt-1.3b TASK=RTE MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix.sh
# MODEL=facebook/opt-1.3b TASK=BoolQ BS=4 MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix.sh
# MODEL=facebook/opt-1.3b TASK=WSC MODE=lora LR=1e-5 LR_base=1e-6 EPS=1e-2 bash mix.sh

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

SEEDS=(13 21 42 87 100)
for SEED in "${SEEDS[@]}"; do

    BS=${BS:-8}
    LR_lora=${LR_lora:-1e-4}
    LR_base=${LR_base:-1e-7}
    EPS=${EPS:-1e-3}
    # SEED=${SEED:-0}
    TRAIN=${TRAIN:-1000}
    DEV=${DEV:-500}
    EVAL=${EVAL:-1000}
    STEPS=${STEPS:-1000}
    EVAL_STEPS=${EVAL_STEPS:-100}

    MODE=${MODE:-ft}
    EXTRA_ARGS=""
    if [ "$MODE" == "prefix" ]; then
        EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
    elif [ "$MODE" == "lora" ]; then
        EXTRA_ARGS="--lora"
    fi
    TAG=mix-$MODE-$STEPS-$BS-$LR-$EPS-$SEED

    TASK_ARGS=""
    case $TASK in
        # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
        CB) # It has <1000 training examples. Only use 100 for dev
            DEV=100
            ;;
        Copa) # It has <1000 training examples. Only use 100 for dev
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
