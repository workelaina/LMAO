#!/bin/bash

export CUDA_VISIBLE_DEVICES=$USE_GPU

export HF_HOME=/ephemeral/home
export HF_DATASETS_CACHE=/ephemeral/dataset
export TRANSFORMERS_CACHE=/ephemeral/tf

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

TASKS=(SST2 RTE WSC WIC CB Copa MultiRC ReCoRD DROP SQuAD BoolQ)

for TASK in "${TASKS[@]}"; do
    TRAIN=${TRAIN:-32}
    EVAL=${EVAL:-1000}
    TAG=icl-$TRAIN-$SEED

    echo $TAG
    echo "TRAIN: $TRAIN"
    echo "SEED: $SEED"
    date

    python run.py --model_name $MODEL --task_name $TASK --output_dir /ephemeral/result/$TASK-${MODEL_NAME}-$TAG --tag $TAG --train_set_seed $SEED --num_train $TRAIN --num_eval $EVAL --load_float16 --verbose "$@"
    date
done
