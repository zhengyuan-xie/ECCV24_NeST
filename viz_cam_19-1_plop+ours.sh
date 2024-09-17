#!/bin/bash

set -e

start=`date +%s`

START_DATE=$(date '+%Y-%m-%d')

PORT=$((9002 + RANDOM % 1000))
GPU=0
NB_GPU=1


DATA_ROOT=/defaultShare/archive/xiezhengyuan/PascalVOC12

DATASET=voc
TASK=19-1
NAME=ours
METHOD=ours
OPTIONS="--checkpoint checkpoints/step/"

SCREENNAME="${DATASET}_${TASK}_${NAME} On GPUs ${GPU}"

RESULTSFILE=results/${START_DATE}_${DATASET}_${TASK}_${NAME}.csv
rm -f ${RESULTSFILE}

echo -ne "\ek${SCREENNAME}\e\\"

echo "Writing in ${RESULTSFILE}"


BATCH_SIZE=1
INITIAL_EPOCHS=30
EPOCHS=30


CUDA_VISIBLE_DEVICES=${GPU} python3 -m torch.distributed.launch --master_port ${PORT} --nproc_per_node=${NB_GPU} CAM_visualization.py --date ${START_DATE} --data_root ${DATA_ROOT} --overlap --batch_size ${BATCH_SIZE} --dataset ${DATASET} --name ${NAME} --task ${TASK} --step 1 --lr 0.001 --epochs ${EPOCHS} --method ${METHOD} --opt_level O1 ${OPTIONS}


echo ${SCREENNAME}


end=`date +%s`
runtime=$((end-start))
echo "Run in ${runtime}s"
