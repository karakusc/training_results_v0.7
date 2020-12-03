#!/bin/bash
#SBATCH --nodes=8
#SBATCH --time 00:10:00
#SBATCH --ntasks-per-node=8
#SBATCH --job-name="pytorch-dist-test"
#SBATCH --output=pt_test_8.out

BASE_LR=0.008
MAX_ITER=45000
WARMUP_FACTOR=0.0001
WARMUP_ITERS=100
NUM_NODES=4
TRAIN_IMS_PER_BATCH=512
TEST_IMS_PER_BATCH=512
WEIGHT_DECAY=5e-4
NSOCKETS_PER_NODE=2
NCORES_PER_SOCKET=24
NPROC_PER_NODE=8
OPTIMIZER="NovoGrad"
LR_SCHEDULE="COSINE"
BETA1=0.9
BETA2=0.4

CONTROLLER=`/bin/hostname -s`
mapfile -t WORKERS < ~/master_host
PORT=1234

for (( RANK = 0 ; RANK < $NUM_NODES ; RANK++))
do
    ssh -q ${WORKERS[$RANK]} \
    python -u -m bind_launch \
    --nnodes=${NUM_NODES} \
    --node_rank=${RANK} \
    --master_addr=${CONTROLLER} \
    --master_port=${PORT} \
    --nsockets_per_node=${NSOCKETS_PER_NODE} \
    --ncores_per_socket=${NCORES_PER_SOCKET} \
    --nproc_per_node=${NPROC_PER_NODE} \
	 tools/train_mlperf.py --config-file 'configs/e2e_mask_rcnn_R_50_FPN_1x_giou_novo_ls.yaml' \
	 DTYPE 'float16' \
	 PATHS_CATALOG 'maskrcnn_benchmark/config/paths_catalog.py' \
	 DISABLE_REDUCED_LOGGING True \
	 INPUT.ADD_NOISE False \
	 DATALOADER.NUM_WORKERS 12 \
	 SOLVER.BASE_LR ${BASE_LR} \
	 SOLVER.WEIGHT_DECAY ${WEIGHT_DECAY} \
	 SOLVER.MAX_ITER ${MAX_ITER} \
	 SOLVER.WARMUP_FACTOR ${WARMUP_FACTOR} \
	 SOLVER.WARMUP_ITERS ${WARMUP_ITERS} \
	 SOLVER.WEIGHT_DECAY_BIAS 0 \
	 SOLVER.WARMUP_METHOD linear \
	 SOLVER.IMS_PER_BATCH ${TRAIN_IMS_PER_BATCH} \
	 SOLVER.OPTIMIZER ${OPTIMIZER} \
	 SOLVER.BETA1 ${BETA1} \
	 SOLVER.BETA2 ${BETA2} \
	 MODEL.RPN.LS ${LS} \
	 MODEL.ROI_HEADS.LS ${LS} \
	 SOLVER.LR_SCHEDULE ${LR_SCHEDULE} \
	 TEST.IMS_PER_BATCH ${TEST_IMS_PER_BATCH} \
	 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN ${FPN_POST_NMS_TOP_N_TRAIN} \
	 NHWC True &
 RANK=$((RANK+1))

done
wait

