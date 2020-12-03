#!/bin/bash
# this script has novograd + herring

#source /shared/herring_mzanur/bin/activate

BASE_LR=0.07
MAX_ITER=3700
WARMUP_FACTOR=0.0133
WARMUP_ITERS=700
TRAIN_IMS_PER_BATCH=512
TEST_IMS_PER_BATCH=512
WEIGHT_DECAY=1.1e-3
NSOCKETS_PER_NODE=2
NCORES_PER_SOCKET=24
NPROC_PER_NODE=8
FPN_POST_NMS_TOP_N_TRAIN=1000
OPTIMIZER="NovoGrad"
LR_SCHEDULE="COSINE"
BETA1=0.9
BETA2=0.32
LS=0.1
rm -rf  /shared/jbsnyder/output/*

# cd /shared/mzanur/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
/shared/mzanur/conda_pt/bin/herringrun -n 64 -c /shared/mzanur/conda_pt \
 USE_HERRING_ALL_REDUCE=1 /shared/mzanur/conda_pt/bin/python tools/train_mlperf.py \
 --config-file 'configs/e2e_mask_rcnn_R_50_FPN_1x_giou_novo_ls.yaml' \
 DTYPE 'float16' \
 PATHS_CATALOG 'maskrcnn_benchmark/config/paths_catalog_dbcluster.py' \
 DISABLE_REDUCED_LOGGING True \
 INPUT.ADD_NOISE False \
 DATALOADER.NUM_WORKERS 1 \
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
 MODEL.WEIGHT "/shared/R-50.pkl" \
 SOLVER.LR_SCHEDULE ${LR_SCHEDULE} \
 TEST.IMS_PER_BATCH ${TEST_IMS_PER_BATCH} \
 MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN ${FPN_POST_NMS_TOP_N_TRAIN} \
 OUTPUT_DIR '/shared/jbsnyder/output' \
 NHWC True | tee ${BASEDIR}/log_${BETA2}