import subprocess
import os
import tqdm
PATH=os.getcwd() # e.g. '/shared/roshanin/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch'
CONDA_ENV="/shared/herring_mzanur/bin/activate" # "/shared/roshanin/conda/bin/activate"
MASTER_ADDR='192.168.91.246' # one of the worker nodes on which evaluation is running
PORT=1234
NUM_NODES=8
CONFIG_FILE='configs/e2e_mask_rcnn_R_50_FPN_1x_giou_novo_ls.yaml' # e.g. /shared/roshanin/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml
PATHS_CATALOG='maskrcnn_benchmark/config/paths_catalog.py' # e.g. /shared/roshanin/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/maskrcnn_benchmark/config/paths_catalog_dbcluster.py
OUTPUT_DIR='/shared/mzanur/mrcnn_bs_64x8_homo_nvme_1workers/' # e.g./shared/datasets/checkpoints_train_eval_test1
EVAL_HOSTS='eval_hosts'
RESULTS_DIR='/shared/mzanur/eval_test'

CMD=F"ssh ubuntu@{{}} \
        screen -dm bash -c \
        \"cd {PATH} \
        && source {CONDA_ENV} \
        && python3 -u -m bind_launch --nnodes {NUM_NODES} --node_rank {{}} --master_addr {MASTER_ADDR} --master_port {PORT} \
        --nsockets_per_node=2 \
        --ncores_per_socket=24 --nproc_per_node=8 \
        tools/test_net.py --config-file '{CONFIG_FILE}' \
        DTYPE 'float16' \
        PATHS_CATALOG '{PATHS_CATALOG}' \
        OUTPUT_DIR '{OUTPUT_DIR}' \
        DISABLE_REDUCED_LOGGING True \
        TEST.IMS_PER_BATCH 256 \
        NHWC True \
        DATALOADER.NUM_WORKERS 1 \
        &> {RESULTS_DIR}\""


f = open(F"{EVAL_HOSTS}", "r")
lines = f.readlines()
count = 0
for line in lines:
    print("#########" + str(count) + "#############")
    bashCommand = CMD.format(line, count)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE) #, shell=True)
    output, error = process.communicate()
    print(bashCommand)
    print(output.decode("utf-8").strip(), error)
    count = count + 1
