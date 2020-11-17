import subprocess

path = '/shared/jbsnyder/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch'

env_path = "/shared/mzanur/conda_pt/bin/activate"

CMD="ssh ubuntu@{} \
        screen -dm bash -c \
        \"cd /shared/jbsnyder/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/ \
        && source /shared/mzanur/conda_pt/bin/activate \
        && python3 -u -m bind_launch --nnodes 8 --node_rank {} --master_addr 192.168.76.46 --master_port 1234 \
        --nsockets_per_node=2 \
        --ncores_per_socket=24 --nproc_per_node=8 \
        tools/test_net.py --config-file '/shared/jbsnyder/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
        DTYPE 'float16' \
        PATHS_CATALOG '/shared/jbsnyder/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/maskrcnn_benchmark/config/paths_catalog_dbcluster.py' \
        OUTPUT_DIR '/shared/jbsnyder/output' \
        DISABLE_REDUCED_LOGGING True \
        TEST.IMS_PER_BATCH 256 \
        NHWC True \
        DATALOADER.NUM_WORKERS 4 \
        &> /shared/jbsnyder/logs\""

f = open("/shared/jbsnyder/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch/newh", "r")
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
