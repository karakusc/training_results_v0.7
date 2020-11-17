import sys
import subprocess
# ip_list = ['ip-192-168-75-157', 'ip-192-168-93-77']
ip_list = sys.argv[1].split(',')
path = "/shared/jbsnyder/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch"
env_path = "/shared/jbsnyder/mrcnn_pt/bin/activate"
output_dir = "/shared/jbsnyder/output"
log_dir = "/shared/jbsnyder/logs/eval_log.log"

CMD="ssh ubuntu@{7} \
        screen -dm bash -c \
        \"cd {0} \
        && source {1} \
        && python3 -u -m bind_launch --nnodes {2} --node_rank {6} --master_addr {3} --master_port 1234 \
        --nsockets_per_node=2 \
        --ncores_per_socket=24 --nproc_per_node=8 \
        tools/test_net.py --config-file '{0}/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
        DTYPE 'float16' \
        PATHS_CATALOG '{0}/maskrcnn_benchmark/config/paths_catalog_dbcluster.py' \
        OUTPUT_DIR '{4}' \
        DISABLE_REDUCED_LOGGING True \
        TEST.IMS_PER_BATCH 256 \
        NHWC True \
        DATALOADER.NUM_WORKERS 4 \
        &> {5}\""

for rank, ip_address in enumerate(ip_list):
    bashCommand = CMD.format(path, env_path, len(ip_list), ip_list[0], output_dir, log_dir, rank, ip_address)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(bashCommand)
    print(output.decode("utf-8").strip(), error)
