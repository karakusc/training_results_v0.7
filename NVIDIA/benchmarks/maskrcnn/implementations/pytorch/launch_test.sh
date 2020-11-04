NSOCKETS_PER_NODE=2
NCORES_PER_SOCKET=16
NPROC_PER_NODE=8

python -u -m bind_launch --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 --master_port 1234 \
    --nsockets_per_node=${NSOCKETS_PER_NODE} \
    --ncores_per_socket=${NCORES_PER_SOCKET} --nproc_per_node=${NPROC_PER_NODE} \
    tools/test_net.py --config-file 'configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
        DTYPE 'float16' \
        PATHS_CATALOG 'maskrcnn_benchmark/config/paths_catalog.py' \
        OUTPUT_DIR '/workspace/object_detection/outputs' \
        DISABLE_REDUCED_LOGGING True \
        TEST.IMS_PER_BATCH 8 \
        NHWC True \
        DATALOADER.NUM_WORKERS 4