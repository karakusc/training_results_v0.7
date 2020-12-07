#!/bin/bash
#sudo rm /usr/local/cuda
#sudo ln -s /usr/local/cuda-10.2 /usr/local/cuda
#sudo apt-get update -y
#sudo apt-get install -y numactl
#mkdir -p ~/data/coco
#pushd ~/data/coco
#COCO_IMAGES_URL=http://images.cocodataset.org/zips/
#TRAIN_ZIP=train2017.zip
#VAL_ZIP=val2017.zip
#TEST_ZIP=test2017.zip
#COCO_ANNO_URL=http://images.cocodataset.org/annotations/
#TRAIN_VAL_ANNO_ZIP=annotations_trainval2017.zip
#wget $COCO_IMAGES_URL$TRAIN_ZIP
#wget $COCO_IMAGES_URL$VAL_ZIP
#wget $COCO_IMAGES_URL$TEST_ZIP
#wget $COCO_ANNO_URL$TRAIN_VAL_ANNO_ZIP
#unzip $TRAIN_ZIP
#unzip $VAL_ZIP
#unzip $TEST_ZIP
#unzip $TRAIN_VAL_ANNO_ZIP
#popd
#conda create -y --name pt_mrcnn python=3.7 ipykernel
#conda activate pt_mrcnn
# install for CUDA 10.2
cd
CUDA_HOME=/usr/local/cuda  pip install -v --no-cache-dir torch==1.6.0 torchvision==0.7.0
#pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pybind11
git clone https://github.com/johnbensnyder/cocoapi
cd cocoapi/PythonAPI
#make install
pip install -v --no-cache-dir -e .
pip install mpi4py
pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip
pip install opencv-python==3.4.11.45
pip install yacs
cd
git clone https://github.com/NVIDIA/apex
cd apex
CUDA_HOME=/usr/local/cuda pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd
git clone -b horovod_scaling_jbsnyder https://github.com/anuragrs/training_results_v0.7/
cd /home/ubuntu/training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/pytorch
rm -rf build
pip install -v --no-cache-dir -e .
pip install -v --no-cache-dir tqdm
