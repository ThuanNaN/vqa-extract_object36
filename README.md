## Faster R-CNN Feature Extraction


We use the Faster R-CNN feature extractor demonstrated in ["Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering", CVPR 2018](https://arxiv.org/abs/1707.07998)
and its released code at [Bottom-Up-Attention github repo](https://github.com/peteanderson80/bottom-up-attention).
It was trained on [Visual Genome](https://visualgenome.org/) dataset and implemented based on a specific [Caffe](https://caffe.berkeleyvision.org/) version.


To extract features with this Caffe Faster R-CNN, we publicly release a docker image `airsplay/bottom-up-attention` on docker hub that takes care of all the dependencies and library installation . Instructions and examples are demonstrated below. You could also follow the installation instructions in the bottom-up attention github to setup the tool: [https://github.com/peteanderson80/bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention). 

The BUTD feature extractor is widely used in many other projects. If you want to reproduce the results from their paper, feel free to use our docker as a tool.


### Feature Extraction with Docker
[Docker](https://www.docker.com/) is a easy-to-use virtualization tool which allows you to plug and play without installing libraries.

The built docker file for bottom-up-attention is released on [docker hub](https://hub.docker.com/r/airsplay/bottom-up-attention) and could be downloaded with command: 
```bash
sudo docker pull airsplay/bottom-up-attention
```
> The `Dockerfile` could be downloaed [here](https://drive.google.com/file/d/1KJjwQtqisXvinWm8OORk-_3XYLBHYCIK/view?usp=sharing), which allows using other CUDA versions.

After pulling the docker, you could test running the docker container with command:
```bash
docker run --gpus all --rm -it airsplay/bottom-up-attention bash
``` 


If errors about `--gpus all` popped up, please read the next section.

#### Docker GPU Access
Note that the purpose of the argument `--gpus all` is to expose GPU devices to the docker container, and it requires Docker >= 19.03 along with `nvidia-container-toolkit`:
1. [Docker CE 19.03](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
2. [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker)

For running Docker with an older version, either update it to 19.03 or use the flag `--runtime=nvidia` instead of `--gpus all`.

#### An Example: Feature Extraction for NLVR2 
We demonstrate how to extract Faster R-CNN features of NLVR2 images.

1. Please first follow the instructions on the [NLVR2 official repo](https://github.com/lil-lab/nlvr/tree/master/nlvr2) to get the images.

2. Download the pre-trained Faster R-CNN model. Instead of using the default pre-trained model (trained with 10 to 100 boxes), we use the ['alternative pretrained model'](https://github.com/peteanderson80/bottom-up-attention#demo) which was trained with 36 boxes. 
    ```bash
    wget 'https://www.dropbox.com/s/2h4hmgcvpaewizu/resnet101_faster_rcnn_final_iter_320000.caffemodel?dl=1' -O data/nlvr2_imgfeat/resnet101_faster_rcnn_final_iter_320000.caffemodel
    ```

3. Run docker container with command:
    ```bash
    docker run --gpus all -v /path/to/nlvr2/images:/workspace/images:ro -v /path/to/lxrt_public/data/nlvr2_imgfeat:/workspace/features --rm -it airsplay/bottom-up-attention bash
    ```
    `-v` mounts the folders on host os to the docker image container.
    > Note0: If it says something about 'privilege', add `sudo` before the command.
    >
    > Note1: If it says something about '--gpus all', it means that the GPU options are not correctly set. Please read [Docker GPU Access](#docker-gpu-access) for the instructions to allow GPU access.
    >
    > Note2: /path/to/nlvr2/images would contain subfolders `train`, `dev`, `test1` and `test2`.
    >
    > Note3: Both paths '/path/to/nlvr2/images/' and '/path/to/lxrt_public' requires absolute paths.


4. Extract the features **inside the docker container**. The extraction script is copied from [butd/tools/generate_tsv.py](https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/generate_tsv.py) and modified by [Jie Lei](http://www.cs.unc.edu/~jielei/) and me.
    ```bash
    cd /workspace/features
    CUDA_VISIBLE_DEVICES=0 python extract_nlvr2_image.py --split train 
    CUDA_VISIBLE_DEVICES=0 python extract_nlvr2_image.py --split valid
    CUDA_VISIBLE_DEVICES=0 python extract_nlvr2_image.py --split test
    ```

5. It would takes around 5 to 6 hours for the training split and 1 to 2 hours for the valid and test splits. Since it is slow, I recommend to run them parallelly if there are multiple GPUs. It could be achieved by changing the `gpu_id` in `CUDA_VISIBLE_DEVICES=$gpu_id`.

The features will be saved in `train.tsv`, `valid.tsv`, and `test.tsv` under the directory `data/nlvr2_imgfeat`, outside the docker container. I have verified the extracted image features are the same to the ones I provided in [NLVR2 fine-tuning](#nlvr2).

#### Yet Another Example: Feature Extraction for MS COCO Images
1. Download the MS COCO train2014, val2014, and test2015 images from [MS COCO official website](http://cocodataset.org/#download).

2. Download the pre-trained Faster R-CNN model. 
    ```bash
    mkdir -p data/mscoco_imgfeat
    wget 'https://www.dropbox.com/s/2h4hmgcvpaewizu/resnet101_faster_rcnn_final_iter_320000.caffemodel?dl=1' -O data/mscoco_imgfeat/resnet101_faster_rcnn_final_iter_320000.caffemodel
    ```

3. Run the docker container with the command:
    ```bash
    docker run --gpus all -v /path/to/mscoco/images:/workspace/images:ro -v $(pwd)/data/mscoco_imgfeat:/workspace/features --rm -it airsplay/bottom-up-attention bash
    ```
    > Note: Option `-v` mounts the folders outside container to the paths inside the container.
    > 
    > Note1: Please use the **absolute path** to the MS COCO images folder `images`. The `images` folder containing the `train2014`, `val2014`, and `test2015` sub-folders. (It's the standard way to save MS COCO images.)

4. Extract the features **inside the docker container**.
    ```bash
    cd /workspace/features
    CUDA_VISIBLE_DEVICES=0 python extract_coco_image.py --split train 
    CUDA_VISIBLE_DEVICES=0 python extract_coco_image.py --split valid
    CUDA_VISIBLE_DEVICES=0 python extract_coco_image.py --split test
    ```
 
5. Exit from the docker container (by executing `exit` command in bash). The extracted features would be saved under folder `data/mscoco_imgfeat`. 
