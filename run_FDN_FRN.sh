#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python2.7 train_FDN_FRN.py --dataroot "/media/he/80FE99D1FE99BFB8/longmao_final/train" --valDataroot "" --pre "" --name "FDN_FRN" --exp "FDN_FRN" --netGDN "./ckpt/netGDN.pth" --netLRN "ckpt/netLRN.pth" --display_port 8099 --originalSize_h 1080 --originalSize_w 1920 --imageSize_h 1024 --imageSize_w 1024 --batchSize 1
