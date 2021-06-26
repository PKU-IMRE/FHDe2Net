#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 python2.7 train_GDN.py --dataroot "/media/he/80FE99D1FE99BFB8/longmao_final/train" --valDataroot "/media/he/80FE99D1FE99BFB8/longmao_final/val" --pre "" --name "GDN" --exp "GDN" --display_port 8099 --originalSize_h 420 --originalSize_w 420 --imageSize_h 384 --imageSize_w 384 --batchSize 2
