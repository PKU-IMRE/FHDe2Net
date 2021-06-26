#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python2 test.py --dataroot "/media/he/80FE99D1FE99BFB8/FHDMi/test"  --netGDN "ckpt/netGDN.pth" --netLRN "ckpt/netLRN.pth" --netFDN "ckpt/netFDN.pth" --netFRN "ckpt/netFRN.pth" --batchSize 2 --originalSize_h 1080 --originalSize_w 1920 --imageSize_h 1080 --imageSize_w 1920 --image_path "results" --write 1 --record "results.txt"
 
