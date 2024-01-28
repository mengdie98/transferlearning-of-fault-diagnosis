#!/usr/bin/env bash
GPU_ID=0
# Office31
echo "start running script"
# CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir D:\\data\\CWRUData-picture\\CWRUData-picture\\12K_Drive_End --src_domain 1772\\7 --tgt_domain 1730\\7 --weights resnet18_2.pth --savename transfer_resnet18_2-0-
# CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir D:\\data\\CWRUData-picture\\CWRUData-picture\\12K_Drive_End --src_domain 1772\\7 --tgt_domain 1750\\7 --weights resnet18_2.pth --savename transfer_resnet18_2-1-
# CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir D:\\data\\CWRUData-picture\\CWRUData-picture\\12K_Drive_End --src_domain 1772\\7 --tgt_domain 1797\\7 --weights resnet18_2.pth --savename transfer_resnet18_2-3-
# CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir D:\\data\\CWRUData-picture\\CWRUData-picture\\12K_Drive_End --src_domain 1797\\7 --tgt_domain 1772\\7 --weights resnet18_3.pth --savename transfer_resnet18_3-2-
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir D:\\data\\CWRUData-picture\\CWRUData-picture\\12K_Drive_End --src_domain 1797\\7 --tgt_domain 1750\\7 --weights resnet18_3.pth --savename transfer_resnet18_3-1-
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir D:\\data\\CWRUData-picture\\CWRUData-picture\\12K_Drive_End --src_domain 1797\\7 --tgt_domain 1730\\7 --weights resnet18_3.pth --savename transfer_resnet18_3-0-
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir D:\\data\\CWRUData-picture\\CWRUData-picture\\12K_Drive_End --src_domain 1750\\7 --tgt_domain 1730\\7 --weights resnet18_1.pth --savename transfer_resnet18_1-0-
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir D:\\data\\CWRUData-picture\\CWRUData-picture\\12K_Drive_End --src_domain 1750\\7 --tgt_domain 1772\\7 --weights resnet18_1.pth --savename transfer_resnet18_1-2-
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir D:\\data\\CWRUData-picture\\CWRUData-picture\\12K_Drive_End --src_domain 1750\\7 --tgt_domain 1797\\7 --weights resnet18_1.pth --savename transfer_resnet18_1-3-


