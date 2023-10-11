# !/usr/bin/env bash
data_dir=E:\\毕设论文\\轴承数据集\\轴承数据集\\StandardSamples\\data-pic\\with_box\\1000
# Office31
# python main.py --config D:\\毕设\\DeepDA\\NEW\\NEW.yaml --data_dir $data_dir --src_domain 0 --tgt_domain 20 --weights resnet18_111.pth
# python main.py --config D:\\毕设\\DeepDA\\DSAN\\DSAN.yaml --data_dir $data_dir --src_domain 0 --tgt_domain 20 --weights resnet18_111.pth
# python main.py --config D:\\毕设\\DeepDA\\BNM\\BNM.yaml --data_dir $data_dir --src_domain 0 --tgt_domain 20 --weights resnet18_111.pth
# python main.py --config D:\\毕设\\DeepDA\\DAAN\\DAAN.yaml --data_dir $data_dir --src_domain 0 --tgt_domain 20 --weights resnet18_111.pth
# python main.py --config D:\\毕设\\DeepDA\\DAN\\DAN.yaml --data_dir $data_dir --src_domain 0 --tgt_domain 20 --weights resnet18_111.pth
# python main.py --config D:\\毕设\\DeepDA\\DANN\\DANN.yaml --data_dir $data_dir --src_domain 0 --tgt_domain 20 --weights resnet18_111.pth
# python main.py --config D:\\毕设\\DeepDA\\DeepCoral\\DeepCoral.yaml --data_dir $data_dir --src_domain 0 --tgt_domain 20 --weights resnet18_111.pth

# python main.py --config D:\\毕设\\DeepDA\\NEW\\NEW.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 0 --weights resnet18_222.pth
python main.py --config DeepDA\\DSAN\\DSAN.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 0 --weights resnet18_222.pth
python main.py --config DeepDA\\BNM\\BNM.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 0 --weights resnet18_222.pth
python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 0 --weights resnet18_222.pth
python main.py --config DeepDA\\DAN\\DAN.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 0 --weights resnet18_222.pth
python main.py --config DeepDA\\DANN\\DANN.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 0 --weights resnet18_222.pth
python main.py --config DeepDA\\DeepCoral\\DeepCoral.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 0 --weights resnet18_222.pth

# python main.py --config D:\\毕设\\DeepDA\\NEW\\NEW.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 40 --weights resnet18_222.pth
python main.py --config DeepDA\\DSAN\\DSAN.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 40 --weights resnet18_222.pth
python main.py --config DeepDA\\BNM\\BNM.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 40 --weights resnet18_222.pth
python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 40 --weights resnet18_222.pth
python main.py --config DeepDA\\DAN\\DAN.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 40 --weights resnet18_222.pth
python main.py --config DeepDA\\DANN\\DANN.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 40 --weights resnet18_222.pth
python main.py --config DeepDA\\DeepCoral\\DeepCoral.yaml --data_dir $data_dir --src_domain 20 --tgt_domain 40 --weights resnet18_222.pth

python main.py --config DeepDA\\DSAN\\DSAN.yaml --data_dir $data_dir --src_domain 40 --tgt_domain 20 --weights resnet18_333.pth
python main.py --config DeepDA\\BNM\\BNM.yaml --data_dir $data_dir --src_domain 40 --tgt_domain 20 --weights resnet18_333.pth
python main.py --config DeepDA\\DAAN\\DAAN.yaml --data_dir $data_dir --src_domain 40 --tgt_domain 20 --weights resnet18_333.pth
python main.py --config DeepDA\\DAN\\DAN.yaml --data_dir $data_dir --src_domain 40 --tgt_domain 20 --weights resnet18_333.pth
python main.py --config DeepDA\\DANN\\DANN.yaml --data_dir $data_dir --src_domain 40 --tgt_domain 20 --weights resnet18_333.pth
python main.py --config DeepDA\\DeepCoral\\DeepCoral.yaml --data_dir $data_dir --src_domain 40 --tgt_domain 20 --weights resnet18_333.pth