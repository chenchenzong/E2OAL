# Revisiting Unknowns: Towards Effective and Efficient Open-Set Active Learning

## Run 

Please use the following command to install the environment dependencies:

```pip install -r requirements.txt```

### Datasets

For CIFAR-10 or CIFAR-100, please download it to `~/data`.
For Tiny-ImageNet, please download it to `~/tiny-imagenet-200`.

### Extract visual features using CLIP
```
python extract_features.py
```

### CIFAR-10

```
python main.py --query-strategy e2oal_sampling --init-percent 1 --known-class 2 --query-batch 1500 --seed 1 --model resnet50 --dataset cifar10 --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy e2oal_sampling --init-percent 1 --known-class 3 --query-batch 1500 --seed 1 --model resnet50 --dataset cifar10 --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy e2oal_sampling --init-percent 1 --known-class 4 --query-batch 1500 --seed 1 --model resnet50 --dataset cifar10 --max-query 11 --max-epoch 200 --gpu 0
```


### CIFAR-100

```
python main.py --query-strategy e2oal_sampling --init-percent 8 --known-class 20 --query-batch 1500 --seed 1 --model resnet50 --dataset cifar100 --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy e2oal_sampling --init-percent 8 --known-class 30 --query-batch 1500 --seed 1 --model resnet50 --dataset cifar100 --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy e2oal_sampling --init-percent 8 --known-class 40 --query-batch 1500 --seed 1 --model resnet50 --dataset cifar100 --max-query 11 --max-epoch 200 --gpu 0
```


### Tiny-ImageNet

```
python main.py --query-strategy e2oal_sampling --init-percent 8 --known-class 20 --query-batch 1500 --seed 1 --model resnet50 --dataset tinyimagenet --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy e2oal_sampling --init-percent 8 --known-class 30 --query-batch 1500 --seed 1 --model resnet50 --dataset tinyimagenet --max-query 11 --max-epoch 200 --gpu 0

python main.py --query-strategy e2oal_sampling --init-percent 8 --known-class 40 --query-batch 1500 --seed 1 --model resnet50 --dataset tinyimagenet --max-query 11 --max-epoch 200 --gpu 0
```

### If you encounter the following error:

```ValueError: too many values to unpack (expected 2)```

You can resolve it with the following steps:

1. Locate the source code of ```torch.utils.data.DataLoader```. The path is typically one of the following:
   - ```anaconda3/lib/python3.x/site-packages/torch/utils/data/dataloader.py```
   - ```anaconda3/envs/your_env_name/lib/python3.x/site-packages/torch/utils/data/dataloader.py```
2. Open the file ```dataloader.py```, and find the ```_SingleProcessDataLoaderIter``` class.
3. Modify the ```_next_data``` method by changing the return value from:```return data``` to ```return index, data```