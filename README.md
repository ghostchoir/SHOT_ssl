# SHOT_ssl

## VisDA-C

## CIFAR-10-C
- Source Pre-training
```
python image_source_ssl.py --gpu_id 1 --lr 1e-3 --max_epoch 1 --dset CIFAR-10-C --net resnet26 --output ../../SHOT_exp/cifar_test
```

- Adaptation
```
python image_target_ssl.py --gpu_id 1 --max_epoch 1 --dset CIFAR-10-C --lr 1e-3 --net resnet26 --output ../../SHOT_exp/cifar_test --output_src ../../SHOT_exp/cifar_test
```

