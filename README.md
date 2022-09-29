### Evaluation in vision domain

Train our method on ImageNet and then perform linear probe

```
python main.py --data /path/to/imagenet/ --batch-size 2048 --name our --dist-port 8873 --mode neuralef --learning-rate 2.4 --alpha 0.0025 --proj_dim 8192 8192 {--no_stop_grad} --epochs {100, 200, 400}; python linear_probe.py --data /path/to/imagenet/ --pretrained logs/our/final.pth
```
When using `--proj_dim 4096 4096`, set `--alpha 0.005`; when using `--proj_dim 2048 2048`, set `--alpha 0.01`. If setting a different batch size from 2048, remember to linearly scale the learning rate.

Evaluate on COCO
```
cd detection;
python convert-pretrain-to-detectron2.py ../logs/neuralef/final.pth output.pkl
python train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml --num-gpus 8 MODEL.WEIGHTS ./output.pkl SOLVER.BASE_LR 0.05
```

Perform spectral hashing
```
python main_hash.py --data /path/to/imagenet/ --batch-size 2024 --name hashing --dist-port 8873 --mode neuralef --learning-rate 2.4 --alpha 0.01 --proj_dim 2048 2048 --t 30
```

### Evaluation in node property prediction
Training:
```
python main_products.py  --root /path/to/ogbn --device 3 --model res_mlp --hidden_channels 2048 --num_layers 12 --proj_dim 8192 8192 --alpha 0.3 --lr 0.3 --weight_decay 0 --batch_size 16384 --epochs 20 --no_stop_grad --output_dir logs/products_al.3_lr.3_w2048_nosg; 
```

Linear probe:
```
python main_products.py --root /path/to/ogbn --device 3 --model res_mlp --hidden_channels 2048 --num_layers 12 --proj_dim 8192 8192 --alpha 0.3 --lr 0.3 --weight_decay 0 --batch_size 16384 --epochs 20 --no_stop_grad --output_dir logs/products_al.3_lr.3_w2048_nosg  --ft_only --ft_mode freeze --ft_lr 0.01 --ft_weight_decay 1e-3 --ft_smoothing 0.1 --runs 10
```
