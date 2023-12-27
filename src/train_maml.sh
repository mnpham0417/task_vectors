python3 train_maml.py \
    --arch mlp \
    --dataset cifar100 \
    --target_class 0 \
    --num_epoch 20 \
    --pretrained_checkpoint "model_mlp_cifar100.pth" --model_name "model_mlp_cifar100_maml_0_20ep_1.1a_0.2cw.pth"


