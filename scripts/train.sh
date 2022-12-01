ACTION=$1

python train_motion.py --log_dir ./outputs/ --batch_size 24 --num_epoch 150 --pretrained_scene_model ./outputs/POINTTRANS_C_32768/model.pth --use_color --npoints 32768 --num_workers 4 --z_size 256 --weight_loss_ground 0.1 --weight_loss_rec_pose 10.0 --weight_loss_rec_vertex 10.0 --weight_loss_kl 0.1 --action "${ACTION}"