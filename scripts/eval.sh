STAMP=$1

python eval_motion.py --pretrained_scene_model ./outputs/POINTTRANS_C_32768/model.pth --log_dir ./outputs/ --npoints 32768 --use_color --batch_size 1 --z_size 256 --stamp ${STAMP} --action "*"