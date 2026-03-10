
cd /ssd/0/lsy/SMF_BSN/pycharm_proj1/0/DBSN-master/dbsn_gray

python gray_DCSW_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --Run_description 'dbsnCBF_Dv6SWNA_51_0_AP2' \
        --device_ids  0,2 \
        --trainset 'Darkv6NA_don_0_300,Darkv6NA_acc_0_300' \
        --valset 'Darkv6NA_don_0_300_val,Darkv6NA_acc_0_300_val' \
        --input_channel 5 \
        --output_channel 1 \
        --Loss_choice L2_video \
        --epoch 40 \
        --bsn_ver dbsn_fuse \
        --weight One \
        --load_thread 4 \
        --dynamic_load True \


python gray_DCSW_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --Run_description 'dbsnCB2_Dv6SWNA_31_0300_AP2' \
        --device_ids  6,7 \
        --trainset 'Darkv6NA_don_0_300,Darkv6NA_acc_0_300' \
        --valset 'Darkv6NA_don_0_300_val,Darkv6NA_acc_0_300_val' \
        --input_channel 3 \
        --output_channel 1 \
        --Loss_choice L2_video \
        --epoch 90 \
        --bsn_ver dbsn_DC \
        --weight One \
        --load_thread 4 \
        --dynamic_load True \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsnCB2_Dv6SWNA_31_0300_AP2/dbsn_gray_Noisy_data_ckpt_e17.pth

python gray_DCSW_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --Run_description 'dbsnCBF_Dv6SWNA_31_0_AP2_ACLs' \
        --device_ids  1 \
        --trainset 'Darkv6NA_don_0_300,Darkv6NA_acc_0_300' \
        --valset 'Darkv6NA_don_0_300_val,Darkv6NA_acc_0_300_val' \
        --input_channel 3 \
        --output_channel 1 \
        --Loss_choice L2_video_AC \
        --epoch 40 \
        --bsn_ver dbsn_fuse \
        --weight One \
        --load_thread 4 \
        --dynamic_load True \

python gray_dualchan_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --Run_description 'dbsn3DCBF_unetskip_SCMOS723NA_31_S14_2binned_P128' \
        --device_ids  6,7 \
        --trainset 'SCMOS723_NA3f_don_S14_2binned,SCMOS723_NA3f_acc_S14_2binned' \
        --valset 'SCMOS723_NA_don_S13_2binned_val,SCMOS723_NA_acc_S13_2binned_val' \
        --input_channel 3 \
        --output_channel 1 \
        --Loss_choice L2_video \
        --epoch 90 \
        --bsn_ver dbsn_fuse \
        --weight One \
        --load_thread 4 \
        --dynamic_load True \
        --blindspot_conv_type Mask3D \
        --Unet_skip True \
        --patch_size 128 \

python gray_dualchan_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --Run_description 'dbsn3DCBF_unetskip_SCMOS1112NA_31_1203+1_AP2' \
        --device_ids  1,4 \
        --trainset 'SCMOS1112_NA3f_don_1203+1,SCMOS1112_NA3f_acc_1203+1' \
        --valset 'SMOCS1104_NA_don_12030003val,SMOCS1104_NA_acc_12030003val' \
        --input_channel 3 \
        --output_channel 1 \
        --Loss_choice L2_video \
        --epoch 40 \
        --bsn_ver dbsn_fuse \
        --weight One \
        --load_thread 4 \
        --dynamic_load True \
        --blindspot_conv_type Mask3D \
        --Unet_skip True


python gray_dualchan_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --Run_description 'dbsn3DCBF_unetskip_SCMOS1104SWNA_31_1203+1_AP2' \
        --device_ids  5,6,7 \
        --trainset 'SCMOS1104_SWNA3f_don_1203_1,SCMOS1104_SWNA3f_acc_1203_1' \
        --valset 'SMOCS1104_NA_don_12030003val,SMOCS1104_NA_acc_12030003val' \
        --input_channel 3 \
        --output_channel 1 \
        --Loss_choice L2_video \
        --epoch 40 \
        --bsn_ver dbsn_fuse \
        --weight One \
        --load_thread 4 \
        --dynamic_load True \
        --blindspot_conv_type Mask3D \
        --Unet_skip True

python gray_dualchan_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --Run_description 'dbsn3DCBF_unetskip_D1112NA_31_AP2' \
        --device_ids  2,3,4 \
        --trainset 'Dark1112_NA3f_don,Dark1112_NA3f_acc' \
        --valset 'Dark1112_NA_don_val,Dark1112_NA_acc_val' \
        --input_channel 3 \
        --output_channel 1 \
        --Loss_choice L2_video \
        --epoch 40 \
        --bsn_ver dbsn_fuse \
        --weight One \
        --load_thread 4 \
        --dynamic_load True \
        --blindspot_conv_type Mask3D \
        --Unet_skip True

python gray_dualchan_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --Run_description 'dbsn3DCBF_unetskip_Dv7SWNA_31_050010_AP2' \
        --device_ids  0,1,2,3 \
        --trainset 'Darkv7SWNA_don_0_500_10,Darkv7SWNA_acc_0_500_10' \
        --valset 'Darkv7NA_don_0_500_val,Darkv7NA_acc_0_500_val' \
        --input_channel 3 \
        --output_channel 1 \
        --Loss_choice L2_video \
        --epoch 40 \
        --bsn_ver dbsn_fuse \
        --weight One \
        --load_thread 4 \
        --dynamic_load True \
        --blindspot_conv_type Mask3D \
        --Unet_skip True

python gray_dualchan_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --Run_description 'dbsn3DCBF_unetskip_Dv7SWNA_31_0500_AP2' \
        --device_ids  2,3 \
        --trainset 'Darkv7SWNA_don_0_500,Darkv7SWNA_acc_0_500' \
        --valset 'Darkv7NA_don_0_500_val,Darkv7NA_acc_0_500_val' \
        --input_channel 3 \
        --output_channel 1 \
        --Loss_choice L2_video \
        --epoch 40 \
        --bsn_ver dbsn_fuse \
        --weight One \
        --load_thread 4 \
        --dynamic_load True \
        --blindspot_conv_type Mask3D \
        --Unet_skip True

python gray_dualchan_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --Run_description 'dbsn3DCBF_unetskip_Dv7SWNA_31_0500_AP2' \
        --device_ids  2,3 \
        --trainset 'Darkv7SWNA_don_0_500,Darkv7SWNA_acc_0_500' \
        --valset 'Darkv7NA_don_0_500_val,Darkv7NA_acc_0_500_val' \
        --input_channel 3 \
        --output_channel 1 \
        --Loss_choice L2_video \
        --epoch 40 \
        --bsn_ver dbsn_fuse \
        --weight One \
        --load_thread 4 \
        --dynamic_load True \
        --blindspot_conv_type Mask3D \
        --Unet_skip True

python gray_dualchan_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --Run_description 'dbsn3DCBF_unetskip_Dv7SWNA_31_0500val_AP2' \
        --device_ids  4 \
        --trainset 'Darkv7SWNA_don_0_500_val,Darkv7SWNA_acc_0_500_val' \
        --valset 'Darkv7NA_don_0_500_val,Darkv7NA_acc_0_500_val' \
        --input_channel 3 \
        --output_channel 1 \
        --Loss_choice L2_video \
        --epoch 40 \
        --bsn_ver dbsn_fuse \
        --weight One \
        --load_thread 4 \
        --dynamic_load True \
        --blindspot_conv_type Mask3D \
        --Unet_skip True

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  0,1,2 \
        --trainset 'Dark1104_SWNA3f_acc25,Dark1104_SWNA3f_acc25' \
        --valset 'Dark1104_NA_acc25_1203val,Dark1104_NA_acc25_1203val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 40 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --Run_description 'dbsn3DCB_Dark1104SWNA_31_acc25_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \
        --load_thread 4 \

export WANDB_BASE_URL=https://api.bandw.top


