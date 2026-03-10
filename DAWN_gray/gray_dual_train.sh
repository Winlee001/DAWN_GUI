
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



