


python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  2 \
        --trainset 'Darkv4_5f_acc_0,Darkv4_5f_acc_0' \
        --valset 'Darkv4_acc_0HS_10_test,Darkv4_acc_0HS_10_test' \
        --input_channel 5 \
        --output_channel 1 \
        --Loss_choice L2_video \
        --epoch 90 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --data_rate 0.25 \
        --Run_description 'dbsnCB_Dv4_51_0Kacc_AP2_P25' \
        --lr_dbsn 1e-4

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume continue \
        --device_ids  1 \
        --trainset 'Darkv5noalign_3f_acc_0,Darkv5noalign_3f_acc_0' \
        --valset 'Darkv5noalign_acc_0_test,Darkv5noalign_acc_0_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --data_rate 1 \
        --Run_description 'dbsnCB_Dv5NA_31_0Kacc_AP2' \
        --lr_dbsn 1e-4 \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsnCB_Dv5NA_31_0Kacc_AP2/dbsn_gray_Noisy_data_ckpt_e6.pth

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume continue \
        --device_ids  2 \
        --trainset 'Darkv5noalign_3f_don_0,Darkv5noalign_3f_don_0' \
        --valset 'Darkv5noalign_don_0_test,Darkv5noalign_don_0_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --data_rate 1 \
        --Run_description 'dbsnCB_Dv5NA_31_0Kdon_AP2' \
        --lr_dbsn 1e-4 \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsnCB_Dv5NA_31_0Kdon_AP2/dbsn_gray_Noisy_data_ckpt_e6.pth


python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  1 \
        --trainset 'Darkv4_15f_don_0,Darkv4_15f_don_0' \
        --valset 'Darkv4_don_0_test,Darkv4_don_0_test' \
        --input_channel 15 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver 2stage_DBSN \
        --weight One \
        --dynamic_load True \
        --load_thread 8 \
        --patch_size 96\
        --Run_description 'BiConvlstmDBSN_Dv4_151_0Kdon_AP2' \
        --lr_dbsn 1e-4\

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  1 \
        --trainset 'Darkv5noalign_3f_don_0,Darkv5noalign_3f_don_0' \
        --valset 'Darkv5noalign_don_0_test,Darkv5noalign_don_0_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver 2stage_DBSN \
        --weight One \
        --dynamic_load True \
        --load_thread 8 \
        --patch_size 96\
        --Run_description 'BiConvlstmDBSN_Dv5NA_31_0Kdon_AP2' \
        --lr_dbsn 1e-4\

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  1 \
        --trainset 'Darkv5noalign_3f_don_0,Darkv5noalign_3f_don_0' \
        --valset 'Darkv5noalign_don_0_test,Darkv5noalign_don_0_test' \
        --input_channel 5 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver 2stage_DBSN \
        --weight One \
        --dynamic_load True \
        --load_thread 8 \
        --patch_size 96\
        --Run_description 'BiConvlstmDBSN_Dv5NA_31_0Kdon_AP2' \
        --lr_dbsn 1e-4\

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  3 \
        --trainset 'Darkv4&5noalign_3f_don_0,Darkv4&5noalign_3f_don_0' \
        --valset 'Darkv5noalign_don_0_test,Darkv5noalign_don_0_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --data_rate 0.5 \
        --Run_description 'dbsnCB_Dv4&5NA_31_0Kdon_AP2_p50' \
        --lr_dbsn 1e-4 \



python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  0 \
        --trainset 'Darkv4&5noalign_3f_don_0,Darkv4&5noalign_3f_don_0' \
        --valset 'Darkv5noalign_don_0_test,Darkv5noalign_don_0_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --data_rate 0.5 \
        --no_flip\
        --Run_description 'dbsnCB_Dv4&5NA_31_0Kdon_AP2_p50_noaug' \
        --lr_dbsn 1e-4 \

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  4 \
        --trainset 'Darkv4&5noalign_5f_don_0,Darkv4&5noalign_5f_don_0' \
        --valset 'Darkv5noalign_don_0_test,Darkv5noalign_don_0_test' \
        --input_channel 5 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --data_rate 0.5 \
        --Run_description 'dbsnCB_Dv4&5NA_51_0Kdon_AP2_p50' \
        --lr_dbsn 1e-4 \


python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume continue \
        --device_ids  0 \
        --trainset 'Darkv5NA_3f_don_0HS_25,Darkv5NA_3f_don_0HS_25' \
        --valset 'Darkv5NA_don_0HS_25_test,Darkv5NA_don_0HS_25_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 20 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsnCB_Dv5NA_31_FT_P180_0HS25don' \
        --lr_dbsn 1e-4 \
        --patch_size 180\
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsnCB_Dv5NA_31_0Kdon_AP2/dbsn_gray_Noisy_data_ckpt_e10.pth

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume continue \
        --device_ids  6 \
        --trainset 'Darkv5NA_3f_don_0_25,Darkv5NA_3f_don_0_25' \
        --valset 'Darkv5NA_don_0_25_test,Darkv5NA_don_0_25_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsnCB_Dv5NA_31_FT_P160_0_25don_lr1e5' \
        --lr_dbsn 1e-5 \
        --patch_size 160\
        --update_opt 0\
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsnCB_Dv5NA_31_0Kdon_AP2/dbsn_gray_Noisy_data_ckpt_e10.pth

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  0 \
        --trainset 'Simu_adj_3f,Simu_adj_3f' \
        --valset 'Simu_adj,Simu_adj' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_DSimuadj_31_AP2' \
        --lr_dbsn 1e-4 \
        --patch_size 96\
        --blindspot_conv_type Mask3D \
       
python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  0 \
        --trainset 'Simu_adj_don_3f_S12_No100,Simu_adj_don_3f_S12_No100' \
        --valset 'Simu_adj_don_S12_No100,Simu_adj_don_S12_No100' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_DSimuadjdon_S12No100_31_AP2_R6' \
        --lr_dbsn 1e-4 \
        --patch_size 96\
        --blindspot_conv_type Mask3D \
        --repeat 6 \

python gray_unpair_pretrain_mu.py\
         --isTrain True \
         --resume new \
         --device_ids  3 \
         --trainset 'Simu_adj_don_3f_S12,Simu_adj_don_3f_S12' \
         --valset 'Simu_adj_don_S12,Simu_adj_don_S12' \
         --input_channel 3 \
         --output_channel 1 \
         --epoch 30 \
         --bsn_ver dbsn_centrblind \
         --weight One \
         --dynamic_load True \
         --load_thread 4 \
         --Run_description 'dbsn3DCB_DSimuadjdon_S12_31_AP2_R6' \
         --lr_dbsn 1e-4 \
         --patch_size 96\
         --blindspot_conv_type Mask3D \
         --repeat 6 \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  2 \
        --trainset 'Simu_adj_don,Simu_adj_don' \
        --valset 'Simu_adj_don,Simu_adj_don' \
        --input_channel 9 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 1 \
        --Run_description 'dbsn3DCB_DSWSimuadj_91_AP2' \
        --lr_dbsn 1e-4 \
        --patch_size 96\
        --blindspot_conv_type Mask3D \

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  7 \
        --trainset 'Simu_3f,Simu_3f' \
        --valset 'Simu,Simu' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_DSimu_31_AP2' \
        --lr_dbsn 1e-4 \
        --patch_size 96\
        --blindspot_conv_type Mask3D \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  3 \
        --trainset 'Darkv5NA_don_0HS_test,Darkv5NA_don_0HS_test' \
        --valset 'Darkv5NA_don_0HS_test,Darkv5NA_don_0HS_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 90 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 2 \
        --Run_description 'dbsn3DCB_Dv5NA_31_0KdonHStest_AP2_FRshuffle' \
        --lr_dbsn 1e-4 \
        --batch_size 8 \
        --patch_size 96\
        --Frame_shuffle True \
        --blindspot_conv_type Mask3D \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  4 \
        --trainset 'Darkv5NA_don_0HS_test,Darkv5NA_don_0HS_test' \
        --valset 'Darkv5NA_don_0HS_test,Darkv5NA_don_0HS_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver DBSN_AttCB \
        --weight One \
        --dynamic_load True \
        --load_thread 2 \
        --Run_description 'dbsn_Dv5NA_31_0KdonHStest_AP2' \
        --lr_dbsn 1e-4 \
        --batch_size 8 \
        --patch_size 96\
        --Frame_shuffle True \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  0 \
        --trainset 'Darkv4NA_don_0HS_test,Darkv4NA_don_0HS_test' \
        --valset 'Darkv4NA_don_0HS_test,Darkv4NA_don_0HS_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 100 \
        --bsn_ver dbsn_centrblind\
        --weight One \
        --dynamic_load True \
        --load_thread 2 \
        --Run_description 'dbsn3DCB_Dv4SWNA_31_0KdonHStest_AP2' \
        --lr_dbsn 1e-4 \
        --batch_size 8 \
        --patch_size 96\
        --blindspot_conv_type Mask3D \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  1 \
        --trainset 'Darkv5noalign_don_0_test,Darkv5noalign_don_0_test' \
        --valset 'Darkv5NA_don_0HS_test,Darkv5NA_don_0HS_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 100 \
        --bsn_ver dbsn_centrblind\
        --weight One \
        --dynamic_load True \
        --load_thread 2 \
        --Run_description 'dbsn3DCB_Dv5SWNA_31_0Kdontest_AP2' \
        --lr_dbsn 1e-4 \
        --batch_size 8 \
        --patch_size 96\
        --blindspot_conv_type Mask3D \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  0 \
        --trainset 'MIV_626,MIV_626' \
        --valset 'MIV_v1_test,MIV_v1_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind\
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_Dv5SW_31_MIV626_P160' \
        --lr_dbsn 1e-4 \
        --batch_size 8 \
        --patch_size 160\
        --blindspot_conv_type Mask3D \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  1 \
        --trainset 'clc7-ms3_701,clc7-ms3_701' \
        --valset 'MIV_v1_test,MIV_v1_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind\
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_Dv5SW_31_clc7ms3701_P160' \
        --lr_dbsn 1e-4 \
        --batch_size 8 \
        --patch_size 160\
        --blindspot_conv_type Mask3D \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  3 \
        --trainset 'Darkv4NA_don_0HS_test,Darkv4NA_don_0HS_test' \
        --valset 'Darkv4NA_don_0HS_test,Darkv4NA_don_0HS_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 100 \
        --bsn_ver dbsnl_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 2 \
        --Run_description 'dbsnL3DCB_Dv4SWNA_31_0KdonHStest_AP2' \
        --lr_dbsn 1e-4 \
        --patch_size 96\
        --blindspot_conv_type Mask3D \

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  3 \
        --trainset 'Darkv4&5noalign_3f_acc_0,Darkv4&5noalign_3f_acc_0' \
        --valset 'Darkv4NA_acc_0HS_test,Darkv4NA_acc_0HS_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 40 \
        --bsn_ver dbsnl_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsnL3DCB_Dv4+5NA_31_0Kacc_AP2_Frshuffle' \
        --lr_dbsn 1e-4 \
        --patch_size 96\
        --blindspot_conv_type Mask3D \
        --Frame_shuffle True \

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  4 \
        --trainset 'Darkv4&5noalign_3f_acc_0,Darkv4&5noalign_3f_acc_0' \
        --valset 'Darkv4NA_acc_0HS_test,Darkv4NA_acc_0HS_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 40 \
        --bsn_ver dbsnl_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsnL3DCB_Dv4+5NA_31_0Kacc_AP2_Frshuffle_P50' \
        --lr_dbsn 1e-4 \
        --patch_size 96\
        --data_rate 0.5 \
        --blindspot_conv_type Mask3D \
        --Frame_shuffle True \

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  7 \
        --trainset 'Darkv4&5noalign_3f_don_0,Darkv4&5noalign_3f_don_0' \
        --valset 'Darkv4&5noalign_don_0_test,Darkv4&5noalign_don_0_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_Dv4+5NA_31_0Kdon_AP2' \
        --lr_dbsn 1e-4 \
        --patch_size 96\
        --blindspot_conv_type Mask3D \

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  6 \
        --trainset 'Darkv5NA_3f_acc_0HS_10,Darkv5NA_3f_acc_0HS_10' \
        --valset 'Darkv5NA_acc_0HS_10_test,Darkv5NA_acc_0HS_10_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_Dv5NA_31_0KaccHS10_AP2' \
        --lr_dbsn 1e-4 \
        --patch_size 96\
        --blindspot_conv_type Mask3D \

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume continue \
        --device_ids  3 \
        --trainset 'Darkv4&5noalign_3f_acc_0,Darkv4&5noalign_3f_acc_0' \
        --valset 'Darkv5NA_acc_0HS_10_test,Darkv5NA_acc_0HS_10_test' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_Dv4+5NA_31_0Kacc_AP2_Parrel' \
        --lr_dbsn 1e-4 \
        --patch_size 96\
        --blindspot_conv_type Mask3D \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_Dv4+5NA_31_0Kacc_AP2/dbsn_gray_Noisy_data_ckpt_e5.pth
export WANDB_BASE_URL=https://api.bandw.top 
python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  1 \
        --trainset 'SCMOS723_SWNA3f_acc_50,SCMOS723_SWNA3f_acc_50' \
        --valset 'SCMOS_SWNA3f_acc_50_val,SCMOS_SWNA3f_acc_50_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_SCMOS723_31_acc50SW_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  6,7 \
        --trainset 'SCMOS915_NA3f_acc_500_2binned,SCMOS915_NA3f_acc_500_2binned' \
        --valset 'SCMOS825_NA_acc_50_2binned_val,SCMOS825_NA_acc_50_2binned_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --Run_description 'dbsn3DCB_SCMOS915NA_31_0acc5002bin_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \
        --load_thread 4 \

  
python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  0 \
        --trainset 'Simu_0728_NA_acc_F104,Simu_0728_NA_acc_F104' \
        --valset 'Simu_0728_NA_acc_val,Simu_0728_NA_acc_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --Run_description 'dbsn3DCB_Dv728SimuNA_31_accF104_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \
        --middle_channel 96 \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  1 \
        --trainset 'Simu_0728_NA_acc_F105,Simu_0728_NA_acc_F105' \
        --valset 'Simu_0728_NA_acc_val,Simu_0728_NA_acc_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --Run_description 'dbsn3DCB_Dv728SimuNA_31_accF105_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \
        --middle_channel 96 \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  2 \
        --trainset 'Simu_0728_NA_acc_F106,Simu_0728_NA_acc_F106' \
        --valset 'Simu_0728_NA_acc_val,Simu_0728_NA_acc_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --Run_description 'dbsn3DCB_Dv728SimuNA_31_accF106_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \
        --middle_channel 96 \

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  2,7 \
        --trainset 'SCMOS1112_NA3f_acc_1203+1,SCMOS1112_NA3f_acc_1203+1' \
        --valset 'SCMOS1022_NA3f_acc_val,SCMOS1022_NA3f_acc_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 40 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --Run_description 'dbsn3DCB_SCMOS1112NA_0Kacc_1203+1_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \
        --load_thread 4 \



        # --br1_block_num 14 \
        # --br2_block_num 14 \

python gray_unpair_train.py\
        --isTrain True \
        --resume new \
        --device_ids  0,1,2 \
        --trainset 'SCMOS915_NA3f_acc_500_2binned1406,SCMOS915_NA3f_acc_500_2binned1406' \
        --valset 'SCMOS915_NA_acc_500_2binned1406_val,SCMOS915_NA_acc_500_2binned1406_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 60 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_Sigma_SCMOS915NA_31_0acc5002bin1406_AP2' \
        --blindspot_conv_type Mask3D\
        --init_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_SCMOS915NA_31_0acc5002bin1406_AP2/dbsn_gray_Noisy_data_ckpt_e39.pth\
        --Loss_choice  DBSN_video \
        --lr_sigma_mu 1e-5\
        --lr_sigma_n 1e-5 \
        # --lr_dbsn 1e-4 \

python gray_SW_sigma_test.py\
        --Real_dataset_mode True \
        --Run_description 'dbsn3DCB_Sigma_SCMOS915NA_31_0acc5002bin1406_AP2' \
        --isTrain False \
        --valset 'SCMOS915_NA_acc_500_2binned1406_val,SCMOS915_NA_acc_500_2binned1406_val' \
        --noise_type 'Noisy_data' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_centrblind \
        --weight One\
        --device_ids 2 \
        --blindspot_conv_type Mask3D \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/g_train_dbsnloss__Noisy_datadbsn3DCB_Sigma_SCMOS915NA_31_0acc5002bin1406_AP2/dbsn_gray_Noisy_data_ckpt_e57.pth
  

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  5 \
        --trainset 'Simu_0728_NA_acc_F108,Simu_0728_NA_acc_F108' \
        --valset 'Simu_0728_NA_acc_val,Simu_0728_NA_acc_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --Run_description 'dbsn3DCB_Dv728SimuNA_31_accF108_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \
        --middle_channel 96 \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  5,6 \
        --trainset 'Darkv7NA_acc_0_500_10,Darkv7NA_acc_0_500_10' \
        --valset 'Darkv7NA_acc_0_500_val,Darkv7NA_acc_0_500_val' \
        --input_channel 1 \
        --output_channel 1 \
        --epoch 50 \
        --bsn_ver dbsn \
        --weight One \
        --dynamic_load True \
        --Run_description 'dbsn_Dv7NA_11_0acc50010_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask \

python gray_unpair_SW_pretrain.py\
        --isTrain True \
        --resume new \
        --device_ids  0,5 \
        --trainset 'Celldata_909,Celldata_909' \
        --valset 'Celldata_908_SA_val,Celldata_908_SA_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 30 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --Run_description 'dbsn3DCB_Cell909_31_P128' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \
        --patch_size 128\

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  5,6,7 \
        --trainset 'Dark1112_NA3f_acc,Dark1112_NA3f_acc' \
        --valset 'Dark1112_NA_acc_val,Dark1112_NA_acc_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 40 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_Dark1112NA_31_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  4,5,6,7 \
        --trainset 'Cell1221_NA3f_hekkd-BHQ,Cell1221_NA3f_hekkd-BHQ' \
        --valset 'Cell1121_hekkd_val,Cell1121_hekkd_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 40 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_Cell1221_hekkd-BHQNA_31_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \

python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  0,1,2,3 \
        --trainset 'Cell1221_NA3f_hekkd-mng-control,Cell1221_NA3f_hekkd-mng-control' \
        --valset 'Cell1121_hekkd_val,Cell1121_hekkd_val' \
        --input_channel 3 \
        --output_channel 1 \
        --epoch 40 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --load_thread 4 \
        --Run_description 'dbsn3DCB_Cell1221_hekkd-mng-controlNA_31_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \
        
python gray_unpair_pretrain_mu.py\
        --isTrain True \
        --resume new \
        --device_ids  2,3\
        --trainset 'SCMOS1104_NA3f_acc_1203_1,SCMOS1104_NA3f_acc_1203_1' \
        --valset 'SCMOS1022_NA3f_acc_val,SCMOS1022_NA3f_acc_val'\
        --input_channel 3 \
        --output_channel 1 \
        --epoch 40 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --dynamic_load True \
        --Run_description 'dbsn3DCB_SCMOS1104NA_31_0acc1203+1_AP2' \
        --lr_dbsn 1e-4 \
        --blindspot_conv_type Mask3D \

python gray_slidewindow_test.py \
        --Run_description 'dbsn3DCB_Inceptionv2_Dark915NA_31_0Kacc6.6_AP2_midchan160' \
        --isTrain False \
        --valset 'Dark915_NA_acc_6.6,Dark915_NA_acc_6.6' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_centrblind \
        --weight One\
        --device_ids 1\
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_Inceptionv2_Dark915NA_31_0Kacc6.6_AP2_midchan160/dbsn_gray_Noisy_data_ckpt_e33.pth\
        --blindspot_conv_type Mask3D \
        --middle_channel 160 \
        # --br1_block_num 14 \
        # --br2_block_num 14 \



python gray_slidewindow_test.py \
        --Real_dataset_mode True \
        --Run_description 'dbsn3DCB_SCMOS1112NA_0Kacc_1203+1_AP2' \
        --isTrain False \
        --valset 'SCMOS1112_NA_acc_1203+1,SCMOS1112_NA_acc_1203+1' \
        --noise_type 'Noisy_data' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_centrblind \
        --weight One\
        --device_ids 6 \
        --blindspot_conv_type Mask3D \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_SCMOS1112NA_0Kacc_1203+1_AP2/dbsn_gray_Noisy_data_ckpt_e39.pth \



python gray_slidewindow_test.py \
        --Run_description 'dbsn3DCB_Mask+_SCMOS1010NA_31_0acc_AP2' \
        --isTrain False \
        --valset 'SCMOS1022_NA_acc,SCMOS1022_NA_acc' \
        --noise_type 'Noisy_data' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_centrblind \
        --weight One\
        --blindspot_conv_type Mask3D \
        --device_ids 2 \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_Mask+_SCMOS1010NA_31_0acc_AP2/dbsn_gray_Noisy_data_ckpt_e61.pth\
        --mask_shape '+' \

python gray_slidewindow_test.py \
        --Run_description 'dbsn3DCB_Dark1112NA_31_AP2' \
        --isTrain False \
        --valset 'Dark1112_NA_acc,Dark1112_NA_acc' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_centrblind \
        --weight One\
        --device_ids 5 \
        --blindspot_conv_type Mask3D \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_Dark1112NA_31_AP2/dbsn_gray_Noisy_data_ckpt_e39.pth

python gray_slidewindow_test.py \
        --Run_description 'dbsn3DCB_Cell1221_hekkd-mng-controlNA_31_AP2' \
        --isTrain False \
        --valset 'Cell1221_hekkd-mng-control,Cell1221_hekkd-mng-control' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_centrblind \
        --weight One\
        --device_ids 6 \
        --blindspot_conv_type Mask3D \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_Cell1221_hekkd-mng-controlNA_31_AP2/dbsn_gray_Noisy_data_ckpt_e39.pth

python gray_slidewindow_test.py \
        --Run_description 'dbsn_Dv7NA_11_0acc50010_AP2' \
        --isTrain False \
        --valset 'Darkv7NA_acc_0_500_10,Darkv7NA_acc_0_500_10' \
        --input_channel 1 \
        --output_channel 1 \
        --bsn_ver dbsn \
        --weight One\
        --device_ids 6 \
        --blindspot_conv_type Mask \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn_Dv7NA_11_0acc50010_AP2/dbsn_gray_Noisy_data_ckpt_e49.pth

python gray_slidwin_dual_test.py \
        --Run_description 'dbsn3DCBF_unetskip_Dv7SWNA_31_050010_AP2' \
        --isTrain False \
        --valset 'Darkv7NA_don_0_500_10,Darkv7NA_acc_0_500_10' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_fuse \
        --weight One\
        --device_ids 0 \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCBF_unetskip_Dv7SWNA_31_050010_AP2/dbsn_gray_Noisy_data_ckpt_e39.pth \
        --Unet_skip True \
        --blindspot_conv_type Mask3D \


python gray_slidwin_dual_test.py \
        --Run_description 'dbsn3DCBF_unetskip_D1022NA_31_AP2' \
        --isTrain False \
        --valset 'Dark1022_NA_don,Dark1022_NA_acc' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_fuse \
        --weight One\
        --device_ids 1 \
        --Unet_skip True \
        --blindspot_conv_type Mask3D \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCBF_unetskip_D1022NA_31_AP2/dbsn_gray_Noisy_data_ckpt_e42.pth

python gray_slidewindow_test.py\
        --Run_description 'dbsn3DCB_Dv5NA_31_0KdonHStest_AP2_FRshuffle' \
        --isTrain False \
        --resume new \
        --device_ids  4 \
        --valset 'Darkv5NA_don_0HS_test,Darkv5NA_don_0HS_test' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_centrblind \
        --weight One \
        --blindspot_conv_type Mask3D \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_Dv5NA_31_0KdonHStest_AP2_FRshuffle/dbsn_gray_Noisy_data_ckpt_e19.pth



python gray_slidewindow_test.py \
        --Real_dataset_mode True \
        --Run_description 'dbsn3DCB_Dv5NA_31_0KdonHStest_AP2_FRshuffle' \
        --isTrain False \
        --valset 'Darkv5NA_don_0HS_test,Darkv5NA_don_0HS_test' \
        --noise_type 'Noisy_data' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_centrblind \
        --weight One\
        --device_ids 7 \
        --blindspot_conv_type Mask3D \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_Dv5NA_31_0Kdon_AP2/dbsn_gray_Noisy_data_ckpt_e28.pth


        --last_ckpt /hdd/0/lsy/DBSN_ckpts/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_Dv5NA_91_0Kdon_AP2/dbsn_gray_Noisy_data_ckpt_e29.pth

python gray_slidewindow_test.py \
        --Run_description 'dbsn3DCB_Dv7SWNA_31_0Kacc1_AP2' \
        --isTrain False \
        --valset 'Darkv7NA_acc_0_1,Darkv7NA_acc_0_1' \
        --noise_type 'Noisy_data' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_centrblind \
        --weight One\
        --blindspot_conv_type Mask3D \
        --device_ids 3 \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_Dv7SWNA_31_0Kacc1_AP2/dbsn_gray_Noisy_data_ckpt_e39.pth\


python gray_slidewindow_test.py \
        --Run_description 'dbsn3DCB_Dv7SWNA_31_0Kacc500_AP2' \
        --isTrain False \
        --valset 'Darkv7SWNA_acc_0_500,Darkv7SWNA_acc_0_500' \
        --noise_type 'Noisy_data' \
        --input_channel 3 \
        --output_channel 1 \
        --bsn_ver dbsn_centrblind \
        --weight One\
        --blindspot_conv_type Mask3D \
        --device_ids 3 \
        --last_ckpt /ssd/0/lsy/DBSN_0327bbncz/dual_ckpts/dbsn_gray_Noisy_data_Noisy_datadbsn3DCB_Dv7SWNA_31_0Kacc500_AP2/dbsn_gray_Noisy_data_ckpt_e30.pth\







