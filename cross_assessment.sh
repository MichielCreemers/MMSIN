python cross_assessment.py --dataset "SJTU" \
                        --model "ckpts/['WPC']_4_6_best_model.pth"\
                        --projections_dir "SJTU/projections"\
                        --mos_data_path "SJTU/SJTU_wpc_scaled.csv" \
                        --nss_path "SJTU/SJTU_NSS_non_scaled.csv" \
                        --batch_size "1"\
                        --number_projections "6"\
                        --minmax_path "WPC/scaler_params.npy"