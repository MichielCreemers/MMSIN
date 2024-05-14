python cross_assessment.py --dataset "SJTU" \
                        --model "ckpts/['WPC2']_2_best_model.pth"\
                        --projections_dir "SJTU/projections"\
                        --mos_data_path "SJTU/SJTU_wpc_scaled.csv" \
                        --nss_path "SJTU/SJTU_NSS.csv" \
                        --batch_size "1"\
                        --number_projections "6"\
                        --minmax_path "WPC2/scaler_params.npy"