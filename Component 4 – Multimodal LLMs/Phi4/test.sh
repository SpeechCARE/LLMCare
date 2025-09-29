python test.py \
    --model_dir "./phi4_ad_90sec_lr1e-5" \
    --test_csv_path "csv_files/test.csv" \
    --test_audio_dir "dementiabank-denoise/test/All-test/" \
    --output_dir "./test_results_90sec_lr1e-5" \
    --max_test_samples 100 \
    --batch_size 1 \
    --max_audio_seconds 60 \
    --use_flash_attention \
    --mixed_precision bf16