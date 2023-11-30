python /home/v-detaixin/librilight_process/auto_asr.py \
    --in_dir="/home/v-detaixin/librilight/small_cut" \
    --out_dir="/home/v-detaixin/librilight/small_processed" \
    --device_id="1" \
    --export_chunk_len=7500 \
    --min_silence_len=500 \
    --keep_silence=500 \
    --spk_num_start=0 \
    --spk_num_end=10 \