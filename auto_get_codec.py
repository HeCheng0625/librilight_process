from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import torch
import os
import numpy as np
import argparse
from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build the statistics on LibriBig")
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--device_id', type=int)
    parser.add_argument('--spk_num_start', type=int, default=0)
    parser.add_argument('--spk_num_end', type=int, default=100)
    args = parser.parse_args()

    device = "cuda:{}".format(str(args.device_id))

    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)

    def extract_encodec_token(wav_path):

        wav, sr = torchaudio.load(wav_path)
        wav = convert_audio(wav, sr, model.sample_rate, model.channels)
        wav = wav.unsqueeze(0)
        wav = wav.to(device)
        with torch.no_grad():
            encoded_frames = model.encode(wav)
            codes_ = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
            codes = codes_.cpu().numpy()[0,:,:].T # [T, 8]
            
            return codes

    input_dir = args.in_dir
    out_dir = args.out_dir

    for spk_id in tqdm(sorted(os.listdir(input_dir))[args.spk_num_start: args.spk_num_end]):
        print("speaker:", spk_id)
        spk_path = os.path.join(input_dir, spk_id)
        if len(os.listdir(spk_path)) == 0:
            continue
        for chapter_id in sorted(os.listdir(spk_path)):
            print("chapter:", chapter_id)
            chapter_path = os.path.join(spk_path, chapter_id)
            if len(os.listdir(chapter_path)) == 0:
                continue
            for wav_name in sorted(os.listdir(chapter_path)):
                if not wav_name.endswith(".wav"):
                    continue
                print(wav_name)
                wav_path = os.path.join(chapter_path, wav_name)
                out_folder = os.path.join(out_dir, spk_id, chapter_id)

                if not os.path.isdir(out_folder):
                    os.makedirs(out_folder, exist_ok=True)    
                
                out_path = os.path.join(out_dir, spk_id, chapter_id, wav_name.replace(".wav", ".npy"))

                try:
                    code = extract_encodec_token(wav_path)
                except:
                    continue
                np.save(out_path, code)