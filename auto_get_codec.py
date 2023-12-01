from encodec import EncodecModel
from encodec.utils import convert_audio

import torchaudio
import torch


if __name__ == "__main__":


    device = ...
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

    