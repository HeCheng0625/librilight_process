import torch
import os
import numpy as np
import librosa
from tqdm import tqdm
import json

data_path = "/home/v-detaixin/librilight"
subset = "small"
phone_dataset = subset + "_phones"
codec_dataset = subset + "_acoustic_encodec"

out_path = os.path.join(data_path, subset+"_filter_train.json")
out_json = []

print(os.path.join(data_path, codec_dataset))
for spk_id in tqdm(sorted(os.listdir(os.path.join(data_path, codec_dataset)))):
    spk_path = os.path.join(data_path, codec_dataset, spk_id)
    for chapter_id in tqdm(sorted(os.listdir(spk_path))):
        chapter_path = os.path.join(spk_path, chapter_id)
        for code_name in sorted(os.listdir(chapter_path)):
            uid = code_name.replace(".npy", "")
            code_path = os.path.join(chapter_path, code_name)
            phone_path = os.path.join(chapter_path.replace("_acoustic_encodec",
                                                           "_phones"), uid + ".phone")
            try:
                code = np.load(code_path)
            except:
                print(code_path)
                print("load code failed")
                continue

            try:
                with open(phone_path, 'r') as fin:
                    phones = fin.readlines()
                    assert len(phones) == 1
                    phones = phones[0].strip()
                phones = phones.split(' ')
            except:
                print(phone_path)
                print("load phone failed")
                continue

            try:
                assert code.shape[1] == 8
            except:
                print("code shape 1 is not equal to 8")
                continue

            if code.shape[0] < 45:
                print("code shape is too short")
                continue

            if code.shape[0] > 30 * 75:
                print(code_path)
                print(code.shape[0])
                print("code shape is too long")
                continue

            utt_info = {}

            utt_info["Dataset"] = subset
            utt_info["Speaker"] = spk_id
            utt_info["Chapter"] = chapter_id
            utt_info["Uid"] = uid
            utt_info["Tokens"] = code.shape[0]

            out_json.append(utt_info)

with open(out_path, "w") as f:
    json.dump(out_json, f)



    