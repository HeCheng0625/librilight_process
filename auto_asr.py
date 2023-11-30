import argparse
import torchaudio
import numpy as np
import os
import librosa
from IPython.display import Audio
from tqdm import tqdm
import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build the statistics on LibriBig")
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--device_id', type=int)
    parser.add_argument('--export_chunk_len', type=int, default=750)
    parser.add_argument('--min_silence_len', type=int, default=500)
    parser.add_argument('--keep_silence', type=int, default=500)
    parser.add_argument('--spk_num_start', type=int, default=0)
    parser.add_argument('--spk_num_end', type=int, default=100)
    
    args = parser.parse_args()

    model = whisper.load_model("base")
    device = "cuda:{}".format(str(args.device_id))
    model.to(device)
    print(model.device)

    def transcribe_audio_whisper(path):
        result = model.transcribe(path)
        text = result['text']
        return text

    def get_large_audio_transcription_on_silence_whisper(wav_path, export_chunk_len, out_path):
        
        wav_name = wav_path.split("/")[-1].split(".")[0]
        temp_chunk_filename = os.path.join(out_path, wav_name + "_01.wav")
        if os.path.isfile(temp_chunk_filename):
            return
       
        sound = AudioSegment.from_file(wav_path)
        chunks = split_on_silence(sound, min_silence_len=500, silence_thresh=sound.dBFS-14, keep_silence=500)
        if not os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)

        try:
            output_chunks = [chunks[0]]
        except:
            return

        for chunk in chunks[1:]:
            if len(output_chunks[-1]) < export_chunk_len:
                output_chunks[-1] += chunk
            else:
                output_chunks.append(chunk)

        wav_name = wav_path.split("/")[-1].split(".")[0]
        for i, audio_chunk in enumerate(output_chunks, start=1):
            chunk_filename = os.path.join(out_path, wav_name + "_{}.wav".format(str(i).zfill(2)))
            text_filename = os.path.join(out_path, wav_name + "_{}.txt".format(str(i).zfill(2)))
            if not os.path.isfile(text_filename):
                audio_chunk.export(chunk_filename, format="wav")
            if os.path.isfile(text_filename):
                continue
            try:
                text = transcribe_audio_whisper(chunk_filename)
            except Exception as e:
                print("Error:", str(e))
            else:
                text = f"{text.lstrip().rstrip().capitalize()}"
                # print(chunk_filename, ":", text)
                with open(text_filename, "w") as f:
                    f.write(text)

    input_dir = args.in_dir
    out_dir = args.out_dir

    for spk_id in tqdm(sorted(os.listdir(input_dir))[args.spk_num_start: args.spk_num_end]):
        print("speaker:", spk_id)
        spk_path = os.path.join(input_dir, spk_id)
        if len(os.listdir(spk_path)) == 0:
            continue
        for chapter_id in tqdm(sorted(os.listdir(spk_path))):
            print("chapter:", chapter_id)
            chapter_path = os.path.join(spk_path, chapter_id)
            if len(os.listdir(chapter_path)) == 0:
                continue
            for wav_name in tqdm(sorted(os.listdir(chapter_path))):
                wav_path = os.path.join(chapter_path, wav_name)
                out_path = os.path.join(out_dir, spk_id, chapter_id)
                get_large_audio_transcription_on_silence_whisper(wav_path, args.export_chunk_len, out_path)
            break
