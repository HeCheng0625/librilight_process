import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union

from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

import torchaudio
import torch
import os
import numpy as np
import argparse
from tqdm import tqdm

try:
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials
except Exception:
    pass

class PypinyinBackend:
    """PypinyinBackend for Chinese. Most codes is referenced from espnet.
    There are two types pinyin or initials_finals, one is
    just like "ni1 hao3", the other is like "n i1 h ao3".
    """

    def __init__(
        self,
        backend="initials_finals",
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
    ) -> None:
        self.backend = backend
        self.punctuation_marks = punctuation_marks

    def phonemize(
        self, text: List[str], separator: Separator, strip=True, njobs=1
    ) -> List[str]:
        assert isinstance(text, List)
        phonemized = []
        for _text in text:
            _text = re.sub(" +", " ", _text.strip())
            _text = _text.replace(" ", separator.word)
            phones = []
            if self.backend == "pypinyin":
                for n, py in enumerate(
                    pinyin(
                        _text, style=Style.TONE3, neutral_tone_with_five=True
                    )
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)

                        phones.extend(list(py[0]))
                    else:
                        phones.extend([py[0], separator.syllable])
            elif self.backend == "pypinyin_initials_finals":
                for n, py in enumerate(
                    pinyin(
                        _text, style=Style.TONE3, neutral_tone_with_five=True
                    )
                ):
                    if all([c in self.punctuation_marks for c in py[0]]):
                        if len(phones):
                            assert phones[-1] == separator.syllable
                            phones.pop(-1)
                        phones.extend(list(py[0]))
                    else:
                        if py[0][-1].isalnum():
                            initial = get_initials(py[0], strict=False)
                            if py[0][-1].isdigit():
                                final = (
                                    get_finals(py[0][:-1], strict=False)
                                    + py[0][-1]
                                )
                            else:
                                final = get_finals(py[0], strict=False)
                            phones.extend(
                                [
                                    initial,
                                    separator.phone,
                                    final,
                                    separator.syllable,
                                ]
                            )
                        else:
                            assert ValueError
            else:
                raise NotImplementedError
            phonemized.append(
                "".join(phones).rstrip(f"{separator.word}{separator.syllable}")
            )
        return phonemized


class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        if backend == "espeak":
            phonemizer = EspeakBackend(
                language,
                punctuation_marks=punctuation_marks,
                preserve_punctuation=preserve_punctuation,
                with_stress=with_stress,
                tie=tie,
                language_switch=language_switch,
                words_mismatch=words_mismatch,
            )
        elif backend in ["pypinyin", "pypinyin_initials_finals"]:
            phonemizer = PypinyinBackend(
                backend=backend,
                punctuation_marks=punctuation_marks + separator.word,
            )
        else:
            raise NotImplementedError(f"{backend}")

        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone]
                + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def __call__(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        return [self.to_list(p) for p in phonemized]

def tokenize_text(tokenizer: TextTokenizer, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes[0]  # k2symbols

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Build the statistics on LibriBig")
    parser.add_argument('--in_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--device_id', type=int)
    parser.add_argument('--spk_num_start', type=int, default=0)
    parser.add_argument('--spk_num_end', type=int, default=100)
    args = parser.parse_args()

    tokenizer = TextTokenizer()

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
            for txt_name in sorted(os.listdir(chapter_path)):
                if not txt_name.endswith(".txt"):
                    continue
                print(txt_name)
                txt_path = os.path.join(chapter_path, txt_name)
                out_folder = os.path.join(out_dir, spk_id, chapter_id)

                if not os.path.isdir(out_folder):
                    os.makedirs(out_folder, exist_ok=True)    
                
                out_path = os.path.join(out_dir, spk_id, chapter_id, txt_name.replace(".txt", ".phone"))

                try:
                    with open(txt_path, "r") as f:
                        lines = f.readlines()
                        line = lines[0].replace("\n", "")
                    # print(line)
                    phone = tokenize_text(tokenizer, line)
                    phone_seq = [phn for phn in phone]
                    with open(out_path, 'w') as fin:
                        fin.write(' '.join(phone_seq))
                    # print(phone)
                except:
                    continue
