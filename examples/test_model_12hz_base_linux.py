# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import ctypes
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def load_ref_text_from_lab(ref_audio_path: str) -> str:
    lab_path = os.path.splitext(ref_audio_path)[0] + ".lab"
    if not os.path.isfile(lab_path):
        raise FileNotFoundError(f"Reference text file not found: {lab_path}")
    with open(lab_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def maybe_patch_flash_attn_abi():
    """
    Workaround for flash-attn / torch ABI mismatch in some environments.
    This only affects the current Python process.
    """
    default_libc10 = "/home/zhoukun/anaconda3/envs/sglang/lib/python3.10/site-packages/torch/lib/libc10.so"
    libc10_path = os.environ.get("FLASH_ATTN_LIBC10_PATH", default_libc10)
    if os.path.isfile(libc10_path):
        ctypes.CDLL(libc10_path, mode=ctypes.RTLD_GLOBAL)
        print(f"Loaded extra libc10 for flash-attn ABI workaround: {libc10_path}")
    else:
        print(f"Skip flash-attn ABI workaround, libc10 not found: {libc10_path}")


def run_case(tts: Qwen3TTSModel, out_dir: str, case_name: str, call_fn):
    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = call_fn()

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[{case_name}] time: {t1 - t0:.3f}s, n_wavs={len(wavs)}, sr={sr}")

    for i, w in enumerate(wavs):
        sf.write(os.path.join(out_dir, f"{case_name}_{i}.wav"), w, sr)


def main():
    device = "cuda:0"
    MODEL_PATH = "/home/zhoukun/gjc/VoiceTTS/model/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3"
    OUT_DIR = "qwen3_tts_test_voice_clone_output_wav"
    ensure_dir(OUT_DIR)
    maybe_patch_flash_attn_abi()

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Reference audio/text from local dataset (Chinese)
    ref_audio_single = "/home/zhoukun/gjc/VoiceTTS/datasets/KTXY/zh_vo_Main_Linaxita_2_4_3_32.wav"
    ref_text_single = load_ref_text_from_lab(ref_audio_single)
    print(f"Loaded reference text from .lab: {ref_text_single}")

    # Single synthesis target
    syn_text_single = "张司瑜，愚人节快乐！"
    syn_lang_single = "Chinese"

    common_gen_kwargs = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
    )

    for xvec_only in [False, True]:
        mode_tag = "xvec_only" if xvec_only else "icl"

        # Case 1: prompt single + synth single, direct
        run_case(
            tts, OUT_DIR, f"case1_promptSingle_synSingle_direct_{mode_tag}",
            lambda: tts.generate_voice_clone(
                text=syn_text_single,
                language=syn_lang_single,
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
                **common_gen_kwargs,
            ),
        )

        # Case 1b: prompt single + synth single, via create_voice_clone_prompt
        def _case1b():
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
            )
            return tts.generate_voice_clone(
                text=syn_text_single,
                language=syn_lang_single,
                voice_clone_prompt=prompt_items,
                **common_gen_kwargs,
            )

        run_case(
            tts, OUT_DIR, f"case1_promptSingle_synSingle_promptThenGen_{mode_tag}",
            _case1b,
        )


if __name__ == "__main__":
    main()
