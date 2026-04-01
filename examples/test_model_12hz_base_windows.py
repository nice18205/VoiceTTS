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
import argparse
from urllib.parse import urlparse
from urllib.request import urlretrieve

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def choose_runtime():
    if torch.cuda.is_available():
        return "cuda:0", torch.float16, "sdpa"
    return "cpu", torch.float32, "eager"


def run_case(tts: Qwen3TTSModel, out_dir: str, case_name: str, call_fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = call_fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    print(f"[{case_name}] time: {t1 - t0:.3f}s, n_wavs={len(wavs)}, sr={sr}")

    for i, w in enumerate(wavs):
        sf.write(os.path.join(out_dir, f"{case_name}_{i}.wav"), w, sr)


def cache_reference_audio(ref_audio: str, cache_dir: str):
    if ref_audio.startswith("http://") or ref_audio.startswith("https://"):
        ensure_dir(cache_dir)
        filename = os.path.basename(urlparse(ref_audio).path) or "ref_audio.wav"
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            print(f"downloading reference audio: {ref_audio}")
            urlretrieve(ref_audio, local_path)
        return local_path
    return ref_audio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--out-dir", default="qwen3_tts_test_voice_clone_output_wav")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--language", default="English")
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Run both xvec modes and promptThenGen case (slower).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device, dtype, attn_impl = choose_runtime()
    model_path = args.model_path
    out_dir = args.out_dir
    ensure_dir(out_dir)
    print(f"runtime: device={device}, dtype={dtype}, attn_implementation={attn_impl}")
    if device == "cpu":
        print("warning: running on CPU can be very slow for this 1.7B model.")
    print("loading model...")

    tts = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    print("model loaded.")

    # Reference audio(s)
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    ref_audio_single = cache_reference_audio(ref_audio_path_1, os.path.join(out_dir, "cache"))
    
    ref_text_single = (
        "Okay. Yeah. I resent you. I love you. I respect you. "
    )
    
    # Synthesis targets
    syn_text_single = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
    syn_lang_single = args.language

    common_gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=False,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
    )

    mode_list = [False, True] if args.full_run else [False]
    print(f"start generation, modes={mode_list}, max_new_tokens={args.max_new_tokens}")
    for xvec_only in mode_list:
        mode_tag = "xvec_only" if xvec_only else "icl"

        # Case 1: prompt single + synth single, direct
        run_case(
            tts,
            out_dir,
            f"case1_promptSingle_synSingle_direct_{mode_tag}",
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

        if args.full_run:
            run_case(
                tts,
                out_dir,
                f"case1_promptSingle_synSingle_promptThenGen_{mode_tag}",
                _case1b,
            )



if __name__ == "__main__":
    main()
