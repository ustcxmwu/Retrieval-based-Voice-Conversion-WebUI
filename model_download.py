#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Retrieval-based-Voice-Conversion-WebUI
@File    :   model_download.py
@Time    :   2023-09-25 15:13
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""

from huggingface_hub import hf_hub_download


def download_pretrained_model(subfolder="pretrained"):
    for model in ["D32k.pth", "D40k.pth", "D48k.pth", "G32k.pth", "G40k.pth", "G48k.pth", "f0D32k.pth", "f0D40k.pth",
                  "f0D48k.pth",
                  "f0G32k.pth", "f0G40k.pth", "f0G48k.pth"]:
        hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", subfolder=subfolder, filename=model,
                        local_dir="./assets/", local_dir_use_symlinks=False)


def download_urv5():
    files = [
        "HP2-人声vocals+非人声instrumentals.pth",
        "HP2_all_vocals.pth",
        "HP3_all_vocals.pth",
        "HP5-主旋律人声vocals+其他instrumentals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoDeReverb.pth",
        "VR-DeEchoNormal.pth"
    ]
    for file in files:
        hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", subfolder="uvr5_weights", filename=file,
                        local_dir="./assets/", local_dir_use_symlinks=False)


def download_rmvpe():
    hf_hub_download(repo_id="lj1995/VoiceConversionWebUI", filename="rmvpe.pt",
                    local_dir="./assets/rmvpe/", local_dir_use_symlinks=False)


if __name__ == '__main__':
    # download_pretrained_model()
    download_rmvpe()
