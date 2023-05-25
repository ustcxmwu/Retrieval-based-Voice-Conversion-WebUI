import torch, os, traceback, sys, warnings, shutil, numpy as np

os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import threading
from time import sleep
from subprocess import Popen, PIPE
import faiss
from random import shuffle

now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
import ffmpeg
from pathlib import Path

# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if (not torch.cuda.is_available()) or ngpu == 0:
    if_gpu_ok = False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if (
                "10" in gpu_name
                or "16" in gpu_name
                or "20" in gpu_name
                or "30" in gpu_name
                or "40" in gpu_name
                or "A2" in gpu_name.upper()
                or "A3" in gpu_name.upper()
                or "A4" in gpu_name.upper()
                or "P4" in gpu_name.upper()
                or "A50" in gpu_name.upper()
                or "70" in gpu_name
                or "80" in gpu_name
                or "90" in gpu_name
                or "M4" in gpu_name.upper()
                or "T4" in gpu_name.upper()
                or "TITAN" in gpu_name.upper()
        ):  # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok == True and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = "很遗憾您这没有能用的显卡来支持您训练"
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])
from infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from scipy.io import wavfile
from fairseq import checkpoint_utils
import gradio as gr
import logging
from vc_infer_pipeline import VC
from config import Config
from infer_uvr5 import _audio_pre_
from my_utils import load_audio
from train.process_ckpt import show_info, change_info, merge, extract_small_model

config = Config()
# from trainset_preprocess_pipeline import PreProcess
logging.getLogger("numba").setLevel(logging.WARNING)


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


hubert_model = None


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


weight_root = "weights"
weight_uvr5_root = "uvr5_weights"
index_root = "logs"
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth"):
        uvr5_names.append(name.replace(".pth", ""))


def vc_single(
        sid,
        input_audio_path,
        f0_up_key,
        f0_file,
        f0_method,
        file_index,
        file_index2,
        # file_big_npy,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
            if file_index != ""
            else file_index2
        )  # 防止小白写错，自动帮他替换掉
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            f0_file=f0_file,
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def vc_multi(
        sid,
        dir_path,
        opt_root,
        paths,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        # file_big_npy,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
):
    try:
        dir_path = (
            dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        os.makedirs(opt_root, exist_ok=True)
        try:
            if dir_path != "":
                paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            else:
                paths = [path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos = []
        for path in paths:
            info, opt = vc_single(
                sid,
                path,
                f0_up_key,
                None,
                f0_method,
                file_index,
                file_index2,
                # file_big_npy,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
            )
            if "Success" in info:
                try:
                    tgt_sr, audio_opt = opt
                    wavfile.write(
                        "%s/%s" % (opt_root, os.path.basename(path)), tgt_sr, audio_opt
                    )
                except:
                    info += traceback.format_exc()
            infos.append("%s->%s" % (os.path.basename(path), info))
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg):
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        pre_fun = _audio_pre_(
            agg=int(agg),
            model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
            device=config.device,
            is_half=config.is_half,
        )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                        info["streams"][0]["channels"] == 2
                        and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal)
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (tmp, os.path.basename(inp_path))
                os.system(
                    "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal)
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            del pre_fun.model
            del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)


# 一个选项卡全局只能有一个音色
def get_vc(sid):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model != None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###楼下不这么折腾清理不干净
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"visible": True, "maximum": n_spk, "__type__": "update"}


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def clean():
    return {"value": "", "__type__": "update"}


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() == None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() == None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    cmd = (
            config.python_cmd
            + " trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s "
            % (trainset_dir, sr, n_p, now_dir, exp_dir)
            + str(config.noparallel)
    )
    print(cmd)
    p = Popen(cmd, shell=True)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0] == True:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)
    yield log


# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        cmd = config.python_cmd + " extract_f0_print.py %s/logs/%s %s %s" % (
            now_dir,
            exp_dir,
            n_p,
            f0method,
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
        ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
        done = [False]
        threading.Thread(
            target=if_done,
            args=(
                done,
                p,
            ),
        ).start()
        while 1:
            with open(
                    "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0] == True:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        print(log)
        yield log
    ####对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
                config.python_cmd
                + " extract_feature_print.py %s %s %s %s %s/logs/%s %s"
                % (
                    config.device,
                    leng,
                    idx,
                    n_g,
                    now_dir,
                    exp_dir,
                    version19,
                )
        )
        print(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0] == True:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)
    yield log


def change_sr2(sr2, if_f0_3, version19):
    vis_v = True if sr2 == "40k" else False
    if sr2 != "40k":
        version19 = "v1"
    path_str = "" if version19 == "v1" else "_v2"
    version_state = {"visible": vis_v, "__type__": "update"}
    if vis_v == False:
        version_state["value"] = "v1"
    f0_str = "f0" if if_f0_3 else ""
    return (
        "pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2),
        "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2),
        version_state,
    )


def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return "pretrained%s/%sG%s.pth" % (
        path_str,
        f0_str,
        sr2,
    ), "pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    if if_f0_3:
        return (
            {"visible": True, "__type__": "update"},
            "pretrained%s/f0G%s.pth" % (path_str, sr2),
            "pretrained%s/f0D%s.pth" % (path_str, sr2),
        )
    return (
        {"visible": False, "__type__": "update"},
        "pretrained%s/G%s.pth" % (path_str, sr2),
        "pretrained%s/D%s.pth" % (path_str, sr2),
    )


# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
        exp_dir1,
        sr2,
        if_f0_3,
        spk_id5,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:", gpus16)
    if gpus16:
        cmd = (
                config.python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s -pg %s -pd %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    gpus16,
                    total_epoch11,
                    save_epoch10,
                    pretrained_G14,
                    pretrained_D15,
                    1 if if_save_latest13 == "是" else 0,
                    1 if if_cache_gpu17 == "是" else 0,
                    1 if if_save_every_weights18 == "是" else 0,
                    version19,
                )
        )
    else:
        cmd = (
                config.python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s -pg %s -pd %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    total_epoch11,
                    save_epoch10,
                    pretrained_G14,
                    pretrained_D15,
                    1 if if_save_latest13 == "是" else 0,
                    1 if if_cache_gpu17 == "是" else 0,
                    1 if if_save_every_weights18 == "是" else 0,
                    version19,
                )
        )
    print(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if os.path.exists(feature_dir) == False:
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos = []
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, version19),
    )
    # faiss.write_index(index, '%s/trained_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i: i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, version19),
    )
    infos.append(
        "成功构建索引，added_IVF%s_Flat_nprobe_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, version19)
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
        exp_dir1,
        sr2,
        if_f0_3,
        trainset_dir4,
        spk_id5,
        np7,
        f0method8,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        pretrained_G14,
        pretrained_D15,
        gpus16,
        if_cache_gpu17,
        if_save_every_weights18,
        version19,
):
    infos = []

    def get_info_str(strr):
        if isinstance(strr, list):
            infos.extend(strr)
        else:
            infos.append(strr)
        return "\n".join(infos)

    model_log_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    preprocess_log_path = "%s/preprocess.log" % model_log_dir
    extract_f0_feature_log_path = "%s/extract_f0_feature.log" % model_log_dir
    gt_wavs_dir = "%s/0_gt_wavs" % model_log_dir
    feature_dir = (
        "%s/3_feature256" % model_log_dir
        if version19 == "v1"
        else "%s/3_feature768" % model_log_dir
    )

    os.makedirs(model_log_dir, exist_ok=True)
    #########step1:处理数据
    open(preprocess_log_path, "w").close()
    cmd = (
            config.python_cmd
            + " trainset_preprocess_pipeline_print.py %s %s %s %s "
            % (trainset_dir4, sr_dict[sr2], np7, model_log_dir)
            + str(config.noparallel)
    )
    yield get_info_str("step1:正在处理数据")
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True)
    p.wait()
    with open(preprocess_log_path, "r") as f:
        print(f.read())
    #########step2a:提取音高
    open(extract_f0_feature_log_path, "w")
    if if_f0_3:
        yield get_info_str("step2a:正在提取音高")
        cmd = config.python_cmd + " extract_f0_print.py %s %s %s" % (
            model_log_dir,
            np7,
            f0method8,
        )
        yield get_info_str(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        p.wait()
        with open(extract_f0_feature_log_path, "r") as f:
            print(f.read())
    else:
        yield get_info_str("step2a:无需提取音高")
    #######step2b:提取特征
    yield get_info_str("step2b:正在提取特征")
    gpus = gpus16.split("-")
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = config.python_cmd + " extract_feature_print.py %s %s %s %s %s %s" % (
            config.device,
            leng,
            idx,
            n_g,
            model_log_dir,
            version19,
        )
        yield get_info_str(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    for p in ps:
        p.wait()
    with open(extract_f0_feature_log_path, "r") as f:
        print(f.read())
    #######step3a:训练模型
    yield get_info_str("step3a:正在训练模型")
    # 生成filelist
    if if_f0_3:
        f0_dir = "%s/2a_f0" % model_log_dir
        f0nsf_dir = "%s/2b-f0nsf" % model_log_dir
        names = (
                set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
                & set([name.split(".")[0] for name in os.listdir(feature_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0_dir)])
                & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % model_log_dir, "w") as f:
        f.write("\n".join(opt))
    yield get_info_str("write filelist done")
    if gpus16:
        cmd = (
                config.python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s -pg %s -pd %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    gpus16,
                    total_epoch11,
                    save_epoch10,
                    pretrained_G14,
                    pretrained_D15,
                    1 if if_save_latest13 == "是" else 0,
                    1 if if_cache_gpu17 == "是" else 0,
                    1 if if_save_every_weights18 == "是" else 0,
                    version19,
                )
        )
    else:
        cmd = (
                config.python_cmd
                + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s -pg %s -pd %s -l %s -c %s -sw %s -v %s"
                % (
                    exp_dir1,
                    sr2,
                    1 if if_f0_3 else 0,
                    batch_size12,
                    total_epoch11,
                    save_epoch10,
                    pretrained_G14,
                    pretrained_D15,
                    1 if if_save_latest13 == "是" else 0,
                    1 if if_cache_gpu17 == "是" else 0,
                    1 if if_save_every_weights18 == "是" else 0,
                    version19,
                )
        )
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    yield get_info_str("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log")
    #######step3b:训练索引
    npys = []
    listdir_res = list(os.listdir(feature_dir))
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % model_log_dir, big_npy)

    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    yield get_info_str("%s,%s" % (big_npy.shape, n_ivf))
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    yield get_info_str("training index")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, version19),
    )
    yield get_info_str("adding index")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i: i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, version19),
    )
    yield get_info_str(
        "成功构建索引, added_IVF%s_Flat_nprobe_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, version19)
    )
    yield get_info_str("全流程结束！")


#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if (
            os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log"))
            == False
    ):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
                ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


from infer_pack.models_onnx import SynthesizerTrnMsNSFsidM


def export_onnx(ModelPath, ExportedPath, MoeVS=True):
    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    hidden_channels = cpt["config"][-2]  # hidden_channels，为768Vec做准备

    test_phone = torch.rand(1, 200, hidden_channels)  # hidden unit
    test_phone_lengths = torch.tensor([200]).long()  # hidden unit 长度（貌似没啥用）
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # 基频（单位赫兹）
    test_pitchf = torch.rand(1, 200)  # nsf基频
    test_ds = torch.LongTensor([0])  # 说话人ID
    test_rnd = torch.rand(1, 192, 200)  # 噪声（加入随机因子）

    device = "cpu"  # 导出时设备（不影响使用模型）

    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], is_half=False
    )  # fp32导出（C++要支持fp16必须手动将内存重新排列所以暂时不用fp16）
    net_g.load_state_dict(cpt["weight"], strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = [
        "audio",
    ]
    # net_g.construct_spkmixmap(n_speaker) 多角色混合轨道导出
    torch.onnx.export(
        net_g,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
            test_rnd.to(device),
        ),
        ExportedPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=16,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    return "Finished"

def show_train_info(experiment):
    log_file = Path("./logs") / experiment / "train.log"
    print(str(log_file).center(80, "="))
    if not log_file.exists():
        return ""
    else:
        with open(log_file, mode='r') as f:
            data = f.readlines()[-5:]
        return "".join(data)


with gr.Blocks() as app:
    gr.Markdown(
        """
        # 小白 Music Box
        Pandada Game 语音生成试验盒. 
        """)
    with gr.Tabs():
        with gr.TabItem("模型推理"):
            with gr.Row():
                sid0 = gr.Dropdown(label="推理音色", choices=sorted(names))
                refresh_button = gr.Button("刷新音色列表和索引路径", variant="primary")
                clean_button = gr.Button("卸载音色省显存", variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label="请选择说话人id",
                    value=0,
                    visible=False,
                    interactive=True,
                )
                clean_button.click(fn=clean, inputs=[], outputs=[sid0])
                sid0.change(
                    fn=get_vc,
                    inputs=[sid0],
                    outputs=[spk_item],
                )
            with gr.Group():
                gr.Markdown(
                    value="男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. "
                )
                with gr.Row():
                    with gr.Column():
                        vc_transform0 = gr.Number(
                            label="变调(整数, 半音数量, 升八度12降八度-12)", value=0
                        )
                        input_audio0 = gr.Textbox(
                            label="输入待处理音频文件路径(默认是正确格式示例)",
                            value="./raw/filename.wav",
                        )
                        f0method0 = gr.Radio(
                            label="选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比",
                            choices=["pm", "harvest"],
                            value="pm",
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音",
                            value=3,
                            step=1,
                            interactive=True,
                        )
                    with gr.Column():
                        file_index1 = gr.Textbox(
                            label="特征检索库文件路径,为空则使用下拉的选择结果",
                            value="",
                            interactive=True,
                        )
                        file_index2 = gr.Dropdown(
                            label="自动检测index路径,下拉式选择(dropdown)",
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        refresh_button.click(
                            fn=change_choices, inputs=[], outputs=[sid0, file_index2]
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="检索特征占比",
                            value=0.76,
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label="后处理重采样至最终采样率，0为不进行重采样",
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label="输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络",
                            value=1,
                            interactive=True,
                        )
                    f0_file = gr.File(label="F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调")
                    but0 = gr.Button("转换", variant="primary")
                    with gr.Column():
                        vc_output1 = gr.Textbox(label="输出信息")
                        vc_output2 = gr.Audio(label="输出音频(右下角三个点,点了可以下载)")
                    but0.click(
                        vc_single,
                        [
                            spk_item,
                            input_audio0,
                            vc_transform0,
                            f0_file,
                            f0method0,
                            file_index1,
                            file_index2,
                            # file_big_npy1,
                            index_rate1,
                            filter_radius0,
                            resample_sr0,
                            rms_mix_rate0,
                        ],
                        [vc_output1, vc_output2],
                    )
        with gr.TabItem("伴奏人声分离"):
            with gr.Group():
                gr.Markdown(
                    value=
                    "人声伴奏分离批量处理, 使用UVR5模型. <br>不带和声用HP2, 带和声且提取的人声不需要和声用HP5<br>合格的文件夹路径格式举例: E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)"
                )
                with gr.Row():
                    with gr.Column():
                        dir_wav_input = gr.Textbox(
                            label="输入待处理音频文件夹路径",
                            value="E:\\codes\\py39\\test-20230416b\\todo-songs\\todo-songs",
                        )
                        wav_inputs = gr.File(
                            file_count="multiple", label="也可批量输入音频文件, 二选一, 优先读文件夹"
                        )
                    with gr.Column():
                        model_choose = gr.Dropdown(label="模型", choices=uvr5_names)
                        agg = gr.Slider(
                            minimum=0,
                            maximum=20,
                            step=1,
                            label="人声提取激进程度",
                            value=10,
                            interactive=True,
                            visible=False,  # 先不开放调整
                        )
                        opt_vocal_root = gr.Textbox(
                            label="指定输出人声文件夹", value="opt"
                        )
                        opt_ins_root = gr.Textbox(label="指定输出乐器文件夹", value="opt")
                    but2 = gr.Button("转换", variant="primary")
                    vc_output4 = gr.Textbox(label="输出信息")
                    but2.click(
                        uvr,
                        [
                            model_choose,
                            dir_wav_input,
                            opt_vocal_root,
                            wav_inputs,
                            opt_ins_root,
                            agg,
                        ],
                        [vc_output4],
                    )
        with gr.TabItem("训练"):
            gr.Markdown(
                value=
                "step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. "
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(label="输入实验名", value="mi-test")
                sr2 = gr.Radio(
                    label="目标采样率",
                    choices=["32k", "40k", "48k"],
                    value="40k",
                    interactive=True,
                )
                if_f0_3 = gr.Radio(
                    label="模型是否带音高指导(唱歌一定要, 语音可以不要)",
                    choices=[True, False],
                    value=True,
                    interactive=True,
                )
                version19 = gr.Radio(
                    label="版本(目前仅40k支持了v2)",
                    choices=["v1", "v2"],
                    value="v1",
                    interactive=True,
                    visible=True,
                )
                np7 = gr.Slider(
                    minimum=0,
                    maximum=config.n_cpu,
                    step=1,
                    label="提取音高和处理数据使用的CPU进程数",
                    value=config.n_cpu,
                    interactive=True,
                )
            with gr.Group():  # 暂时单人的, 后面支持最多4人的#数据处理
                gr.Markdown(
                    value="step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. "
                )
                with gr.Row():
                    trainset_dir4 = gr.Textbox(label="输入训练文件夹路径", value="./train/Xiaobai")
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label="请指定说话人id",
                        value=0,
                        interactive=True,
                    )
                    but1 = gr.Button("处理数据", variant="primary")
                    info1 = gr.Textbox(label="输出信息", value="", max_lines=6)
                    but1.click(
                        preprocess_dataset, [trainset_dir4, exp_dir1, sr2, np7], [info1]
                    )
            with gr.Group():
                gr.Markdown(value="step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)")
                with gr.Row():
                    with gr.Column():
                        gpus6 = gr.Textbox(
                            label="以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2",
                            value=gpus,
                            interactive=True,
                        )
                        gpu_info9 = gr.Textbox(label="显卡信息", value=gpu_info)
                    with gr.Column():
                        f0method8 = gr.Radio(
                            label="选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢",
                            choices=["pm", "harvest", "dio"],
                            value="harvest",
                            interactive=True,
                        )
                    but2 = gr.Button("特征提取", variant="primary")
                    info2 = gr.Textbox(label="输出信息", value="", max_lines=8)
                    but2.click(
                        extract_f0_feature,
                        [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19],
                        [info2],
                    )
            with gr.Group():
                gr.Markdown(value="step3: 填写训练设置, 开始训练模型和索引")
                with gr.Row():
                    save_epoch10 = gr.Slider(
                        minimum=0,
                        maximum=50,
                        step=1,
                        label="保存频率save_every_epoch",
                        value=5,
                        interactive=True,
                    )
                    total_epoch11 = gr.Slider(
                        minimum=0,
                        maximum=1000,
                        step=1,
                        label="总训练轮数total_epoch",
                        value=20,
                        interactive=True,
                    )
                    batch_size12 = gr.Slider(
                        minimum=1,
                        maximum=40,
                        step=1,
                        label="每张显卡的batch_size",
                        value=default_batch_size,
                        interactive=True,
                    )
                    if_save_latest13 = gr.Radio(
                        label="是否仅保存最新的ckpt文件以节省硬盘空间",
                        choices=["是", "否"],
                        value="否",
                        interactive=True,
                    )
                    if_cache_gpu17 = gr.Radio(
                        label="是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速",
                        choices=["是", "否"],
                        value="否",
                        interactive=True,
                    )
                    if_save_every_weights18 = gr.Radio(
                        label="是否在每次保存时间点将最终小模型保存至weights文件夹",
                        choices=["是", "否"],
                        value="否",
                        interactive=True,
                    )
                with gr.Row():
                    pretrained_G14 = gr.Textbox(
                        label="加载预训练底模G路径",
                        value="pretrained/f0G40k.pth",
                        interactive=True,
                    )
                    pretrained_D15 = gr.Textbox(
                        label="加载预训练底模D路径",
                        value="pretrained/f0D40k.pth",
                        interactive=True,
                    )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, version19],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    if_f0_3.change(
                        change_f0,
                        [if_f0_3, sr2, version19],
                        [f0method8, pretrained_G14, pretrained_D15],
                    )
                    gpus16 = gr.Textbox(
                        label="以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2",
                        value=gpus,
                        interactive=True,
                    )
                    but3 = gr.Button("训练模型", variant="primary")
                    but4 = gr.Button("训练特征索引", variant="primary")
                    but5 = gr.Button("一键训练", variant="primary")
                    info3 = gr.Textbox(label="输出信息", value="", max_lines=10)
                    but3.click(
                        click_train,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            spk_id5,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                        ],
                        info3,
                    )
                    but4.click(train_index, [exp_dir1, version19], info3)
                    but5.click(
                        train1key,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            trainset_dir4,
                            spk_id5,
                            np7,
                            f0method8,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                        ],
                        info3,
                    )
                with gr.Row():
                    # train_infos = gr.Textbox(label="输出信息", value=TrainLogParser(Path(".") / str(exp_dir1)), max_lines=10, every=10)
                    train_infos = gr.Textbox(label="训练日志", value="", max_lines=10)
                    train_infos_btn = gr.Button("查看训练日志", variant="primary")
                    train_infos_btn.click(show_train_info, exp_dir1, train_infos)

        with gr.TabItem("常见问题解答"):
            try:
                with open("docs/faq.md", "r", encoding="utf8") as f:
                    info = f.read()
                gr.Markdown(value=info)
            except:
                gr.Markdown(traceback.format_exc())

    app.queue(concurrency_count=511, max_size=1022).launch(
        server_name="0.0.0.0",
        inbrowser=not config.noautoopen,
        server_port=config.listen_port,
        quiet=True,
    )
