import argparse
import json
import os
from functools import wraps

import tensorrt as trt
import torch
from cuda import cudart
import pycuda.autoinit
import time as timm
import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper
from tensorrt_llm.runtime.session import Session, TensorInfo

import re
import sys
import jieba
import numpy as np
import onnxruntime
import torchaudio
from pypinyin import lazy_pinyin, Style
import math
from x_transformers.x_transformers import RotaryEmbedding

onnx_model_A         = "./F5_Preprocess.onnx"                        # The exported onnx model path.
onnx_model_C         = "./F5_Decode.onnx"                            # The exported onnx model path.

ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
HOP_LENGTH = 256                        # Number of samples between successive frames in the STFT
SAMPLE_RATE = 24000                     # The generated audio sample rate
RANDOM_SEED = 9527                      # Set seed to reproduce the generated audio
NFE_STEP = 32                           # F5-TTS model setting
SPEED = 1.0                             # Set for talking speed. Only works with dynamic_axes=True


with open("./vocab.txt", "r", encoding="utf-8") as f:
    vocab_char_map = {}
    for i, char in enumerate(f):
        vocab_char_map[char[:-1]] = i
vocab_size = len(vocab_char_map)

def is_chinese_char(c):
    cp = ord(c)
    return (
        0x4E00 <= cp <= 0x9FFF or    # CJK Unified Ideographs
        0x3400 <= cp <= 0x4DBF or    # CJK Unified Ideographs Extension A
        0x20000 <= cp <= 0x2A6DF or  # CJK Unified Ideographs Extension B
        0x2A700 <= cp <= 0x2B73F or  # CJK Unified Ideographs Extension C
        0x2B740 <= cp <= 0x2B81F or  # CJK Unified Ideographs Extension D
        0x2B820 <= cp <= 0x2CEAF or  # CJK Unified Ideographs Extension E
        0xF900 <= cp <= 0xFAFF or    # CJK Compatibility Ideographs
        0x2F800 <= cp <= 0x2FA1F     # CJK Compatibility Ideographs Supplement
    )

def convert_char_to_pinyin(text_list, polyphone=True):
    final_text_list = []
    merged_trans = str.maketrans({
        '“': '"', '”': '"', '‘': "'", '’': "'",
        ';': ','
    })
    chinese_punctuations = set("。，、；：？！《》【】—…")
    for text in text_list:
        char_list = []
        text = text.translate(merged_trans)
        for seg in jieba.cut(text):
            if seg.isascii():
                if char_list and len(seg) > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and all(is_chinese_char(c) for c in seg):
                pinyin_list = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for c in pinyin_list:
                    if c not in chinese_punctuations:
                        char_list.append(" ")
                    char_list.append(c)
            else:
                for c in seg:
                    if c.isascii():
                        char_list.append(c)
                    elif c in chinese_punctuations:
                        char_list.append(c)
                    else:
                        char_list.append(" ")
                        pinyin = lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)
                        char_list.extend(pinyin)
        final_text_list.append(char_list)
    return final_text_list

def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1
):
    get_idx = vocab_char_map.get
    list_idx_tensors = [torch.tensor([get_idx(c, 0) for c in t], dtype=torch.int32) for t in text]
    text = torch.nn.utils.rnn.pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text

def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None

# ONNX Runtime settings
onnxruntime.set_seed(RANDOM_SEED)
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 3  # error level, it a adjustable value.
session_opts.inter_op_num_threads = 0  # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0  # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True  # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
model_type = ort_session_A._inputs_meta[0].type
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
in_name_A2 = in_name_A[2].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name
out_name_A3 = out_name_A[3].name
out_name_A4 = out_name_A[4].name
out_name_A5 = out_name_A[5].name
out_name_A6 = out_name_A[6].name

ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
in_name_C1 = in_name_C[1].name
out_name_C0 = out_name_C[0].name


def get_input(reference_audio, ref_text, gen_text):                                                        # The target TTS.
    audio, sr = torchaudio.load(reference_audio)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        audio = resampler(audio)
    audio = audio.unsqueeze(0).numpy()
    if "float16" in model_type:
        audio = audio.astype(np.float16)
    zh_pause_punc = r"。，、；：？！"
    ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
    gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
    ref_audio_len = audio.shape[-1] // HOP_LENGTH + 1
    max_duration = np.array(ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / SPEED), dtype=np.int64)
    gen_text = convert_char_to_pinyin([ref_text + gen_text])
    text_ids = list_str_to_idx(gen_text, vocab_char_map).numpy()

    return audio, text_ids, max_duration

def preprocess(audio, text_ids, max_duration):
    print("\n\nRun F5-TTS preprocess.")
    noise, rope_cos, rope_sin, cat_mel_text, cat_mel_text_drop, qk_rotated_empty, ref_signal_len = ort_session_A.run(
            [out_name_A0, out_name_A1, out_name_A2, out_name_A3, out_name_A4, out_name_A5, out_name_A6],
            {
                in_name_A0: audio,
                in_name_A1: text_ids,
                in_name_A2: max_duration
            })
    
    t = torch.linspace(0, 1, 32 + 1, dtype=torch.float32)
    time_step = t + (-1.0) * (torch.cos(torch.pi * 0.5 * t) - 1 + t)
    delta_t = torch.diff(time_step)
    time_expand = torch.zeros((1, 32, 256), dtype=torch.float32)
    half_dim = 256 // 2
    emb_factor = math.log(10000) / (half_dim - 1)
    emb_factor = 1000.0 * torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_factor)
    for i in range(32):
        emb = time_step[i] * emb_factor
        time_expand[:, i, :] = torch.cat((emb.sin(), emb.cos()), dim=-1)

    # t（33,）,time_expand(1,32,256),delta_t(32,)都是固定值
    return noise, cat_mel_text, cat_mel_text_drop, time_expand, rope_cos, rope_sin, delta_t, ref_signal_len

class F5TTS(object):

    def __init__(self,
                 config,
                 debug_mode=True,
                 stream: torch.cuda.Stream = None):
        self.dtype = config['pretrained_config']['dtype']

        rank = tensorrt_llm.mpi_rank()
        world_size = config['pretrained_config']['mapping']['world_size']
        cp_size = config['pretrained_config']['mapping']['cp_size']
        tp_size = config['pretrained_config']['mapping']['tp_size']
        pp_size = config['pretrained_config']['mapping']['pp_size']
        assert pp_size == 1
        self.mapping = tensorrt_llm.Mapping(world_size=world_size,
                                            rank=rank,
                                            cp_size=cp_size,
                                            tp_size=tp_size,
                                            pp_size=1,
                                            gpus_per_node=args.gpus_per_node)

        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f'cuda:{5}')
        torch.cuda.set_device(self.device)
        # CUASSERT(cudart.cudaSetDevice(local_rank))

        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        engine_file = os.path.join(args.tllm_model_dir, f"rank{rank}.engine")
        logger.info(f'Loading engine from {engine_file}')
        with open(engine_file, "rb") as f:
            engine_buffer = f.read()

        assert engine_buffer is not None

        self.session = Session.from_serialized_engine(engine_buffer)

        self.debug_mode = debug_mode

        self.inputs = {}
        self.outputs = {}
        self.buffer_allocated = False

        expected_tensor_names = ['noise', 'cond', 'cond_drop', 'time', 'rope_cos', 'rope_sin', 't_scale', 'denoised']

        if self.mapping.tp_size > 1:
            self.buffer, self.all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
                self.mapping,
                CustomAllReduceHelper.max_workspace_size_auto(
                    self.mapping.tp_size))
            self.inputs['all_reduce_workspace'] = self.all_reduce_workspace
            expected_tensor_names += ['all_reduce_workspace']

        found_tensor_names = [
            self.session.engine.get_tensor_name(i)
            for i in range(self.session.engine.num_io_tensors)
        ]
        if not self.debug_mode and set(expected_tensor_names) != set(
                found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            logger.error(f"Expected tensor names: {expected_tensor_names}")
            logger.error(f"Found tensor names: {found_tensor_names}")
            raise RuntimeError(
                "Tensor names in engine are not the same as expected.")
        if self.debug_mode:
            self.debug_tensors = list(
                set(found_tensor_names) - set(expected_tensor_names))

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.session.engine.get_tensor_dtype(name))
        return dtype

    def _setup(self, batch_size):
        for i in range(self.session.engine.num_io_tensors):
            name = self.session.engine.get_tensor_name(i)
            if self.session.engine.get_tensor_mode(
                    name) == trt.TensorIOMode.OUTPUT:
                shape = list(self.session.engine.get_tensor_shape(name))
                shape[1] = batch_size
                self.outputs[name] = torch.empty(shape,
                                                 dtype=self._tensor_dtype(name),
                                                 device=self.device)

        self.buffer_allocated = True

    def cuda_stream_guard(func):
        """Sync external stream and set current stream to the one bound to the session. Reset on exit.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            external_stream = torch.cuda.current_stream()
            if external_stream != self.stream:
                external_stream.synchronize()
                torch.cuda.set_stream(self.stream)
            ret = func(self, *args, **kwargs)
            if external_stream != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external_stream)
            return ret

        return wrapper

    @cuda_stream_guard
    def forward(self, noise: torch.Tensor, cond: torch.Tensor,
                cond_drop: torch.Tensor, time_expand: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor, delta_t: torch.Tensor):

        self._setup(noise.shape[1])
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        time = time_expand[:, 0]
        t_scale = delta_t[0].unsqueeze(0)
        input_type = str_dtype_to_torch(self.dtype)
        inputs = {
            'noise': noise.to(input_type),
            'cond': cond.to(input_type),
            'cond_drop': cond_drop.to(input_type),
            'time': time.to(input_type),
            'rope_cos': rope_cos.to(input_type),
            'rope_sin': rope_sin.to(input_type),
            't_scale': t_scale.to(input_type)
        }
        self.inputs.update(**inputs)
        self.session.set_shapes(self.inputs)

        for tensor_name in self.inputs:
            tensor = self.inputs[tensor_name]
            ptr = tensor.data_ptr()
            self.session.context.set_tensor_address(tensor_name, ptr)

        for tensor_name in self.outputs:
            tensor = self.outputs[tensor_name]
            ptr = tensor.data_ptr() if isinstance(tensor,
                                                torch.Tensor) else tensor
            self.session.context.set_tensor_address(tensor_name, ptr)
    
        i = 0
        while i < NFE_STEP:
            if 0 != i:
                self.inputs['time'] = time_expand[:, i].to(input_type)
                self.inputs['t_scale'] = delta_t[i].unsqueeze(0).to(input_type)
                self.inputs['noise'] = self.outputs["denoised"]
                self.session.context.set_tensor_address('time', self.inputs['time'].data_ptr())
                self.session.context.set_tensor_address('t_scale', self.inputs['t_scale'].data_ptr())
                self.session.context.set_tensor_address('noise', self.inputs['noise'].data_ptr())
            self.session.context.execute_async_v3(self.stream.cuda_stream)
            i += 1
        return self.outputs["denoised"]

def decode(noise, ref_signal_len, audio_save_path = './gen.wav'):
    
    generated_signal = ort_session_C.run(
            [out_name_C0],
            {
                in_name_C0: noise,
                in_name_C1: ref_signal_len
            })[0]
    # Save to audio
    audio_tensor = torch.tensor(generated_signal, dtype=torch.float32).squeeze(0)
    torchaudio.save(audio_save_path, audio_tensor, SAMPLE_RATE)

def main(args):
    tensorrt_llm.logger.set_level(args.log_level)
    torch.manual_seed(args.seed)
    assert torch.cuda.is_available()
    device = "cuda"
    # Load model:
    config_file = os.path.join(args.tllm_model_dir, 'config.json')
    with open(config_file) as f:
        config = json.load(f)
    model = F5TTS(config, debug_mode=args.debug_mode)
    #------------------------get input-------------------------#
    reference_audio      = "./basic_ref_zh.wav"     # The reference audio path.
    ref_text             = "对，这就是我，万人敬仰的太乙真人。" # The ASR result of reference audio.
    gen_text             = "对，这就是我，万人敬仰的超级玛丽"  
    audio, text_ids, max_duration = get_input(reference_audio, ref_text, gen_text)
    #------------------------preprocess-------------------------#
    preprocess_time = timm.time()
    noise, cond, cond_drop, time_expand, rope_cos, rope_sin, delta_t, ref_signal_len= preprocess(audio, text_ids, max_duration)
    noise = torch.from_numpy(noise)
    cond = torch.from_numpy(cond)
    cond_drop = torch.from_numpy(cond_drop)
    rope_cos = torch.from_numpy(rope_cos)
    rope_sin = torch.from_numpy(rope_sin)
    # model forward
    denoised = model.forward(noise.cuda(), cond.cuda(), cond_drop.cuda(), time_expand.cuda(), rope_cos.cuda(), rope_sin.cuda(),  delta_t.cuda())
    # decode with vocos
    decode(denoised.cpu().numpy().astype(np.float32), ref_signal_len)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tllm_model_dir",
                        type=str,
                        default='./engine_outputs/')
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument("--debug_mode", type=bool, default=True)
    args = parser.parse_args()
    main(args)
