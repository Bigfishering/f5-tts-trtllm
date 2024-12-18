# f5-tts-trtllm

目录中model和example分别对应Tensorrt-LLM源码中的tensorrt_llm/models和example。
在Tensorrt-LLM源码中的tensorrt_llm/models和example目录下分别新建f5tts/ 目录，然后将repo中的代码放入对应的目录。

example/f5tts目录如下。需要通过导出前处理以及后处理的onnx、basic_ref_zh.wav和vocab.txt可从源码repo中获取。
导出ONNX参考 ：https://github.com/DakeQQ/F5-TTS-ONNX （由于前处理和后处理耗时较小，并没有进行优化，直接用了onnx的方案。）
F5-TTS源码：https://github.com/SWivid/F5-TTS

.
├── F5_Decode.onnx
├── F5_Preprocess.onnx
├── __pycache__
│   └── diffusion.cpython-310.pyc
├── basic_ref_zh.wav
├── ckpts
│   └── model_1200000.pt
├── convert_checkpoint.py
├── sample_tts.py
└── vocab.txt


tensorrt_llm/models/f5tts目录如下。

.
├── model.py
├── module.py


在tensorrt_llm/models/__init__.py导入f5tts模型并在MODEL_MAP中注册模型。

cd example/f5tts
1. convert_checkpoint
   python convert_checkpoint.py
2. build engine(支持Tensor 并行, --tp_size)
   trtllm-build --checkpoint_dir ./tllm_checkpoint/ --remove_input_padding disable --bert_attention_plugin disable
3. python sample.py
