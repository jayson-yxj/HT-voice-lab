# ASR 长音频显存与轻量模型复核

日期：2026-09-17。设备：RTX 4060（8188 MiB）。

## 结论

Qwen3-ASR-0.6B 的显存会随**当前连续声学片段**变长，而不是随整场会话总时长永久增长。官方流式状态保存累计音频，更新时会再次处理该状态内的音频；片段在换人、长静音或窗口滚动时重建后，输入长度才归零。CUDA 分配器还会保留已经申请的显存，所以系统看到的占用不会立刻下降。

当前保留 Qwen 作为中英质量候选，并把单个声学窗口限制为：12 秒后优先在说话人停顿处滚动，连续讲话最迟 15 秒滚动。窗口滚动只关闭 ASR 状态，人物编号不变，`turn_complete` 仍为 `null`。

同一段 102 秒中英素材中，46.48 秒英文发言原先保持一个 Qwen 状态；加入限制后分成最多约 15 秒的声学窗口：

| 配置 | 整卡峰值 | 最低空闲显存 | 首次出字 | RTF |
| --- | ---: | ---: | ---: | ---: |
| 原 Qwen 累积片段 | 7619 MiB | 197 MiB | 2.121s | 0.143 |
| Qwen 12/15s 有界窗口 | **4525 MiB** | **3291 MiB** | 2.146s | 0.148 |
| Qwen 有界窗口 + 1.5s 预卷去重 | **4464 MiB** | **3351 MiB** | 2.149s | 0.158 |

2026-09-18 已加入 1.5 秒预卷和窗口文字近似去重。原始 `utterances.json` 保留各声学窗口；`speech_spans.json` 和 `transcript.txt` 合并人工窗口，但仍保留 `turn_complete=null`。长英文中的 `when it / comes to`、`automated / production system` 已连续输出，未重复或漏词。相比无窗口配置，整卡峰值降低 3155 MiB；额外预卷使 RTF 从 0.148 增至 0.158，仍可实时运行。

## SenseVoiceSmall 实测

已固定并下载官方 `SenseVoiceSmall` 234M 权重，用相同人物区间做 15 秒定长窗口筛选：

| 素材 | 单模型 PyTorch 峰值分配 | 整卡峰值 | 计算 RTF | 观察 |
| --- | ---: | ---: | ---: | --- |
| 73 秒中文 | 964 MiB | 1718 MiB | 0.0043 | 有标点，速度快；出现“人形→人心/人性”等错词和情绪标签误触发 |
| 102 秒中英 | 964 MiB | 1719 MiB | 0.0036 | 英文基本连贯；`echo/reskilling/production` 等词不如 Qwen 准确 |

SenseVoice 数字是**单独模型筛选**，没有同时加载 Sortformer，不能与 Qwen 的双模型整卡峰值直接等同。它明显更轻、更快，也原生输出标点；当前两段素材没有证明它在中英文准确度上强于 Qwen。官方接口主要按不超过 30 秒的片段推理，首版如使用它，仍应放在有界滚动窗口中。

## 候选取舍

- **当前推荐：带预卷去重的有界 Qwen。** 现有素材的中文专名和长英文整体最好，15 秒限制让 8GB 显卡保留约 3.3GB 余量。
- **轻量备选：SenseVoiceSmall。** 适合显存或吞吐优先的部署；先补实时滚动快照和边界测试，再决定是否替换。
- **英文备选：faster-whisper small。** Whisper small 为 244M，多语种且有标点；原生 Whisper 使用 30 秒滑窗，实时使用还要增加稳定文本策略。
- **CPU/端侧备选：sherpa-onnx Zipformer。** 真流式、状态有界，但现有中英模型较旧，标点需要单独模型。
- Fun-ASR-Nano 约 800M、GLM-ASR-Nano 约 1.5B，不符合“更轻”；Moonshine v2 的当前流式模型以英文为主，不适合作为本项目唯一的中英模型。

本轮调研复核了 10 个方向的 102 条检索结果，并以官方仓库、模型卡和论文为准。没有找到一个经证实同时满足“比 Qwen3-ASR-0.6B 更轻、中文更强、英文更强、原生真流式”的单模型。

## 证据

- 有界 Qwen 结果：[无预卷转写](results/cctv-industry-qwen-bounded-15s-v1/transcript.txt)、[预卷去重转写](results/cctv-industry-qwen-overlap-15s-v2/transcript.txt)、[预卷去重指标](results/cctv-industry-qwen-overlap-15s-v2/summary.json)
- SenseVoice 结果：[中文](results/cctv-life-sensevoice-15s-v1/transcript.txt)、[中英](results/cctv-industry-sensevoice-15s-v1/transcript.txt)
- 机器可读对比：[asr-memory-comparison-20260917.json](results/asr-memory-comparison-20260917.json)
- 官方资料：[Qwen3-ASR 流式源码](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)、[SenseVoiceSmall 模型卡](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)、[Whisper](https://github.com/openai/whisper)、[faster-whisper](https://github.com/SYSTRAN/faster-whisper)、[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、[Moonshine v2 论文](https://arxiv.org/abs/2602.12241)

当前没有人工逐字参考稿，以上质量结论是同素材输出复核，不是正式 CER/WER。
