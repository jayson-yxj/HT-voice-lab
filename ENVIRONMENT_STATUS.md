# 环境部署记录

日期：2026-09-16

目录：`/home/sunteng/Desktop/HighTorque_vision/HT-voice-lab`

## 当前选型与范围

用户已选定 Streaming Sortformer v2.1，首版支持每次会话累计 1～4 个不同身份（含 4 人）。Paraformer Streaming 继续作为转写基线。默认单模型入口使用 Sortformer，默认批量入口运行 Sortformer 与 ASR；融合入口 `--backend conversation` 已按原始音频时间建立说话人／文字关联。以下 diart 部署与对比数据保留为历史记录。

## 就绪情况

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| 独立 Python 3.11.16 | 已就绪 | `.python/`、`.venv/` |
| PyTorch 2.6.0 / CUDA 12.4 | GPU 运算检查通过 | `results/environment.json` |
| FunASR 1.4.15 / Paraformer Streaming | 权重已下载，真实 1 倍速转写通过 | `results/smoke-asr/` |
| NeMo 2.6.2 / Streaming Sortformer v2.1 | 权重已下载，真实 1 倍速辨人运行通过 | `results/smoke-sortformer/` |
| diart 0.9.2 / pyannote.audio 3.3.2 | 两个权重均已下载，真实 1 倍速辨人运行通过 | `results/smoke-diart/`、`logs/embedding-authorized-fetch.log` |
| 音视频转码 | 官方 WAV 转码入口通过 | `data/prepared/asr_example-8ab66f147266.wav` |
| 多人测试素材 | 共五份 16kHz 单声道 PCM16 音频；当前使用 3 / 4 人片段，5 / 6 人素材归档；清单用 in_scope 标明范围 | `data/samples.json`、`data/raw/cctv-*.source.json` |
| 人工区间初稿 | 生活篇 A→B→C→D→A→E；产业篇 A→B→C→D→A→E→F；整秒范围，尚不能直接作为正式 DER 参考 | `data/reference/speaker-turns.draft.json` |
| 多人首轮流式实验 | 3 人、4 人片段各完成 diart、Sortformer、ASR 的 1 倍速运行；有初步身份核对和速度，无正式 DER / CER | `BENCHMARK_20260916.md`、`results/cctv-life-3spk-20260916/`、`results/cctv-industry-4spk-20260916/` |
| 辨人 + 转写融合 | 两段 1 倍速验证通过；增量文字归属 A～D，A 返回保持编号；原始辨人分段与旧基线一致 | `FUSION_20260916.md`、`results/fusion-verification.json` |
| 融合边界与兼容性自检 | 换人／重叠 PCM 隔离、自然停顿、尾字排出、ASR 缓存隔离、事件版本和英文块间空格通过；独立模型短样本回归一致 | `logs/selftest-fusion.log`、`results/smoke-sortformer-after-fusion/`、`results/smoke-asr-after-fusion/` |
| DER / CER 与音频时间边界 | 自检通过 | `logs/selftest.log` |
| Sortformer 分块音频特征 | 与参考特征一致性检查通过 | `logs/selftest.log` |
| 批量实验与 CSV 汇总 | 三项均以真实 1 倍速完成；早期缺权重失败记录另行保留 | `results/smoke-comparison-ready/comparison.csv`；历史记录 `results/smoke-comparison/` |

## 本机配置

- NVIDIA GeForce RTX 4060，8188 MiB 显存，驱动 570.133.07。
- 24 个 CPU 逻辑线程；运行入口默认每个模型使用 4 个 CPU 线程。
- 项目环境约 6.7 GiB，Python 约 97 MiB，已下载权重约 1.4 GiB；不包含共享下载缓存。
- diart 依赖旧版 matplotlib 接口，已固定 `matplotlib==3.8.4`。全部 238 个已安装包的依赖兼容性检查通过。
- 完整版本固定在 `requirements/locked.txt`；原 `HT-agent` 工作区无修改。

## 短音频运行证据

使用 Paraformer 仓库自带、长度约 **5.547 秒**的单人音频，以 1 倍速回放。

- Paraformer 输出：`欢迎大家来体验达摩院推出的语音识别模型`。
- Sortformer 和 diart 均完成逐块运行，各输出一个说话人编号 `speaker0`。
- diart 完整流程使用真实 segmentation / embedding 权重，输出发言区间约为 0.892～5.225 秒；记录在 `results/smoke-diart/segments.json`，尚未做人工时间边界核对。
- 三套模型均生成了增量事件、完整输出和摘要，状态为 `complete`。
- 这只能说明本机环境与流式入口能运行，**不能据此判断多人辨人准确度或长期性能**。

实际数字保留在各自 `summary.json`。其中首次输出时间包含录音开头静音、模型分块等待和首块内核初始化；未预热。最新三项批量运行位于 `results/smoke-comparison-ready/`，均为真实 1 倍速；早期批量自检的 `--fast` 结果不作为实时延迟证据。

## 当前验证与后续工作

本机授权、权重下载和 diart 完整推理验证均已完成。旧权重加载时会提示训练版本不同，实际 GPU 推理已通过；原始提示保留在 `logs/smoke-diart.log`。

已按用户最新逐人区间修正素材：共同对比使用生活篇前 73 秒（预期 3 人）和产业篇前 102 秒（预期 4 人，含 A 再次发言）。原产业篇前 161 秒修正为 5 人，产业篇完整版修正为 6 人；生活篇完整版仍为 5 人。转码格式、时长和裁剪前段与完整版逐采样一致性已检查；素材与使用说明见 README。原视频和来源校验信息均保留。

已有用户提供的整秒说话人区间初稿，尚无核对语音起止、停顿和重叠后的 RTTM 或转写参考。初步模型运行可核对说话人顺序及实时性能，正式 DER / CER 需补全精细参考标注。产业篇含英文，应按语言拆分转写评估。长录音的运行稳定性、说话人编号持续性和多次运行的速度波动仍需后续测试。

首轮多人实验已完成：默认 Sortformer 在两段中输出 3 / 4 个编号，产业篇 A 返回时保持原编号；历史 diart 实验输出 1 / 2 个编号，存在合并身份和 A 返回时换号的问题，具体原因未确定。两套辨人结果发布延迟 P95 分别约为 diart 0.517 秒、Sortformer 0.624 秒；这不是正确识别或话轮结束的响应时间。每段每个模型运行一次，未调参、未预热，详见 [首轮测试报告](BENCHMARK_20260916.md)。

融合已完成：先等待 Sortformer 提供音频归属，再按连续片段转写原始 PCM。默认连续 0.8 秒无活跃说话人时收尾 ASR，`turn_complete` 保持 null，未实现语义话轮完成判断。两个模型同时驻留本机 GPU；实测首次出字约 1.58 秒、计算 RTF 约 0.10、PyTorch 峰值显存约 1382 MiB。具体结果和指标解释见 [融合验证报告](FUSION_20260916.md)。

逐说话人语义话轮、显式点名回复和当前会话记忆模块已在后续章节完成。当前中文同音字及英文转写质量仍需改善；没有正式 DER / CER。麦克风、语音回复和意图图可视化尚未接入。当前已收敛到最多 4 人，diart 调参和 5～6 人测试不在当前任务范围内。

## 常用入口

```bash
./lab doctor --imports
./lab selftest
./lab prepare "/路径/会议.mp4"
./lab compare "data/prepared/转码后的文件.wav"
./lab run "data/prepared/转码后的文件.wav" --backend conversation
```

更详细的使用和指标定义见 [README.md](README.md)。

## 2026-09-16：Qwen 中英文候选验证

新增独立 `.venv-qwen/`（Python 3.12.14、qwen-asr 0.0.6、vLLM 0.14.0、Torch 2.9.1/CUDA 12.8），不更改原 `.venv/`。权重已固定 revision 并完成 SHA-256 校验，212 个包的依赖检查通过，环境约 10 GiB。安装入口：`bash setup-qwen.sh`。

已完成两段素材、三组配置的共六次完整 1 倍速运行。最终候选使用 1 秒块、合成静音预热、前 4 秒完整修订窗口和 512 MiB KV 缓存；首次出字均约 2.12 秒、RTF 约 0.14。原 Sortformer 分段与文字音频归属均保持一致；标点和英文可读性改善，但仍有同音词、重复词及未核对的人名，没有正式 CER/WER。

2026-09-18 已完成 12/15 秒有界声学窗口和 1.5 秒预卷去重。相同长英文样本的整卡峰值从 7619 MiB 降至 4464 MiB，最低空闲显存从 197 MiB 增至 3351 MiB；首次出字约 2.15 秒，RTF 约 0.158。原始窗口写入 `utterances.json`，合并后的可读声学跨度写入 `speech_spans.json`，所有 `turn_complete` 保持 null。SenseVoiceSmall 234M 已完成独立候选筛选，明显更轻更快，但两段素材的中英文准确度未超过 Qwen。详见 [显存与轻量模型复核](ASR_MEMORY_20260917.md)。

完整结果、命令和限制见 [QWEN_ASR_20260916.md](QWEN_ASR_20260916.md)。当前用 `--backend conversation --asr-model qwen3` 启用；不传模型选项仍保留 Paraformer 基线。

## 2026-09-18：每说话人语义状态与回复门控

新增独立命令 `./lab analyze-dialogue speech_spans.json`。它为 A/B/C/D 分别记录待判断贡献、被打断与续说状态；换人、静音和声学窗口不会自动令 `turn_complete=true`。本地点名门控要求同时出现助手名字和请求表达，支持同一说话人的跨片段点名问题；有标点时本地确认完整性，无标点时由语义模型确认。GroqCloud `qwen/qwen3.8-27b` 已完成真实调用：中英素材 6/6 返回、平均云端请求约 0.69 秒且 `reply_allowed=false`；22 条定向开发集为 22/22；无标点、被打断后续说的点名请求端到端通过。密钥保存在用户配置目录，权限为 0600，未写入项目。当前网络通过本机 HTTPS 代理访问 Groq。详见 [语义话轮与回复门控](SEMANTIC_TURN_20260918.md)。

## 2026-09-20：当前会话记忆与 VoiceMem 评审

新增 `./lab build-memory dialogue_state.json`，只压缩已完成的个人语义话轮，未完成发言按说话人保留在 `working_memory`。压缩结果含逐人摘要、言语行为、意图、立场、话题、实体及整体场景状态，并以贡献 ID、原文哈希和源文件哈希保留审计链。真实产业篇生成 6 条话轮记忆，无虚构决定和行动项，压缩语义文本约为原始的 56%，单次云端压缩约 1.9 秒。VoiceMem 的上层索引和流式预取思想适合后续阶段，但整包音频依赖与现有链路重复，因此当前没有安装或复制其源码。详见 [当前会话记忆与 VoiceMem 评审](MEMORY_AND_VOICEMEM_20260920.md)。
