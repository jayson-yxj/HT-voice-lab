# HT Voice Lab

独立的中文流式转写与说话人区分测试环境，与旁边的 `HT-agent` 分开运行。

## 已确定的首版方案

2026-09-16，用户根据首轮测试选定 **Streaming Sortformer v2.1**。每次会话支持 **1～4 个不同的说话人（含 4 人）**，上限按整场会话累计身份计，使用会话内编号。

- 说话人区分：Sortformer；融合入口再用 CPU pyannote 声纹稳定会话身份。转写保留 Paraformer Streaming 中文基线，Qwen3-ASR-0.6B 为当前中英文候选，SenseVoiceSmall 为轻量备选。
- 日常测试使用生活篇前 1:13（3 人）和产业篇前 1:42（4 人，含 A 再次发言）。清单 `data/samples.json` 的 `in_scope` 标记当前范围。
- `./lab run 音频.wav` 默认运行 Sortformer；`./lab compare 音频.wav` 顺序运行 Sortformer 和 ASR。diart 代码、权重及首轮数据作为历史实验保留；5～6 人素材归档，当前开发不包含 diart 调参或超 4 人测试。
- 融合入口为 `./lab run 音频.wav --backend conversation`：按同一音频时钟运行 Sortformer，经会话内声纹稳定后，将已确定归属的原始 PCM 分段送入 ASR，增量输出“谁在什么时候说了什么”。独立实验入口继续用于回归比较。
- 后续在独立项目内接入每个人的话轮状态、显式点名回复、语音合成、会话记忆和观点／意图图。话轮完成按具体说话人判断；ASR 片段结束、其他人发言或全场静音都不能单独当作“这个人讲完了”。

## 当前状态

- Python 3.11、CUDA PyTorch、FunASR、diart、pyannote.audio、NeMo 已安装，依赖和导入检查通过。
- Paraformer Streaming、Streaming Sortformer v2.1 和 diart 所需的两个 pyannote 模型均已下载，在本机 RTX 4060 上完成短音频、1 倍速推理验证。
- Hugging Face 两个 pyannote 仓库的授权和本机登录均已完成；diart 完整推理结果见 `results/smoke-diart/`。
- 已准备两段央视多人视频、五份测试 WAV 和用户提供的说话人区间初稿；正式 DER / CER 参考标注仍待核对。
- 3 人、4 人片段已各完成三项 1 倍速实验。默认配置下 Sortformer 分别输出 3 / 4 个编号，diart 输出 1 / 2 个；详细的区间核对和速度见 [首轮测试报告](BENCHMARK_20260916.md)，不作为正式准确率。
- Sortformer + ASR 融合已完成两段 1 倍速验证：输出 A～D、原始音频范围和增量文字，产业篇 A 返回时保持编号。结果与限制见 [融合验证报告](FUSION_20260916.md)。
- Qwen 已加入 12/15 秒有界声学窗口和 1.5 秒预卷去重；中英长样本整卡峰值从 7619 MiB 降至 4464 MiB，首次出字约 2.15 秒。SenseVoiceSmall 的独立 15 秒窗口筛选峰值约 1719 MiB，但当前中英文准确度不如 Qwen 稳定。详见 [显存与轻量模型复核](ASR_MEMORY_20260917.md)。
- 已新增独立的每说话人语义状态与回复门控：其他人讲话、静音和声学窗口都不会自动结束某人的话轮；只有完整的显式点名请求才允许回复。普通语义结束仍等待接入模型。详见 [语义话轮与回复门控](SEMANTIC_TURN_20260918.md)。
- 已加入临时槽位稳定层：1.5 秒开始比对声纹、每 0.5 秒复核、持续 4 秒的新槽位才创建新人；不足时长的孤立槽位保留转写和审计，但不创建人物。短暂槽位允许以 0.30 阈值合并，持续槽位须达到 0.45 才能合并，避免把音色相近的真实新人错误折叠。两条双人和一条四人长样本分别稳定为 2 / 2 / 4 人，详见 [英文长会话测试](ENGLISH_LONG_SESSION_20260920.md)。

详细检查结果见 [ENVIRONMENT_STATUS.md](ENVIRONMENT_STATUS.md)。

## 先把你的音视频放在哪里

可以放进 `data/raw/`，也可以直接传入外部文件路径。支持 FFmpeg 能读取且包含音轨的音视频。

```bash
cd /home/sunteng/Desktop/HighTorque_vision/HT-voice-lab
./lab prepare "/完整路径/多人会议.mp4"
```

程序输出转码后的 WAV 路径。它会保存为 16kHz、单声道 PCM，并记录原文件 SHA-256、裁剪起点和时长。

选取某一段：

```bash
./lab prepare "/完整路径/多人会议.mp4" --start 30 --duration 120
```

标注时间以**裁剪后 WAV 的第 0 秒**为起点。相同文件、相同裁剪参数复用同一转码结果。

## 已准备的多人测试素材

来源：[人形机器人何时走进我们的生活？](https://tv.cctv.com/2025/05/27/VIDEW9BOAgdCfgiRLv0fpgIc250527.shtml)、[机器人产业诞生的土壤是什么？](https://tv.cctv.com/2025/05/27/VIDE7L5yhFV1CRkLNFjK3a0j250527.shtml)。原视频保存在 [生活篇 MP4](data/raw/cctv-robot-life.mp4) 和 [产业篇 MP4](data/raw/cctv-robot-industry.mp4)，对应 `.source.json` 记录来源、下载时间和文件校验值。

| 样本 | 原片范围 | 预期人数 | 语言 | 测试用途 |
| --- | --- | --- | --- | --- |
| [生活篇前段](data/prepared/cctv-robot-life-f511f6d1a0cb.wav) | 0:00～1:13 | 3 | 中文 | 当前 Sortformer / ASR 测试 |
| [产业篇四人段](data/prepared/cctv-robot-industry-30e19903a3e0.wav) | 0:00～1:42 | 4 | 原片中英混合 | 当前 Sortformer / ASR 测试，包含 A 再次发言 |
| [生活篇完整版](data/prepared/cctv-robot-life-66f0b0734228.wav) | 0:00～约 2:50 | 5 | 中文 | 归档，超出当前范围 |
| [产业篇五人段](data/prepared/cctv-robot-industry-90d3173bf487.wav) | 0:00～2:41 | 5 | 原片中英混合 | 归档，超出当前范围 |
| [产业篇完整版](data/prepared/cctv-robot-industry-df857f3c72b2.wav) | 0:00～约 3:19 | 6 | 中英混合 | 归档，超出当前范围 |

- 五份音频均为 16kHz、单声道 PCM16；裁剪片段与对应完整版的前 73 秒、102 秒或 161 秒逐采样一致。机器可读清单见 [data/samples.json](data/samples.json)。
- 人数已按用户最新逐人区间修正：产业篇前 2:41 有 5 人，完整版有 6 人。共同对比改用前 1:42，包含第一人再次发言，仍为 4 个不同身份。
- [说话人区间初稿](data/reference/speaker-turns.draft.json) 保留用户原始描述及 A～F 编号。同一视频内重复的 A 是同一个人；两个视频的字母编号独立。正式评分前需核对切点附近和每个人的实际语音起止。
- 当前 Paraformer 是中文转写基线。产业篇需核对裁剪内的中英文区间、分语言评估；不能把中英混合结果直接并入纯中文 CER 比较。
- 5 人和 6 人素材已归档，不纳入当前开发和验证范围；人数上限指整段会话的不同身份数量。
- 文件裁剪处只是输入结束，不作为“这个人已经讲完”的标注。这两段经过剪辑的节目可用于初步辨人和转写测试，现场连续对话与话轮结束仍需另测。

用户提供的区间（整秒初稿）：

| 顺序 | 生活篇 | 产业篇 |
| --- | --- | --- |
| 1 | A：0:00～0:08 | A：0:00～0:19 |
| 2 | B：0:09～0:21 | B：0:19～0:30 |
| 3 | C：0:22～1:13 | C：0:31～0:52 |
| 4 | D：1:14～1:47 | D：0:53～1:39 |
| 5 | A：1:47～1:51 | A：1:40～1:42 |
| 6 | E：1:51～2:50 | E：1:43～2:41 |
| 7 | — | F：2:42～3:19 |

这些是大致发言范围，区间内部的停顿未标注，区间之间的空隙也未确认是静音。因此先保存为 JSON 初稿，**不直接当作逐帧 RTTM 标准答案计算正式 DER**。产业篇用户记为 3:19 的终点保留原值 199 秒；实际 WAV 为 198.955 秒左右，后续导出标注时需按音频长度裁定。

初步运行可检查说话人顺序、编号保持和实时性能；正式准确度评分需要核对后的参考标注。运行命令（默认 1 倍速，Sortformer 和 ASR 依次运行）：

```bash
./lab compare data/prepared/cctv-robot-life-f511f6d1a0cb.wav
./lab compare data/prepared/cctv-robot-industry-30e19903a3e0.wav
```

## pyannote 授权（本机已完成）

`pyannote/embedding` 用于融合入口的 CPU 声纹稳定；`pyannote/segmentation` 仅用于复现历史 diart 实验。

1. 在浏览器登录 Hugging Face，分别接受 [segmentation](https://huggingface.co/pyannote/segmentation) 和 [embedding](https://huggingface.co/pyannote/embedding) 页面上的访问条款。
2. 在 Hugging Face 创建可读取这两个仓库的 Token，然后在**本机终端**执行：

```bash
./lab login
./lab fetch --only segmentation embedding
./lab doctor --imports
```

Token 通过终端交互输入，不放入命令参数、项目文件或聊天。登录使用 Hugging Face 的用户凭据缓存。权重下载固定仓库 revision，并记录本地 SHA-256。

## 运行实验

准备好 WAV 后，下面各命令都从第 0 秒开始，以 **1 倍速**向模型提供音频：

```bash
./lab run "data/prepared/你的文件.wav"
./lab run "data/prepared/你的文件.wav" --backend asr
```

默认使用 GPU，辨人默认 Sortformer，也可以显式传 `--backend sortformer`。可以使用 `--device cpu`。历史 diart 实验仍可用 `--backend diart` 单独复现，其等待参数为 `--latency`；每次参数都会写入结果文件。

一次顺序运行 Sortformer 与 ASR 两个独立实验并汇总：

```bash
./lab compare "data/prepared/你的文件.wav"
```

`compare` 按顺序运行，避免模型互相争抢 GPU。它是**独立辨人测试 + 独立 ASR 基线**，尚未将文字与说话人实时融合。首轮历史报告中的三项运行发生在本次选型之前；当前默认批量入口只运行选定的两个组件。

如果某个模型缺失，其他实验仍会执行；CSV 会标明失败，日志说明原因，命令最终返回非零状态。

仅做功能检查或吞吐量测试可加 `--fast`。此模式不等待真实播放时间，**不报告实时输出延迟**。

## 实时辨人和转写融合

```bash
./lab run data/prepared/cctv-robot-life-f511f6d1a0cb.wav --backend conversation
./lab run data/prepared/cctv-robot-industry-30e19903a3e0.wav --backend conversation
```

### Qwen 中英文流式对比入口

Qwen 使用独立的 Python 3.12 环境 `.venv-qwen/`，通过本机子进程与原环境中的 Sortformer 配合。
安装入口为 `bash setup-qwen.sh`；原环境仍由 `setup.sh` 管理。

```bash
./lab run data/prepared/cctv-robot-life-f511f6d1a0cb.wav --backend conversation --asr-model qwen3
./lab run data/prepared/cctv-robot-industry-30e19903a3e0.wav --backend conversation --asr-model qwen3
```

- 使用官方 `qwen-asr==0.0.6` / `vllm==0.14.0` 流式接口和固定版本的 `Qwen3-ASR-0.6B` 权重，自动判断语言、保留原文；不额外调用翻译或标点模型。
- 选择 Qwen 后默认每 1 秒解码一次；`--qwen-chunk-seconds 2` 可测试 2 秒块。音频仍以 20ms 输入，先经过相同的 Sortformer 路由。不传 `--asr-model qwen3` 时仍运行 Paraformer 基线。
- Qwen 默认在 12 秒后优先于说话人停顿处重建 ASR 状态，连续讲话最迟 15 秒重建，并用 1.5 秒预卷恢复切点词语；可用 `--qwen-window-pause`、`--qwen-window-limit` 和 `--qwen-window-overlap` 调整。滚动窗口保留人物编号，所有片段的 `turn_complete` 仍为 `null`。
- Qwen 在开始回放前用 2 秒合成静音预热，丢弃预热状态和文字；不读取测试音频。每个片段最初 4 秒不固定旧文本前缀，之后允许回退 5 个 token，降低过早固定错字的风险。
- Qwen 可以回改文字。事件中 `text_update=replace`、`text_delta=null`，消费端应按 `id` 和 `revision` 更新整段 `text`，不能拼接各次快照。Paraformer 的追加字段保持兼容。
- 片段仍使用原始音频时间范围；流式接口没有字级时间戳。标点和 `status=final` 都不表示语义话轮已结束，`turn_complete` 仍为 null。
- 运行日志保存在 `logs/qwen-worker-*.log`，具体路径和模型参数写入结果的 `run.json`。`summary.json` 原有 PyTorch 显存字段只统计父进程，**不包含 Qwen/vLLM 子进程**；查看整卡测量时应使用另行记录的显存采样数据。
- 官方接口会重新编码当前状态已经收到的音频，因此必须限制单个声学窗口。当前 12/15 秒窗口加 1.5 秒预卷的中英样本整卡峰值为 4464 MiB；预卷文字使用窗口末尾与开头的受限近似匹配去重。

### 每说话人语义状态与显式回复门控

语义状态模块独立读取融合结果，不加载音频模型：

```bash
./lab analyze-dialogue results/某次运行/speech_spans.json --assistant-alias 小派
```

默认生成同目录的 `dialogue_state.json`。未启用语义模型时，普通发言的 `turn_complete` 保持 `null`；其他人开始讲话、全场静音和 ASR 收尾都不会改变它。只有出现助手名字、明确请求表达且请求已完整时，才会写入 `reply_requests` 并令 `reply_allowed=true`。点名请求可跨多个片段延续，默认上下文为 30 秒，可用 `--invocation-context-seconds` 调整。ASR 没有问号时，本地规则负责确认点名和请求线索，语义模型负责确认请求是否已经说完整。

已提供 GroqCloud 严格 JSON 语义判定适配器，默认使用 `qwen/qwen3.8-27b`。先运行 `./lab login-groq` 在用户配置目录安全保存 Key，再增加 `--semantic-provider groq`；当前网络还需传入 `--semantic-proxy http://127.0.0.1:7890`。Key 不写入项目、日志或结果。真实中英样本 6 次判断全部返回，定向开发集为 22/22；这只是小规模冒烟测试。未选择云模型时仍使用安全的 `pending-model` 行为。具体选择依据、命令、延迟和数据策略见 [语义话轮报告](SEMANTIC_TURN_20260918.md)。

### 一条命令实时演示

下面的命令从文件音频开始，以真实时间运行辨人和转写；每个稳定个人片段完成后，依次更新语义状态、当前会话记忆和浏览器关系图：

```bash
./lab demo data/prepared/cctv-robot-industry-30e19903a3e0.wav \
  --groq-proxy http://127.0.0.1:7890
```

默认使用 Qwen3-ASR、Groq 语义判断与记忆压缩，并打开 `http://127.0.0.1:8765`。输出保存到新的 `results/*-demo/` 目录；图在音频处理期间持续读取同一份 `session_memory.json`。按 `Ctrl+C` 停止演示服务。已有服务占用端口时可传 `--port 8766`，固定结果目录可传 `--output`。

`demo` 会在加载音频模型前检查 Groq 密钥及模型权限，避免整段音频完成后才发现语义图为空。也可以单独检查：

```bash
./lab check-groq
```

只有 `speaker_change`、`silence_gap`、`overlap_boundary` 或 `eof` 等稳定片段才进入语义链路。Qwen 为限制显存产生的 `asr_window_pause` / `asr_window_limit` 人工窗口只更新转写快照，不会被误当成某个人讲完。未完成话轮保留在 `working_memory`，其他人开始发言不会替它完成。

图中的“话题”采用会话级粒度，表示多人共同讨论的问题或议程；每位人物提出的场景、供应链、成本等答案内容属于各自观点，不会再拆成独立话题。音频结束后仍需等待最后一次 `[live] 正在压缩会话记忆…` 完成；看到 `Demo complete` 后再按 `Ctrl+C`，否则最后一个话轮可能尚未写入图。

快速检查整条链路可使用较短音频、Paraformer 和非实时回放：

```bash
./lab demo data/prepared/asr_example-8ab66f147266.wav \
  --asr-model paraformer --fast \
  --groq-proxy http://127.0.0.1:7890
```

### 当前会话记忆

`build-memory` 只压缩已经由语义模型确认完成的个人话轮；被打断且尚未讲完的内容保留在逐说话人的 `working_memory`：

```bash
./lab build-memory results/某次运行/dialogue_state_groq.json \
  --memory-provider groq \
  --memory-proxy http://127.0.0.1:7890
```

输出 `session_memory.json`，包含逐人话轮摘要、言语行为、意图、立场、话题、实体，以及场景摘要、未解决问题、决定和行动项。压缩记录只保存原始贡献 ID 和文本 SHA-256，完整原文继续保存在 `dialogue_state.json`，可按人物和音频时间回溯。`--memory-provider local` 是不调用网络的结构保底，不提供真正的语义压缩。

长会话会按每 4 个完整话轮分批压缩，再合并整场摘要并统一话题名称，以适配免费 API 的单次输出限制。Groq 返回 429 时会按服务端等待时间自动重试；未分离的重叠片段保留在工作记忆中，不计作人物节点。

把记忆实时投影为“人物—话题—观点—意图”关系图：

```bash
./lab visualize-memory results/某次运行/session_memory.json
```

命令默认打开 `http://127.0.0.1:8765`。页面每秒检查文件版本；`session_memory.json` 被重新生成后自动更新，并保留每条观点的话轮编号、人物、时间、立场和原始贡献 ID。尚未完成的 `working_memory` 单独列出，不会提前固化为观点。服务只监听本机，按 `Ctrl+C` 停止；可用 `--port` 更换端口，或用 `--no-open` 禁止自动打开浏览器。

已评审 VoiceMem。当前复用其“证据保留、话轮后异步写入、轻量上层索引、流式预取”思想，不安装整包；其音频前端与本项目重复，固定依赖也会改变现有验证环境。真实 4 人样本已生成 6 条话轮记忆，压缩文本约为原始的 56%。评审、结构和结果见 [当前会话记忆与 VoiceMem 评审](MEMORY_AND_VOICEMEM_20260920.md)。

### 共用的音频路由规则

默认以 1 倍速、20ms 小块回放，保持同一音频时间轴。当前采用 **Sortformer 临时槽位 → CPU 声纹稳定 A/B/C/D → 对应 PCM 进入流式 ASR** 的流程：ASR 没有字级时间戳，按音频归属后再转写，可以避免把滞后输出的文字按输出时刻误归给下一人。代价是首次出现的人最多需要等待 4 秒声纹确认，之后同一槽位可直接复用身份。

- 单人区间使用 A / B / C / D。原始 Sortformer 槽位可通过声纹匹配到已有身份，因此不再假定 raw speaker0～speaker3 永远等同于 A～D。
- 新槽位累计 1.5 秒干净单人语音后开始声纹匹配，每 0.5 秒复核；持续发言最多等待 4 秒再创建新人。同一临时槽位允许跨越 0.8 秒以内的静音继续累计。低置信度声纹匹配只用于不足 4 秒的槽位碎片，持续槽位需要更高阈值才能复用旧身份。
- 不足确认时长的孤立槽位使用 `speaker=null`、`attribution=short_unresolved`，保留候选人与原始槽位供审计，但不会投影为新人物。可用 `--no-speaker-stabilization` 复现原始槽位路由。
- 每个连续音频片段使用独立 ASR 缓存。换人时先排出前一段剩余文字，再开始下一段；模型权重共用。
- 同一说话人的短停顿保留原始 PCM；默认连续 0.8 秒无活跃说话人时收尾当前转写片段，可用 `--silence-gap 秒数` 调整。较长停顿后的发言会生成新片段，同一人的编号保持一致。
- 模型判定多个说话人同时活跃时，混合音频只转写一次，`speaker=null`、`attribution=overlap_unresolved`，保留 `speaker_candidates`。当前没有声源分离。
- 音频时间使用整数采样位置，`start_s/end_s` 是对应音频片段的范围，包含保留的短停顿，**不是每个字的时间戳**。
- `status=partial` 是累计文字快照；相同 `id` 的 `revision` 递增。`status=final` 只表示该音频片段的 ASR 已收尾；所有记录的 `turn_complete=null`，换人、停顿和文件结束均不声称某人已在语义上讲完。
- Sortformer 的漏检、错分仍会影响后续转写归属。稳定层只回溯尚未发布的临时槽位，不能修复已经信任的 A～D 槽位漂移，也不提供声源分离或字级对齐。

### 融合输出

| 文件 / 字段 | 含义 |
| --- | --- |
| `events.jsonl` | 实时追加 `diarization_raw`、稳定后的 `diarization` 和 `transcript` 事件；转写事件内的 `utterance` 是带编号和时间范围的快照 |
| `utterances.json` | 全部已收尾的转写片段，包含 `id`、`speaker`、`text`、采样位置、秒数、归属状态及收尾原因 |
| `transcript.txt` | 按片段排列的可读文本，例如 `[0.88–5.20] A: 欢迎大家来体验…` |
| `segments.json` / `prediction.rttm` | 稳定后的会话身份区间，使用 speaker0～speaker3 命名 |
| `raw_segments.json` | 未经身份稳定的 Sortformer 原始槽位区间 |
| `speaker_stabilization.json` | 每次临时槽位出现的时间、最终身份、判定原因及声纹相似度 |
| `first_transcript_output_seconds` | 回放开始至首次非空文字输出的时间，包含开头静音与两级模型缓冲 |
| `max_pending_audio_seconds` | 等待辨人和身份确认的 PCM 缓冲峰值，不包含模型内部缓存 |

`summary.json` 的 `compute_rtf` 汇总两套模型的推理耗时；`output_delay_s` 仍只统计辨人结果发布相对音频时间的延迟，不能当成文字延迟或话轮结束延迟。两种事件共有 `available_audio_s` 和 `emitted_at_s`，可检查音频进度与实际输出时刻。

FunASR 1.4.15 将小于 60ms 的收尾输入当作缓存排出操作；融合入口会给这类非空余音补零到 60ms，避免漏掉真实尾音。补零量写入 `final_padding_samples`，不增加原始音频时间。语义话轮状态、显式点名回复、当前会话压缩记忆和实时记忆图已经独立实现；TTS 仍为后续模块。

两种 ASR 入口都保留相邻英文文本块之间的单词空格。当前中文基线仍有中文同音字错误；产业篇 D 的英文转写明显不可靠，空格修复不代表已解决英文识别。精细字级时间对齐、英文模型和识别质量评估尚待后续工作。

### 输出文件

每次运行写入新的 `results/日期时间-模型/`，不覆盖已有结果：

| 文件 | 含义 |
| --- | --- |
| `run.json` | 模型版本、参数、设备和运行条件 |
| `events.jsonl` | 每次增量结果、音频进度、输出时刻、计算耗时 |
| `summary.json` | 加载时间、计算实时率、输出延迟、内存和显存统计 |
| `segments.json` / `prediction.rttm` | 辨人的时间区间和会话内编号 |
| `utterances.json` | 原始声学 ASR 窗口；窗口完成不代表语义话轮完成 |
| `speech_spans.json` | 原子更新的稳定转写快照；合并同一说话人的人工 ASR 窗口，保留来源窗口编号和 `turn_complete=null` |
| `transcript.txt` | 合并人工窗口后的可读转写 |
| `dialogue_state.json` | 每个说话人的独立语义状态、打断／续说记录和显式回复请求 |
| `session_memory.json` | `demo` 持续更新的会话摘要、逐人观点、话题、意图、证据和待完成发言 |
| `comparison.csv` | `compare` 的汇总结果 |

### 如何理解速度数字

- `model_load_seconds`：模型构造和加载权重的时间，不包含 Python 启动、模块导入。
- `compute_rtf`：各次推理处理时间之和 ÷ 音频时长，**不包含 1 倍速等待、模型加载及结果写盘**。小于 1 表示该次实验的计算耗时少于音频时长。
- `first_nonempty_output_seconds`：回放开始后首次得到非空结果的时间，包含录音开头本身的静音，不能直接当成“有人开口后的响应时间”。
- `output_delay_s`：辨人结果输出时刻，减去本次结果覆盖到的音频时间末端。它表示结果发布时间的滞后，**不是话轮完成判断耗时，也不是人工验证过的正确说话人响应时间**。
- ASR 暂无字级时间戳，所以这个通用输出延迟字段为 `null`；原始事件仍记录输入进度与文字输出时刻。
- 第一段推理包含 CUDA 内核初始化；日志明确 `warmup_performed=false`。正式比较要在相同启动条件下运行，并结合逐块时间查看首块与后续块差别。
- PyTorch 的峰值显存统计包含其管理的内存，不等于整张显卡被所有进程占用的总量。

## 计算准确度

### 说话人区分：DER

需要人工核对的 RTTM，覆盖所测完整片段；未标注时间会视为无人讲话。RTTM 每行格式：

```text
SPEAKER audio 1 0.500 2.000 <NA> <NA> A <NA> <NA>
SPEAKER audio 1 2.700 1.500 <NA> <NA> B <NA> <NA>
```

这里 `0.500` 是开始秒数，`2.000` 是**持续时间**。`A/B` 可任意命名，评分会匹配匿名编号。重叠发言用重叠区间表示。

```bash
./lab score \
  --reference data/reference/会议.rttm \
  --hypothesis results/某次运行/prediction.rttm \
  --output results/某次运行/accuracy.json
```

默认 `collar=0`，包含重叠语音。可显式使用 `--collar 0.25` 或 `--skip-overlap`，比较双方必须用相同口径。没有标注时不生成准确率数字。

### 中文转写：CER

```bash
./lab score-asr \
  --reference data/reference/会议.txt \
  --hypothesis results/某次ASR运行/transcript.txt
```

CER 为字错误率。使用 NFKC 规范化、统一小写、去掉空白和 Unicode 标点；不自动转换繁简体、数字写法或纠正名字。输出同时给出替换、删除、插入字数。

## 当前模型与流式实现

| 组件 | 固定选择 |
| --- | --- |
| 说话人区分（已选定） | `nvidia/diar_streaming_sortformer_4spk-v2.1`；6帧块、7帧右上下文、188帧 FIFO/声纹缓存；80ms一帧 |
| 会话身份稳定 | CPU `pyannote/embedding`；1.5s 起匹配、0.5s 复核、4s 新人确认上限、短片段/持续片段阈值 0.30/0.45、最多4人 |
| 转写基线 | `funasr/paraformer-zh-streaming`，600ms 块 |
| 历史 diart 实验 | `diart 0.9.2` + `pyannote/segmentation` + `pyannote/embedding`；5s 历史窗口、0.5s 步进、默认1s延迟；会话聚类容量8 |

精确权重提交见 `models.json`，本地文件校验记录在 `models/*/download.json`，依赖完整版本见 `requirements/locked.txt`。

Sortformer 保留跨块 speaker cache；仅从已经到达的 PCM 提取特征，音频缓冲大小有界。官方低延迟配置有 1.04s 输入缓冲，另外保留 20ms STFT 边缘上下文。测试已校验分块特征与参考特征一致。

diart 开头用左侧零填充提供历史窗口，文件末尾用零填充排出延迟结果；末尾补齐事件标记 `final_flush`，不混入稳态延迟统计。两者均裁掉超出原音频时长的输出。

**首版支持最多 4 个会话身份，与当前选定 Sortformer 的容量一致。** 当前工作集中在 1～4 人场景。

## 环境检查与复现

```bash
./lab doctor --imports
./lab selftest
```

在兼容的 Linux x86_64 / NVIDIA 主机上重建：

```bash
bash setup.sh
./lab fetch --only paraformer sortformer
```

环境位于项目内 `.python/` 和 `.venv/`。`lab` 启动器隔离已有 ROS 的 `PYTHONPATH` 和系统 `LD_LIBRARY_PATH`。旧版 pyannote 检查点通过项目启动器使用兼容的 PyTorch 加载行为，仅加载固定官方仓库权重。

`doctor` 的退出码检查依赖与导入，权重状态单独显示在 `models_ready` 中。

## 官方资料

- [FunASR](https://github.com/modelscope/FunASR)
- [diart](https://github.com/juanmc2005/diart)
- [Sortformer 模型卡](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)
- [NeMo 2.6.2 流式实现](https://github.com/NVIDIA-NeMo/NeMo/blob/v2.6.2/nemo/collections/asr/models/sortformer_diar_models.py)
