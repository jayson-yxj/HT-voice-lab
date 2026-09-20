# 首轮流式测试：2026-09-16

后续决策：用户已选定 Sortformer，首版范围为每场会话最多 4 个不同身份（含 4 人）。本文保留选型前的测试事实；末尾列出的 diart 排查和 5～6 人测试不再属于当前工作。当前入口和范围见 [README](README.md)。

## 初步结果

用户最新区间表确认：生活篇完整版为 5 人，产业篇完整版为 6 人。共同对比使用生活篇前 73 秒（3 人）和产业篇前 102 秒（4 人，含 A 再次发言）。原产业篇前 161 秒保留为 5 人边界测试素材。

当前默认配置下，Sortformer 在两段中分别输出 3、4 个编号，主要区间与用户标出的身份顺序对应，产业篇 A 再次发言时保持了原编号。diart 分别只输出 1、2 个编号，存在不同人被合并的问题，产业篇 A 再次发言也换了编号。

**这是人数和粗区间核对，未计算正式 DER / CER。** 区间内部停顿、重叠和精确边界尚未标注，不能把“人数相符”写成“识别准确率 100%”。

## 条件和速度

- RTX 4060，CUDA；所有实验顺序执行、1 倍速供给音频。每段每个模型只运行一次，未预热，未调整默认参数。
- diart：5 秒历史窗口、0.5 秒步进、1 秒配置延迟，tau_active=0.6、rho_update=0.3、delta_new=1，聚类容量 8。
- Sortformer：官方 4spk v2.1，6 帧块、7 帧右上下文、188 帧 FIFO / 声纹缓存；ASR：Paraformer-zh-streaming，600ms 块。
- 参数和固定模型版本保留在每次运行的 run.json；原始事件、分段和日志与汇总位于同一目录。

| 素材 | 模型 | 输出身份数 / 预期 | 计算 RTF | 结果发布延迟 P95 |
| --- | --- | --- | --- | --- |
| 生活篇前 1:13 | diart | 1 / 3 | 0.0316 | 0.517 秒 |
| 生活篇前 1:13 | sortformer | 3 / 3 | 0.0719 | 0.623 秒 |
| 生活篇前 1:13 | asr | — | 0.0479 | 不报告（暂无字级时间戳） |
| 产业篇前 1:42 | diart | 2 / 4 | 0.0302 | 0.517 秒 |
| 产业篇前 1:42 | sortformer | 4 / 4 | 0.0717 | 0.624 秒 |
| 产业篇前 1:42 | asr | — | 0.0479 | 不报告（暂无字级时间戳） |

计算 RTF 是推理计算总耗时 ÷ 音频时长，不包含模型加载和回放等待。结果发布延迟是输出时刻相对其覆盖音频末端的滞后，不是识别出正确说话人的耗时，也不是语义话轮结束判断耗时。未预热、单次运行的数字只用于本轮初步比较。

## 按用户区间核对身份

下表取每个用户区间内预测累计时长最多的编号；并非该区间内每一帧都正确。A～F 在两个视频内分别编号，不推断跨视频身份。原始描述与区间见 [标注初稿](data/reference/speaker-turns.draft.json)。

| 素材 | 用户区间（秒） | 用户身份 | diart 主要编号 | Sortformer 主要编号 |
| --- | --- | --- | --- | --- |
| 生活篇前 1:13 | 0～8 | A | speaker0 | speaker0 |
| 生活篇前 1:13 | 9～21 | B | speaker0 | speaker1 |
| 生活篇前 1:13 | 22～73 | C | speaker0 | speaker2 |
| 产业篇前 1:42 | 0～19 | A | speaker0 | speaker0 |
| 产业篇前 1:42 | 19～30 | B | speaker1 | speaker1 |
| 产业篇前 1:42 | 31～52 | C | speaker1 | speaker2 |
| 产业篇前 1:42 | 53～99 | D | speaker0 | speaker3 |
| 产业篇前 1:42 | 100～102 | A | speaker1 | speaker0 |

产业篇的 B 区间在 Sortformer 中还包含约 0.92 秒 speaker0，D 区间包含约 0.04 秒 speaker2。由于参考是整秒粗区间，暂不将这些交界差异定性为模型错误；需回听细标。

## 原始结果

- 生活篇：[汇总 CSV](results/cctv-life-3spk-20260916/comparison.csv)、[diart 分段](results/cctv-life-3spk-20260916/diart/segments.json)、[Sortformer 分段](results/cctv-life-3spk-20260916/sortformer/segments.json)、[转写](results/cctv-life-3spk-20260916/asr/transcript.txt)。
- 产业篇：[汇总 CSV](results/cctv-industry-4spk-20260916/comparison.csv)、[diart 分段](results/cctv-industry-4spk-20260916/diart/segments.json)、[Sortformer 分段](results/cctv-industry-4spk-20260916/sortformer/segments.json)、[转写](results/cctv-industry-4spk-20260916/asr/transcript.txt)。

## 后续要核对的事项

- 细标两段音频的语音起止及参考文字，再计算 DER / CER。英文区间单独核对，当前 ASR 为中文基线。
- 排查 diart 的聚类阈值、声纹输出与模型兼容性；本轮观察尚不足以认定具体原因。调参实验单独记录，不覆盖这轮默认配置结果，也不将同素材调参后的结果当作独立测试集成绩。
- 两份 5 人素材及 6 人完整版尚未运行。它们超过当前 Sortformer 的 4 人容量，应独立作为边界测试报告。
