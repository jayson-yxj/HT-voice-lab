# 每个说话人的语义话轮与回复门控

日期：2026-09-18。

## 当前实现

新增独立的 `analyze-dialogue` 阶段，输入 ASR 生成的 `speech_spans.json`，输出 `dialogue_state.json`。该模块不加载音频模型，也不与 Sortformer/Qwen 耦合。

它为 A/B/C/D 分别维护：

- `semantic_state`：`unknown` 或由语义判定器确认的 `complete`；
- `acoustic_state`：当前暂停或被其他说话人打断；
- `pending_contribution_ids`：该说话人仍需语义判断的发言；
- `interruption_count` / `resumption_count`：被打断和回来续说的次数；
- `reply_state`：是否产生了明确的助手回复请求。

当 B 开始讲话时，A 只进入 `suspended`，A 的 `semantic_state` 仍保持 `unknown`。全场静音、换人和声学窗口结束也不会自动把任何人的话轮标为完成。

## 回复门控

默认助手名字为“小派”。只有同一说话人的上下文同时满足以下条件，才生成 `reply_requests`：

1. 明确出现助手名字；
2. 出现“你觉得呢”“你怎么看”“请回答”“帮我”等请求表达；
3. 请求在语言上已经完整。标点存在时本地规则可直接确认；ASR 没有标点时，由语义模型结合该说话人的连续贡献确认。

点名和问题可以跨多个声学片段，也允许中间被另一个说话人短暂打断。默认保留 30 秒点名上下文。以下内容不会触发回复：

- “小派参与了这个项目。”
- “大家觉得小派怎么样？”
- 只有“小派？”而没有具体问题；
- 没有出现“小派”的普通问题。

回复门控与语义结束判断分开。即使语义模型认为某人的发言已经完整，没有显式点名和请求线索也不会允许系统回复。模型确认“小派参与了昨天的测试”是完整陈述后，本地门控会清除这次名字上下文，避免该说话人之后未点名的问题被误认为在叫助手。

## 语义模型接口

`DialogueStateTracker` 接受可替换的 `semantic_judge`，接口为：

```python
semantic_judge.assess(
    speaker="A",
    contribution_ids=["contribution-000001"],
    text="当前新增的发言",
    close_reason="silence_gap",
    context=[{"id": "contribution-000001", "text": "当前新增的发言"}],
)
```

返回：

```json
{
  "turn_complete": true,
  "confidence": 0.92,
  "reason": "观点和请求均已完整",
  "decision_source": "模型名称"
}
```

CLI 已实现可选 GroqCloud 适配器，默认模型为 `qwen/qwen3.8-27b`，使用严格 JSON Schema 返回 `complete / incomplete / uncertain`、置信度和简短原因。未指定 `--semantic-provider groq` 时，普通发言仍返回 `turn_complete=null` 和 `decision_source=pending-model`。显式点名与请求线索始终由本地门控确认；无标点请求的完整性可由语义模型补充确认。

### 云模型选择

当前首选 GroqCloud `qwen/qwen3.8-27b`：

- 官方当前免费限额为 30 RPM、1000 RPD、8000 TPM、200000 TPD，足以按声学跨度做开发评测；
- 该模型在 Groq 支持 `strict: true` 的结构化输出，服务端约束结果符合 JSON Schema；
- 普通推理内容默认不保留，控制台可启用 Zero Data Retention；
- API 与 OpenAI Chat Completions 格式兼容，适配器只使用 Python 标准库。

官方资料：[结构化输出](https://console.groq.com/docs/structured-outputs)、[免费限额](https://console.groq.com/docs/rate-limits)、[数据保留](https://console.groq.com/docs/your-data)。Gemini 免费层虽也支持结构化输出，但官方条款允许将免费服务的输入输出用于改进产品，因此不作为会议文本的首选：[Gemini 免费层条款](https://ai.google.dev/gemini-api/terms)。阿里云 Model Studio 的国际免费额度为限时额度，当前定价页标注有效期为 90 天：[Model Studio 定价](https://www.alibabacloud.com/help/en/model-studio/model-pricing)。

## 使用

```bash
./lab analyze-dialogue \
  results/cctv-industry-qwen-overlap-15s-v2/speech_spans.json \
  --assistant-alias 小派
```

使用 Groq 语义判定时，在本机终端安全保存 Key；输入不会回显，文件保存到用户配置目录并设置为 `0600`，不要把 Key 写进项目或聊天：

```bash
./lab login-groq
./lab analyze-dialogue \
  results/cctv-industry-qwen-overlap-15s-v2/speech_spans.json \
  --assistant-alias 小派 \
  --semantic-provider groq \
  --semantic-proxy http://127.0.0.1:7890
```

当前网络直连 `api.groq.com` 会在鉴权前被 Cloudflare 拒绝，因此本机测试通过现有的 `127.0.0.1:7890` HTTPS 代理。也可以设置 `GROQ_HTTPS_PROXY`。批量离线分析可增加 `--semantic-minimum-interval 6.1`；实时输入通常由人类发言时长自然拉开请求，不应人为增加这段等待。

适配器把 `max_completion_tokens` 限制为 160。创建的免费账户实际返回 1000 OTPM（每分钟输出 token）限制；未限制单次输出时，服务端会按默认预期输出量拒绝请求。160 足以容纳当前严格 JSON 结果，也让短时间内的多次判断可用。

实际中英素材的 6 条贡献均完成真实云端判断，无错误；6 条均为完整发言，平均云端请求时间约 0.69 秒，仍为 0 个回复请求，符合素材中没有人点名“小派”的事实：

- [Groq 状态输出](results/cctv-industry-qwen-overlap-15s-v2/dialogue_state_groq.json)
- [输入声学跨度](results/cctv-industry-qwen-overlap-15s-v2/speech_spans.json)

22 条人工编写的小型开发集覆盖中文、英文、无标点、未完条件句、未完枚举和跨片段续说，结果为 22/22，平均云端请求时间约 0.65 秒、P95 约 0.96 秒。这个结果只说明当前规则和提示词通过定向冒烟测试，不能当作真实会议准确率：

- [开发集](fixtures/semantic_turn_eval.json)
- [评测结果](results/semantic-groq-qwen3.8-27b-eval.json)

端到端合成场景验证了 A 被打断后继续、普通名字提及不触发、D 的无标点请求被 B 打断后继续。系统只生成 D 的一次回复请求，文本为“小派你觉得这个方案应该怎么改”：

- [场景输入](fixtures/dialogue_interruption_request.json)
- [状态输出](results/dialogue-interruption-request-groq.json)

自检另外覆盖：跨说话人打断与续说、名字被第三人提及后清除、分段点名请求、仅点名无问题、语义模型完成但不允许回复。

## 下一步

真实云端接入与第一轮定向评测已经完成。当前会话记忆模块现已读取这套状态：只压缩逐人的已完成话轮，保留未完成工作记忆，并通过贡献 ID 与哈希回溯原文。详见 [当前会话记忆与 VoiceMem 评审](MEMORY_AND_VOICEMEM_20260920.md)。下一步从压缩结果生成观点、意图与话题关系图。最终 `reply_allowed` 继续由本地显式点名门控决定。
