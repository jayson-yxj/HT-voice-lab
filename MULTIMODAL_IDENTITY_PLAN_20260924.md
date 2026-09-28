# 多模态人物与证据开发顺序

日期：2026-09-24
状态：已确认；语音侧统一协议、匿名参与者和个人信息提取已完成

## 1. 模块边界

当前仍在 `HT-voice-lab` 中开发语音产生的人物上下文：

- Sortformer、声纹稳定和 ASR 继续回答“会话内谁在什么时候说了什么”；
- 新的独立后处理模块把 A/B/C/D 转成匿名参与者，并从已完成话轮提取带证据的个人信息候选；
- 不修改现有 `session_memory.json`，避免影响实时图和已经通过的回归；
- 新结果计划单独写入 `participant_context.json`；
- 手动创建、命名和绑定人物不作为正常流程，只保留为后续纠错入口。

未来的 `HT-vision-lab` 独立输出人脸轨迹、活跃说话人和视觉观察。等语音与视觉分别验证后，再决定由 `HT-agent` 还是独立融合模块负责跨模态和跨会话身份；现在不提前创建共享项目。

## 2. 固定开发顺序

1. **统一多模态人物与证据协议。**
2. **从 A/B/C/D 自动建立匿名参与者。**
3. **自动提取对话中的个人信息，未知字段保持为空。**
4. 创建独立的 `HT-vision-lab`。
5. 完成人脸轨迹和活跃说话人检测。
6. 融合 A/B/C/D 与 Face-01/02/03。
7. 加入关键帧场景和二维空间关系理解。
8. 建立跨会话 `person_id` 和长期个人档案。
9. 增加人工纠错、拆分错误合并、编辑和删除入口。

步骤 1–3 已完成。步骤 4–9 不提前混入。

## 3. 下一切片的输入与输出

输入仍使用已有文件：

```text
session_memory.json
  └─ source.dialogue_state → dialogue_state.json 中的完成话轮原文
```

不能只分析压缩摘要。个人姓名、地点和偏好可能没有进入摘要，所有结论也必须回到原始完成话轮。

计划输出：

```json
{
  "schema_version": 1,
  "session_id": "session-001",
  "participants": [
    {
      "participant_id": "session-001:participant:A",
      "speaker_label": "A",
      "person_id": null,
      "display_name": null,
      "identity_status": "anonymous",
      "observation_ids": [],
      "evidence_ids": []
    }
  ],
  "observations": [],
  "evidence": [],
  "associations": []
}
```

四类核心对象：

- `Participant`：会话内匿名参与者；
- `Observation`：文本、音频或未来视觉产生的候选信息；
- `Evidence`：原始话轮、音频时间段或未来视频帧；
- `Association`：未来连接说话人轨迹、人脸轨迹和人物的带置信度关系。

## 4. 身份和信息规则

- A/B/C/D 只在当前会话有效，不是永久人物 ID；
- 同一会话中的相同字母必须稳定映射到同一个 `participant_id`；
- 不同会话中的 A 默认是不同参与者；
- `unresolved`、重叠且无法归属的片段不能创建确定人物；
- 姓名相同不能触发人物合并；
- 未获得的姓名和属性保持 `null` 或空数组，模型不得补全；
- 本人清晰陈述标记为 `explicit`，系统根据重复行为概括的倾向标记为 `inferred`；
- 本人明确说出的性格描述仍是 `explicit`，但只属于会话候选，不自动成为长期确认事实；
- 他人对某人的评价不能写成被评价者的确认属性；
- 每条观察必须引用同一说话人的完成话轮，并保存逐字证据；
- 同一人修正旧信息时保留旧证据并标记 `superseded`，不能静默覆盖。

首版属性种类：

```text
name / role / occupation / location / experience
preference / goal / relationship / communication_style / trait
```

## 5. 开发前验收夹具

固定夹具位于 `fixtures/personal_identity_eval.json`，覆盖：

1. 四人会话中的明确自述与完全未知人物；
2. 两个不同参与者使用相同姓名；
3. 转述他人信息、猜测和直接点名不造成错误绑定；
4. 地点等事实发生明确修正；
5. 本人性格自述与系统交流风格推断分开。

`./lab selftest` 会先检查夹具自身是否满足证据和身份约束。后续 Groq 提取器必须使用同一份夹具做效果评测，不能为了某次真实视频临时更改预期答案。

## 6. 下一实现的完成条件

- 2～4 人会话自动产生等量匿名参与者；
- 没有资料的人所有未知字段保持为空；
- 每条信息都能回溯到相同说话人的逐字证据；
- 转述、猜测、点名和同名不会导致错误归属或合并；
- 输出是独立文件，现有转写、`session_memory.json` 和关系图完全不变；
- 确定性回归通过后，再接 Groq 真实提取并报告准确结果。

## 7. 当前进度

2026-09-24 已完成前三项：

- 正式协议：`schemas/participant_context.schema.json`；
- 命令：`./lab build-participant-context session_memory.json`；
- 自动生成会话内匿名参与者、稳定 `participant_id` 和完成话轮原文证据；
- 校验 `dialogue_state.json` 与每条完成话轮的 SHA-256，拒绝被修改或跨人物的证据；
- 不启用模型时 `observations` 和 `associations` 保持为空，不伪造个人资料；
- 四人真实样例得到 A/B/C/D 四个匿名参与者和 10 条完成话轮证据。
- 启用 Groq 时按人物独立提取个人资料候选；未知字段保持空，所有观察必须通过同人物逐字引文校验；
- `demo` 已异步增量更新 `participant_context.json`，网页人物节点显示个人信息、回答证据和必要的问答上下文；
- 冻结夹具的 15 个匿名人物状态、19 个属性槽位和 20 条证据全部匹配，禁止项命中为 0；不同但等价的值表述仍保留原文语义，不强行改写成固定词形。

下一项是创建独立 `HT-vision-lab`，先固定视觉输出协议，再实现人脸轨迹和活跃说话人检测。
