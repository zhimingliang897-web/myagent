"""
辩论流程引擎 — 编排4个阶段、调度各辩手发言、管理上下文、裁判评分

辩论流程:
  阶段1: 开篇立论 — 正一 → 反一
  阶段2: 攻辩质询 — 正二质询反一 → 反二质询正一 → 正一质询反二 → 反一质询正二
  阶段3: 攻辩小结 — 正一 → 反一
  阶段4: 自由辩论 — 正反交替发言（N轮）
  阶段5: 总结陈词 — 反二 → 正二
  裁判点评 + 评分 + 宣布胜负
"""

import time
import threading
import config as _cfg
from config import JUDGE, MAX_HISTORY
from llm_client import LLMClient


SIDE_LABELS = {"pro": "正方", "con": "反方"}


class DebateEngine:
    """辩论引擎：管理完整辩论流程，支持运行时配置覆盖"""

    def __init__(self, topic: str, pro_position: str, con_position: str,
                 debaters=None, providers=None, params=None):
        self.topic = topic
        self.pro_position = pro_position
        self.con_position = con_position
        self.history = []       # [{"id", "name", "side", "role", "content", "phase"}]
        self.clients = {}       # debater_id -> LLMClient
        self.debater_map = {}   # debater_id -> debater dict
        self.finished = False
        self._start_time = None
        # 同步锁：初始为 Set (允许第一步运行)，之后每次运行一步后 Clear (暂停)
        self.step_event = threading.Event()
        self.step_event.set()

        # 运行时配置（可由前端覆盖）
        self._debaters = debaters if debaters is not None else _cfg.DEBATERS
        self._providers = providers if providers is not None else _cfg.LLM_PROVIDERS

        params = params or {}
        self._max_words = int(params.get("max_words", _cfg.MAX_WORDS))
        self._free_debate_rounds = int(params.get("free_debate_rounds", _cfg.FREE_DEBATE_ROUNDS))
        self._debate_time_limit = float(params.get("debate_time_limit", _cfg.DEBATE_TIME_LIMIT))

    def _is_timeout(self) -> bool:
        """检查是否超过辩论时间上限（0 表示不限制）"""
        if self._start_time is None or self._debate_time_limit <= 0:
            return False
        return (time.time() - self._start_time) >= self._debate_time_limit

    def stop(self):
        """强制停止辩论"""
        self.finished = True
        self.step_event.set()  # 解锁以便循环可以退出

    def _elapsed_str(self) -> str:
        """返回已用时间的可读字符串"""
        if self._start_time is None:
            return "0:00"
        elapsed = int(time.time() - self._start_time)
        return f"{elapsed // 60}:{elapsed % 60:02d}"

    def init_clients(self):
        """根据配置初始化每个辩手和裁判的 LLM 客户端"""
        for d in self._debaters:
            provider_key = d["provider"]
            provider = self._providers.get(provider_key)
            if not provider:
                raise ValueError(f"辩手 {d['name']} 引用了不存在的服务商: {provider_key}")

            # 支持辩手级别的模型覆盖
            model = d.get("model_override") or provider["model"]
            self.clients[d["id"]] = LLMClient(
                base_url=provider["base_url"],
                api_key=provider["api_key"],
                model=model,
            )
            self.debater_map[d["id"]] = d

        # 裁判客户端（使用 JUDGE 配置中指定的 provider）
        judge_provider_key = JUDGE["provider"]
        judge_provider = self._providers.get(judge_provider_key)
        if not judge_provider:
            # 降级：用第一个可用 provider
            judge_provider = next(iter(self._providers.values()))
        self.clients["judge"] = LLMClient(
            base_url=judge_provider["base_url"],
            api_key=judge_provider["api_key"],
            model=judge_provider["model"],
        )

    def _build_system_prompt(self, debater: dict, phase: str) -> str:
        """构建辩手的系统提示词"""
        side_label = SIDE_LABELS[debater["side"]]
        position = self.pro_position if debater["side"] == "pro" else self.con_position

        phase_rules = {
            "opening": """当前是【开篇陈词】环节。你需要：
1. 首先明确定义辩题中的关键概念，为己方争取有利的定义权
2. 提出2-3个核心论点，构建完整的论证框架
3. 用一个有力的事例或数据开场，吸引注意力
4. 语气自信、有气势，体现辩手风范""",
            "question": """当前是【攻辩质询】环节。你需要：
1. 用"请问对方辩友"开头，提出尖锐的封闭式问题
2. 问题应直指对方论证链条中最薄弱的环节
3. 设计问题陷阱——无论对方如何回答，都可以为己方所用
4. 每次只问1-2个紧密关联的问题，步步紧逼""",
            "answer": """当前是【攻辩质询】环节，你正在回应对方的质询。你需要：
1. 先正面回应问题，不要回避，否则会被裁判扣分
2. 回应后立即反击——"但我更想请对方辩友注意的是..."
3. 将对方的问题转化为对己方有利的论据
4. 保持冷静和从容，不要被对方的节奏带跑""",
            "free": """当前是【自由辩论】环节。你需要：
1. 紧盯对方上一位辩手的发言，逐点反驳，不能让对方的论点"站住"
2. 反驳后迅速拉回己方战场，抛出新的进攻点
3. 善用"对方辩友始终无法回应我方提出的……"来制造压力
4. 语言节奏要快、要短，有攻击性，这是辩论赛的高潮环节
5. 可以适当呼应队友之前的论点，体现团队配合""",
            "summary": """当前是【攻辩小结】环节。你需要：
1. 总结刚才攻辩质询阶段的交锋成果，点明己方在质询中取得的关键突破
2. 指出对方在回应质询时暴露的逻辑漏洞或回避的问题
3. 将攻辩中的零散交锋归纳为系统性结论，巩固己方论证链条
4. 语气沉稳、有说服力，像是在向裁判做阶段性汇报""",
            "closing": """当前是【总结陈词】环节。你需要：
1. 先梳理全场交锋的关键战场，指出对方始终未能有效回应的问题
2. 重新强调己方最有力的2-3个论点
3. 上升到价值层面——从具体论点升华到更深层的价值判断
4. 结尾要有力量感，用一句精炼的话收束全场""",
        }

        return f"""你是"{debater['name']}"，正在参加一场高水平大学辩论赛（华语辩论世界杯级别）。你是{side_label}{debater['role']}。

【辩题】{self.topic}
【你的立场】{side_label}：{position}
【你的辩论风格】{debater['personality']}

{phase_rules.get(phase, '')}

【辩论赛纪律】
- 每次发言严格控制在{self._max_words}字以内
- 称呼对方为"对方辩友"，称呼队友为"我方x辩"
- 必须针对对方已有的发言内容进行回应和反驳，严禁自说自话
- 论证要有层次：提出观点→给出论据（事实/数据/学理）→得出结论
- 善用反问、类比、归谬等辩论技巧增强说服力
- 语言要口语化、有节奏感，像是真人在现场辩论，不要书面化

【重要】直接输出你的发言内容，不要加任何角色标记、括号说明或前缀。不要输出"我认为"开头的书面体，要像真正站在辩论台上一样说话。"""

    def _build_messages(self, debater: dict, phase: str) -> list[dict]:
        """构建消息列表：system prompt + 历史发言 + 当前指令"""
        messages = [{"role": "system", "content": self._build_system_prompt(debater, phase)}]

        # 注入历史发言（最近 MAX_HISTORY 条）
        recent = self.history[-MAX_HISTORY:]
        for entry in recent:
            side_label = SIDE_LABELS.get(entry["side"], entry["side"])
            if entry["id"] == debater["id"]:
                messages.append({"role": "assistant", "content": entry["content"]})
            else:
                prefix = f"[{entry['name']}（{side_label}{entry['role']}）]"
                messages.append({"role": "user", "content": f"{prefix} {entry['content']}"})

        # 当前轮次指令
        instructions = {
            "opening": "请进行开篇陈词。",
            "question": "请向对方提出质询问题。",
            "answer": "请回应对方刚才的质询。",
            "summary": "请对攻辩质询环节进行小结。",
            "free": "请进行自由辩论发言。",
            "closing": "请进行总结陈词。",
        }
        messages.append({"role": "user", "content": instructions.get(phase, "请发言。")})

        return messages

    def _run_turn(self, debater_id: str, phase: str):
        """执行一个辩手的一轮发言，yield 每个 token"""
        # [Sync] 等待上一轮播放完毕的信号
        self.step_event.wait()
        # [Sync] 立即清除信号，防止跑下一轮，直到前端再次 Set
        self.step_event.clear()

        # [Check] 如果已被停止或超时，直接返回（不 yield 任何内容）
        if self.finished or self._is_timeout():
            return

        debater = self.debater_map[debater_id]
        messages = self._build_messages(debater, phase)

        full_text = ""
        try:
            for token in self.clients[debater_id].chat_stream(messages):
                full_text += token
                yield token
                # 超字数截断
                if len(full_text) > self._max_words * 2:
                    break
        except Exception as e:
            if not full_text:
                full_text = f"（{debater['name']}发言出现技术问题：{str(e)[:50]}）"
                yield full_text

        # 记录到历史
        self.history.append({
            "id": debater_id,
            "name": debater["name"],
            "side": debater["side"],
            "role": debater["role"],
            "content": full_text,
            "phase": phase,
            "voice": debater["voice"],
        })

    def _judge_evaluate(self) -> str:
        """裁判评分"""
        # [Sync] 裁判也需要等待
        self.step_event.wait()
        self.step_event.clear()

        transcript = ""
        for entry in self.history:
            side_label = SIDE_LABELS.get(entry["side"], entry["side"])
            transcript += f"\n【{entry['name']}（{side_label}{entry['role']}）- {entry['phase']}】\n{entry['content']}\n"

        prompt = f"""你是一位资深辩论赛裁判，请根据以下辩论记录进行评判。

【辩题】{self.topic}
【正方立场】{self.pro_position}
【反方立场】{self.con_position}

【辩论全文记录】
{transcript}

请按以下格式输出评判结果：

## 各辩手表现点评
（每位辩手2-3句点评）

## 评分（满分100分）

| 维度 | 正方 | 反方 |
|------|------|------|
| 论点说服力（25分） | ? | ? |
| 逻辑严密性（25分） | ? | ? |
| 反驳有效性（25分） | ? | ? |
| 语言表达（25分） | ? | ? |
| **总分** | **?** | **?** |

## 最终判定
（宣布胜方及理由，1-2句）"""

        messages = [
            {"role": "system", "content": "你是一位公正权威的辩论赛裁判。请客观评价双方表现。"},
            {"role": "user", "content": prompt},
        ]
        return self.clients["judge"].chat(messages, temperature=0.3)

    def _emit_turn(self, debater, phase):
        """执行一个辩手的发言并 yield 所有 SSE 事件"""
        # 提前检查：已停止或超时则跳过整个发言（无气泡）
        if self.finished:
            return

        yield {"event": "speaker", "data": {
            "id": debater["id"], "name": debater["name"],
            "side": debater["side"], "role": debater["role"],
        }}
        full_text = ""
        for token in self._run_turn(debater["id"], phase):
            full_text += token
            yield {"event": "token", "data": {"speaker_id": debater["id"], "token": token}}

        # 修复：超时/停止时 full_text 为空，用占位文本避免 TTS 报错和气泡卡死
        if not full_text.strip():
            full_text = "（此环节已超时跳过）"

        yield {"event": "turn_end", "data": {
            "speaker_id": debater["id"],
            "full_text": full_text,
            "voice": debater["voice"],
            "skipped": full_text == "（此环节已超时跳过）",
        }}
        time.sleep(0.3)

    def run_debate(self):
        """
        执行完整辩论流程（标准华语辩论赛赛制），yield SSE 事件字典。

        标准赛制流程:
          阶段1: 开篇立论 — 正一 → 反一
          阶段2: 攻辩质询 — 正二质询反一 → 反二质询正一 → 正一质询反二 → 反一质询正二
          阶段3: 攻辩小结 — 正一 → 反一
          阶段4: 自由辩论 — 正反交替发言（N轮）
          阶段5: 总结陈词 — 反二 → 正二
          裁判点评 + 评分 + 宣布胜负
        """
        self._start_time = time.time()

        pro = [d for d in self._debaters if d["side"] == "pro"]
        con = [d for d in self._debaters if d["side"] == "con"]

        # ===== 阶段1: 开篇立论（仅一辩）=====
        yield {"event": "phase", "data": {"phase": "开篇立论", "description": "双方一辩阐述己方立场，构建论证框架"}}
        time.sleep(0.5)

        for debater in [pro[0], con[0]]:
            yield from self._emit_turn(debater, "opening")

        # ===== 阶段2: 攻辩质询（交叉配对）=====
        yield {"event": "phase", "data": {"phase": "攻辩质询", "description": "双方交叉质询，攻防对决"}}
        time.sleep(0.5)

        # 标准赛制：正二质询反一 → 反二质询正一 → 正一质询反二 → 反一质询正二
        cross_exam_pairs = [
            (pro[1], con[0]),   # 正二质询反一
            (con[1], pro[0]),   # 反二质询正一
            (pro[0], con[1]),   # 正一质询反二
            (con[0], pro[1]),   # 反一质询正二
        ]
        for questioner, answerer in cross_exam_pairs:
            yield from self._emit_turn(questioner, "question")
            yield from self._emit_turn(answerer, "answer")

        # ===== 阶段3: 攻辩小结 =====
        yield {"event": "phase", "data": {"phase": "攻辩小结", "description": "双方一辩总结质询阶段的交锋成果"}}
        time.sleep(0.5)

        for debater in [pro[0], con[0]]:
            yield from self._emit_turn(debater, "summary")

        # ===== 阶段4: 自由辩论 =====
        yield {"event": "phase", "data": {"phase": "自由辩论", "description": f"双方交替发言，共{self._free_debate_rounds}轮"}}
        time.sleep(0.5)

        for round_num in range(self._free_debate_rounds):
            yield from self._emit_turn(pro[round_num % len(pro)], "free")
            yield from self._emit_turn(con[round_num % len(con)], "free")

        # ===== 阶段5: 总结陈词（二辩）=====
        yield {"event": "phase", "data": {"phase": "总结陈词", "description": "双方二辩总结全场辩论"}}
        time.sleep(0.5)

        for debater in [con[-1], pro[-1]]:
            yield from self._emit_turn(debater, "closing")

        # ===== 裁判点评 =====
        yield {"event": "phase", "data": {"phase": "裁判点评", "description": "AI裁判进行评判"}}
        time.sleep(0.5)

        judge_result = self._judge_evaluate()
        self.history.append({
            "id": "judge",
            "name": "AI裁判",
            "side": "neutral",
            "role": "裁判",
            "content": judge_result,
            "phase": "judge",
            "voice": JUDGE["voice"],
        })
        yield {"event": "judge", "data": {"content": judge_result, "voice": JUDGE["voice"]}}

        self.finished = True
        yield {"event": "debate_end", "data": {"total_turns": len(self.history)}}
