import time
import threading
import copy
import os
from config import JUDGE, MAX_HISTORY, MAX_WORDS, FREE_DEBATE_ROUNDS, DEBATE_TIME_LIMIT, LLM_PROVIDERS, DEBATERS
from app.tools.debate.llm import LLMClient

SIDE_LABELS = {"pro": "正方", "con": "反方"}

class DebateEngine:
    def __init__(self, topic: str, pro_position: str, con_position: str,
                 debaters=None, providers=None, params=None, output_dir=None):
        self.topic = topic
        self.pro_position = pro_position
        self.con_position = con_position
        self.history = []
        self.clients = {}
        self.debater_map = {}
        self.finished = False
        self._start_time = None
        self.step_event = threading.Event()
        self.step_event.set()
        self.output_dir = output_dir or os.path.join(os.path.dirname(__file__), 'output')
        
        self._debaters = debaters if debaters is not None else DEBATERS
        self._providers = providers if providers is not None else LLM_PROVIDERS
        
        params = params or {}
        self._max_words = int(params.get("max_words", MAX_WORDS))
        self._free_debate_rounds = int(params.get("free_debate_rounds", FREE_DEBATE_ROUNDS))
        self._debate_time_limit = float(params.get("debate_time_limit", DEBATE_TIME_LIMIT))
    
    def _is_timeout(self) -> bool:
        if self._start_time is None or self._debate_time_limit <= 0:
            return False
        return (time.time() - self._start_time) >= self._debate_time_limit
    
    def stop(self):
        self.finished = True
        self.step_event.set()
    
    def init_clients(self):
        for d in self._debaters:
            provider_key = d["provider"]
            provider = self._providers.get(provider_key)
            if not provider:
                raise ValueError(f"Provider not found: {provider_key}")
            model = d.get("model_override") or provider["model"]
            self.clients[d["id"]] = LLMClient(
                base_url=provider["base_url"],
                api_key=provider["api_key"],
                model=model,
            )
            self.debater_map[d["id"]] = d
        
        judge_provider_key = JUDGE["provider"]
        judge_provider = self._providers.get(judge_provider_key)
        if not judge_provider:
            judge_provider = next(iter(self._providers.values()))
        self.clients["judge"] = LLMClient(
            base_url=judge_provider["base_url"],
            api_key=judge_provider["api_key"],
            model=judge_provider["model"],
        )
    
    def _build_system_prompt(self, debater: dict, phase: str) -> str:
        side_label = SIDE_LABELS[debater["side"]]
        position = self.pro_position if debater["side"] == "pro" else self.con_position
        
        phase_rules = {
            "opening": "【开篇陈词】明确定义关键概念，提出2-3个核心论点，用有力的事例开场。",
            "question": "【攻辩质询】用'请问对方辩友'开头，提出尖锐的封闭式问题。",
            "answer": "【攻辩质询】先正面回应问题，然后反击。",
            "free": "【自由辩论】紧盯对方发言逐点反驳，语言快、短、有攻击性。",
            "summary": "【攻辩小结】总结质询阶段的交锋成果。",
            "closing": "【总结陈词】梳理全场交锋，重新强调己方论点，上升到价值层面。",
        }
        
        return f"""你是"{debater['name']}"，正在参加辩论赛。你是{side_label}{debater['role']}。

【辩题】{self.topic}
【你的立场】{side_label}：{position}
【你的辩论风格】{debater['personality']}

{phase_rules.get(phase, '')}

【规则】
- 每次发言严格控制在{self._max_words}字以内
- 称呼对方为"对方辩友"
- 直接输出发言内容，不要加任何前缀。"""
    
    def _build_messages(self, debater: dict, phase: str) -> list:
        messages = [{"role": "system", "content": self._build_system_prompt(debater, phase)}]
        recent = self.history[-MAX_HISTORY:]
        for entry in recent:
            side_label = SIDE_LABELS.get(entry["side"], entry["side"])
            if entry["id"] == debater["id"]:
                messages.append({"role": "assistant", "content": entry["content"]})
            else:
                prefix = f"[{entry['name']}（{side_label}{entry['role']}）]"
                messages.append({"role": "user", "content": f"{prefix} {entry['content']}"})
        
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
        self.step_event.wait()
        self.step_event.clear()
        
        if self.finished or self._is_timeout():
            return
        
        debater = self.debater_map[debater_id]
        messages = self._build_messages(debater, phase)
        
        full_text = ""
        try:
            for token in self.clients[debater_id].chat_stream(messages):
                full_text += token
                yield token
                if len(full_text) > self._max_words * 2:
                    break
        except Exception as e:
            if not full_text:
                full_text = f"（{debater['name']}发言出现技术问题：{str(e)[:50]}）"
                yield full_text
        
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
（宣布胜方及理由）"""
        
        messages = [
            {"role": "system", "content": "你是一位公正权威的辩论赛裁判。"},
            {"role": "user", "content": prompt},
        ]
        return self.clients["judge"].chat(messages, temperature=0.3)
    
    def _emit_turn(self, debater, phase):
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
        self._start_time = time.time()
        
        pro = [d for d in self._debaters if d["side"] == "pro"]
        con = [d for d in self._debaters if d["side"] == "con"]
        
        yield {"event": "phase", "data": {"phase": "开篇立论", "description": "双方一辩阐述己方立场"}}
        time.sleep(0.5)
        for debater in [pro[0], con[0]]:
            yield from self._emit_turn(debater, "opening")
        
        yield {"event": "phase", "data": {"phase": "攻辩质询", "description": "双方交叉质询"}}
        time.sleep(0.5)
        cross_exam_pairs = [
            (pro[1], con[0]), (con[1], pro[0]),
            (pro[0], con[1]), (con[0], pro[1]),
        ]
        for questioner, answerer in cross_exam_pairs:
            yield from self._emit_turn(questioner, "question")
            yield from self._emit_turn(answerer, "answer")
        
        yield {"event": "phase", "data": {"phase": "攻辩小结", "description": "双方一辩总结"}}
        time.sleep(0.5)
        for debater in [pro[0], con[0]]:
            yield from self._emit_turn(debater, "summary")
        
        yield {"event": "phase", "data": {"phase": "自由辩论", "description": f"双方交替发言，共{self._free_debate_rounds}轮"}}
        time.sleep(0.5)
        for round_num in range(self._free_debate_rounds):
            yield from self._emit_turn(pro[round_num % len(pro)], "free")
            yield from self._emit_turn(con[round_num % len(con)], "free")
        
        yield {"event": "phase", "data": {"phase": "总结陈词", "description": "双方二辩总结全场"}}
        time.sleep(0.5)
        for debater in [con[-1], pro[-1]]:
            yield from self._emit_turn(debater, "closing")
        
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