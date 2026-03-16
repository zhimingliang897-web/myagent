import re
import uuid
import asyncio
import threading
import gradio as gr
from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.tools import get_all_tools
from agent.rag.retriever import create_rag_tool
from agent.memory.checkpointer import get_checkpointer_context
from agent.graph import build_agent
from agent.multi import build_multi_agent_graph
from agent.multi import event_bus
from agent.callbacks import get_token_counter, reset_token_counter
from main import SYSTEM_PROMPT, MAX_MESSAGES
from agent.memory.profile import get_profile_summary, clear_profile

# ----------------- 会话名称规范化 ----------------- #

def _sanitize_thread_name(name: str, max_len: int = 32) -> str:
    """将会话名称规范为可作 thread_id 的字符串：去首尾空格、特殊字符改下划线、长度上限。"""
    if not name or not isinstance(name, str):
        return ""
    s = name.strip()
    s = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if s else ""


# ----------------- 全局状态初始化 ----------------- #
print("[启动] 初始化 LLM...")
llm = get_llm()
print("[启动] LLM 初始化完成")

# 全局变量：agent 和 checkpointer 上下文
global_agent = None
global_multi_agent = None  # 多智能体图
global_checkpointer = None
_rag_mode = "advanced"
_agent_mode = "single"  # "single" 或 "multi"
_init_lock = threading.Lock()


def init_agent_sync(rag_mode: str = "advanced", agent_mode: str = "single"):
    """同步初始化 Agent。

    Args:
        rag_mode: RAG 模式 ("classic" 或 "advanced")
        agent_mode: Agent 模式 ("single" 或 "multi")
    """
    global global_agent, global_multi_agent, global_checkpointer, _rag_mode, _agent_mode

    with _init_lock:
        _rag_mode = rag_mode
        _agent_mode = agent_mode

        print(f"[Agent] 正在初始化 (RAG: {rag_mode}, 模式: {agent_mode})...")

        # 在新的事件循环中初始化
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _init():
            global global_checkpointer
            if global_checkpointer is None:
                global_checkpointer = await get_checkpointer_context().__aenter__()

            tools = list(get_all_tools())
            rag_tool = create_rag_tool(mode=rag_mode)
            if rag_tool:
                tools.append(rag_tool)

            return build_agent(llm, tools, global_checkpointer, SYSTEM_PROMPT, MAX_MESSAGES)

        # 初始化单智能体
        global_agent = loop.run_until_complete(_init())

        # 初始化多智能体
        global_multi_agent = build_multi_agent_graph()

        print(f"[Agent] 初始化完成")

        if agent_mode == "multi":
            return "✅ 多智能体模式已启用 (Supervisor + Code/Data/Writer)"
        else:
            return f"✅ 单智能体模式 (RAG: {rag_mode})"


# 启动时初始化
print("[启动] 初始化 Agent...")
init_agent_sync("advanced", "single")
print("[启动] 准备就绪")


# ----------------- 核心交互逻辑 ----------------- #

async def bot_response(message, thread_id: str):
    """处理用户消息，返回完整响应（非流式）。

    message 可能是字符串或由分段组成的列表，这里统一规范为纯文本。
    """
    global global_agent, global_multi_agent, _agent_mode

    # 统一将 message 规范为字符串
    if isinstance(message, list):
        text_parts = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        message = "".join(text_parts)
    else:
        message = str(message) if message is not None else ""

    if not message or not message.strip():
        yield "请输入有效内容。"
        return

    # ================== 多智能体模式 ==================
    if _agent_mode == "multi":
        if global_multi_agent is None:
            yield "多智能体系统未初始化，请刷新页面重试。"
            return

        try:
            print(f"[Multi-Agent] 收到消息: {message[:30]}...")

            # 构建初始状态
            initial_state = {
                "messages": [],
                "iteration_count": 0,
                "task_plan": [],
                "current_worker": "supervisor",
                "worker_results": {},
                "handoff_context": "",
                "original_query": message,
            }

            # 使用 asyncio 线程运行同步的 invoke，从而允许我们在其运行时让前端刷新 UI
            loop = asyncio.get_running_loop()
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as pool:
                task = loop.run_in_executor(pool, global_multi_agent.invoke, initial_state)
                
                # 等待完成期间，不断 yield None 来刷新日志
                while not task.done():
                    yield None
                    await asyncio.sleep(0.2)
                
                result = task.result()

            print(f"[Multi-Agent] 响应完成")

            # 提取最终回答
            messages = result.get("messages", [])
            if messages:
                final_msg = messages[-1]
                content = final_msg.content if hasattr(final_msg, "content") else str(final_msg)
                yield content
            else:
                yield "（多智能体处理完成，但没有生成回答）"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"多智能体出错: {str(e)}"
        return

    # ================== 单智能体模式 ==================
    if global_agent is None:
        yield "Agent 未初始化，请刷新页面重试。"
        return

    config = {"configurable": {"thread_id": thread_id}}

    try:
        print(f"[Agent] 收到消息: {message[:30]}...")

        # 非流式调用
        result = await global_agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )

        print(f"[Agent] 响应完成")

        # 获取最后一条 AI 消息
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type in ("ai", "assistant"):
                if hasattr(msg, "content") and msg.content:
                    content = msg.content
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        yield "".join(text_parts)
                    return

        yield "（无响应）"

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"出错了: {str(e)}"


# ----------------- 自定义 CSS 样式（Tabler / 控制台风格）----------------- #

CUSTOM_CSS = """
/* ========== 全局与容器 ========== */
.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 0 1rem 2rem !important;
    font-family: var(--font), -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* 页面背景：Tabler 风格浅灰 */
.main .gradio-container {
    background: #f1f5f9 !important;
    min-height: 100vh;
}

/* ========== 顶部导航栏（类似 Tabler Navbar）========== */
.title-row {
    background: #fff !important;
    border-bottom: 1px solid #e2e8f0 !important;
    padding: 1rem 1.5rem !important;
    margin: 0 -1rem 1.5rem -1rem !important;
    border-radius: 0 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}

.title-row h1 {
    color: #1e293b !important;
    margin: 0 !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

.title-row p {
    color: #64748b !important;
    margin: 0.25rem 0 0 0 !important;
    font-size: 0.875rem !important;
}

/* ========== 主布局：左侧边栏 + 右侧内容 ========== */
.control-panel {
    background: transparent !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 1rem !important;
}

/* 左侧卡片区块（Tabler 卡片风格）*/
.control-section {
    background: #fff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

.control-section .gr-markdown h3 {
    font-size: 0.8125rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    margin: 0 0 0.75rem 0 !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

/* 右侧聊天主区域（大卡片）*/
.chat-area {
    background: #fff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 1.25rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

/* 欢迎/状态条（可选，放在聊天区上方）*/
.dashboard-welcome {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.875rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.875rem;
    color: #475569;
}

/* 输入行 */
.input-row {
    display: flex !important;
    gap: 0.75rem !important;
    margin-top: 1rem !important;
    align-items: flex-end !important;
}

.input-row .gr-textbox {
    border-radius: 8px !important;
    border: 1px solid #e2e8f0 !important;
}

.input-row .gr-button {
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* ========== 模式状态徽章（Tabler badge 风格）========== */
.mode-badge {
    padding: 0.625rem 0.875rem !important;
    border-radius: 6px !important;
    font-size: 0.8125rem !important;
    line-height: 1.4 !important;
    border: 1px solid transparent !important;
}

.mode-single {
    background: #eff6ff !important;
    border-color: #bfdbfe !important;
    color: #1e40af !important;
}

.mode-multi {
    background: #f5f3ff !important;
    border-color: #ddd6fe !important;
    color: #5b21b6 !important;
}

/* 按钮组 */
.button-row {
    display: flex !important;
    gap: 0.5rem !important;
}

.button-row .gr-button {
    border-radius: 6px !important;
    flex: 1 !important;
}

/* 对话管理 */
.thread-section .thread-hint {
    font-size: 0.75rem !important;
    color: #94a3b8 !important;
    margin: 0.25rem 0 0.5rem 0 !important;
}

.thread-section [data-testid="textbox"] {
    border-radius: 6px !important;
    border: 1px solid #e2e8f0 !important;
}

.thread-section .gr-button {
    border-radius: 6px !important;
}

/* Accordion 标题（长期记忆、Token、工具箱）*/
.control-panel .gr-accordion {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    background: #fff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}

.control-panel .gr-accordion .gr-form {
    border: none !important;
}

/* 执行过程 Terminal 风格 */
#process-log-box textarea {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 0.8125rem !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    padding: 0.75rem 1rem !important;
}

/* Chatbot 气泡与容器 */
.chat-area .gr-chatbot {
    border-radius: 8px !important;
}

/* 统一卡片内 Accordion */
.chat-area .gr-accordion {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    margin-bottom: 1rem !important;
    overflow: hidden !important;
    background: #fff !important;
}

/* 主按钮强调色（与主题一致）*/
.primary .gr-button {
    border-radius: 8px !important;
}
"""

# ----------------- 前端界面构建 ----------------- #

with gr.Blocks(title="MyAgent 智能体助手") as demo:

    # 状态变量
    session_id = gr.State(value=lambda: str(uuid.uuid4())[:8])

    # ================== 顶部导航栏（Tabler 风格）==================
    with gr.Row(elem_classes="title-row"):
        with gr.Column():
            gr.Markdown("# MyAgent")
            gr.Markdown("多智能体协作 · RAG 知识库 · 长期记忆")

    # ================== 主内容区 ==================
    with gr.Row():
        # 左侧控制面板
        with gr.Column(scale=1, min_width=280, elem_classes="control-panel"):

            # 智能体模式
            with gr.Group(elem_classes="control-section"):
                gr.Markdown("### 🤖 智能体模式")
                agent_mode_radio = gr.Radio(
                    choices=[
                        ("单智能体", "single"),
                        ("多智能体协作", "multi")
                    ],
                    value="single",
                    label="",
                    interactive=True
                )
                mode_status = gr.Markdown(
                    "**单智能体模式**\n\n通用问答，支持工具调用与 RAG 检索",
                    elem_classes="mode-badge mode-single"
                )
                apply_mode_btn = gr.Button("应用模式", variant="primary", size="sm")

            # RAG 设置
            with gr.Group(elem_classes="control-section"):
                gr.Markdown("### 📚 知识库")
                rag_mode_radio = gr.Radio(
                    choices=[
                        ("FAISS 向量检索", "classic"),
                        ("混合检索 (推荐)", "advanced")
                    ],
                    value="advanced",
                    label="",
                    interactive=True
                )
                apply_rag_btn = gr.Button("应用设置", size="sm")

            # 对话管理（可编辑会话名称）
            with gr.Group(elem_classes="control-section thread-section"):
                gr.Markdown("### 💬 对话管理")
                with gr.Row():
                    thread_name_input = gr.Textbox(
                        label="会话名称",
                        value="",
                        placeholder="输入名称便于区分，留空则使用随机 ID",
                        interactive=True,
                        max_lines=1,
                        scale=3,
                        show_label=True,
                        container=False,
                    )
                    save_name_btn = gr.Button("保存名称", variant="secondary", size="sm", scale=1, min_width=72)
                gr.Markdown("同一名称会延续历史对话", elem_classes="thread-hint")
                new_thread_btn = gr.Button("开启新对话", variant="secondary", size="sm")

            # 记忆管理
            with gr.Accordion("🧠 长期记忆", open=False):
                current_memory_box = gr.Textbox(
                    label="",
                    value=get_profile_summary(),
                    interactive=False,
                    lines=4,
                    max_lines=6,
                    placeholder="暂无用户画像记录..."
                )
                with gr.Row(elem_classes="button-row"):
                    refresh_btn = gr.Button("刷新", size="sm")
                    clear_mem_btn = gr.Button("清除", variant="stop", size="sm")

            # Token 统计面板
            with gr.Accordion("📊 Token 统计", open=True):
                token_stats_box = gr.Textbox(
                    label="",
                    value="暂无统计数据",
                    interactive=False,
                    lines=4,
                    max_lines=5,
                )

            # 工具箱面板
            with gr.Accordion("🧰 工具箱", open=False):
                tool_names = [t.name for t in get_all_tools()]
                gr.CheckboxGroup(
                    choices=tool_names,
                    value=tool_names,
                    label="已启用工具",
                    interactive=False,
                    info="工具呈现中，可插拔功能开发中..."
                )

        # 右侧聊天区
        with gr.Column(scale=3, elem_classes="chat-area"):
            # 欢迎/状态条（仪表盘风格）
            with gr.Row(elem_classes="dashboard-welcome"):
                gr.Markdown("**对话** — 在此输入问题，Agent 将调用工具与知识库进行回答。")
            # 执行过程展示（多智能体模式下有内容）
            with gr.Accordion("🔍 执行过程 (Terminal)", open=False):
                process_log_box = gr.Textbox(
                    label="",
                    value="等待任务执行...",
                    interactive=False,
                    lines=8,
                    max_lines=15,
                    autoscroll=True,
                    elem_id="process-log-box"
                )

            chatbot = gr.Chatbot(
                height=460,
                show_label=False,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=agent"),
            )
            with gr.Row(elem_classes="input-row"):
                msg_input = gr.Textbox(
                    placeholder="输入问题，按 Enter 发送... (支持 Shift+Enter 换行)",
                    show_label=False,
                    scale=6,
                    lines=1,
                    max_lines=5,
                    autofocus=True,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1, min_width=80)

    # ----------------- 事件绑定 ----------------- #

    # 页面加载时用当前 session_id 回填会话名称输入框
    demo.load(
        lambda sid: sid or "",
        inputs=[session_id],
        outputs=[thread_name_input]
    )

    def save_thread_name(name):
        """将输入框内容规范化为 thread_id 并写回 state，返回 (session_id, 显示值)。"""
        sanitized = _sanitize_thread_name(name)
        if not sanitized:
            new_id = str(uuid.uuid4())[:8]
            return new_id, ""
        return sanitized, sanitized

    def add_user_message(history, text):
        """将用户消息加入对话历史，并预留一条空的 assistant 消息用于后续填充；发送时禁用输入与按钮。"""
        if not text or not text.strip():
            return history, gr.update(), gr.update()
        history = history or []
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": ""})
        return (
            history,
            gr.update(value="", interactive=False),
            gr.update(interactive=False),
        )

    async def generate_response(history, thread_id):
        """根据最新一条用户消息生成助手回复。"""
        if not history or len(history) < 2:
            yield history, gr.update(), gr.update()
            return

        last = history[-1]
        prev = history[-2]

        # 兼容多种 history 结构
        if isinstance(last, dict) and last.get("role") == "assistant":
            raw_content = prev.get("content", "")
        elif isinstance(history[-1], (list, tuple)) and len(history[-1]) == 2:
            raw_content = history[-1][0]
        else:
            yield history, gr.update(), gr.update()
            return

        if isinstance(raw_content, list):
            text_parts = []
            for item in raw_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            user_message = "".join(text_parts)
        else:
            user_message = str(raw_content) if raw_content is not None else ""

        # 重置 Token 计数和事件队列
        reset_token_counter()
        event_bus.clear()
        accumulated_log = ""

        async for chunk in bot_response(user_message, thread_id):
            if chunk is not None:
                if isinstance(last, dict):
                    last["content"] = chunk
                elif isinstance(last, (list, tuple)) and len(last) == 2:
                    last[1] = chunk
                history[-1] = last

            # 刷新执行过程日志
            new_events = event_bus.drain()
            if new_events:
                accumulated_log = accumulated_log + ("\n" if accumulated_log else "") + new_events

            yield history, accumulated_log or "运行后展示 Agent 执行步骤...", gr.update()

        # 对话结束后刷新 Token 统计
        final_events = event_bus.drain()
        if final_events:
            accumulated_log = accumulated_log + ("\n" if accumulated_log else "") + final_events
        token_summary = get_token_counter().summary()
        yield history, accumulated_log or "执行完毕", token_summary

    def reenable_input_and_send():
        """回复结束后恢复输入框与发送按钮。"""
        return gr.update(interactive=True), gr.update(interactive=True)

    # 发送消息（发送时禁用输入框与按钮，回复结束后恢复）
    msg_input.submit(
        add_user_message, [chatbot, msg_input], [chatbot, msg_input, send_btn], queue=False
    ).then(
        generate_response, [chatbot, session_id], [chatbot, process_log_box, token_stats_box]
    ).then(
        reenable_input_and_send, None, [msg_input, send_btn], queue=False
    )

    send_btn.click(
        add_user_message, [chatbot, msg_input], [chatbot, msg_input, send_btn], queue=False
    ).then(
        generate_response, [chatbot, session_id], [chatbot, process_log_box, token_stats_box]
    ).then(
        reenable_input_and_send, None, [msg_input, send_btn], queue=False
    )

    # 切换智能体模式
    def switch_agent_mode(agent_mode, rag_mode):
        global _agent_mode
        _agent_mode = agent_mode
        init_agent_sync(rag_mode, agent_mode)
        if agent_mode == "multi":
            return gr.Markdown(
                "**多智能体协作模式**\n\n"
                "• Supervisor 任务分解\n"
                "• Code Agent 代码生成\n"
                "• Data Agent 数据分析\n"
                "• Writer Agent 文章撰写",
                elem_classes="mode-badge mode-multi"
            )
        else:
            return gr.Markdown(
                "**单智能体模式**\n\n通用问答，支持工具调用与 RAG 检索",
                elem_classes="mode-badge mode-single"
            )

    apply_mode_btn.click(
        switch_agent_mode,
        [agent_mode_radio, rag_mode_radio],
        [mode_status]
    )

    # 切换 RAG 模式
    apply_rag_btn.click(
        lambda rag, agent: init_agent_sync(rag, agent),
        [rag_mode_radio, agent_mode_radio],
        None
    )

    # 刷新记忆
    refresh_btn.click(lambda: get_profile_summary(), None, [current_memory_box])

    # 清除记忆
    def clear_and_refresh():
        clear_profile()
        return get_profile_summary()
    clear_mem_btn.click(clear_and_refresh, None, [current_memory_box])

    # 保存会话名称：规范化后写回 session_id，并更新输入框显示
    save_name_btn.click(
        save_thread_name,
        inputs=[thread_name_input],
        outputs=[session_id, thread_name_input]
    ).then(
        lambda: gr.Info("已保存，后续消息将使用此会话"),
        None,
        None
    )

    # 新话题（重置 session_id、清空名称输入框与对话、过程日志）
    def reset_thread():
        new_id = str(uuid.uuid4())[:8]
        reset_token_counter()
        event_bus.clear()
        return new_id, "", [], "暂无统计数据", "运行后展示 Agent 执行步骤..."
    new_thread_btn.click(
        reset_thread,
        None,
        [session_id, thread_name_input, chatbot, token_stats_box, process_log_box]
    )


if __name__ == "__main__":
    import signal
    import sys

    def _shutdown(sig, frame):
        print("\n[关闭] 正在停止服务，释放端口 7860...", flush=True)
        try:
            demo.close()
        except Exception:
            pass
        print("[关闭] 服务已停止。", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            css=CUSTOM_CSS,
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate",
                neutral_hue="slate",
                font=gr.themes.GoogleFont("Inter"),
                radius_size="sm",
            ),
        )
    except KeyboardInterrupt:
        print("\n[关闭] 检测到 Ctrl+C，正在停止服务...", flush=True)
        demo.close()
        print("[关闭] 端口已释放，服务停止。", flush=True)
