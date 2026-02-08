import ast
import operator
import re
from datetime import datetime

import httpx
from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """获取当前日期和时间，包括星期几。
    当用户询问今天的日期、当前时间或星期几时使用此工具。"""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays[now.weekday()]
    return now.strftime(f"%Y-%m-%d %H:%M:%S ({weekday})")


@tool
def calculate(expression: str) -> str:
    """计算数学表达式并返回结果。
    用于任何算术计算，支持 +, -, *, /, **（幂运算）和括号。
    示例输入: '2 + 3', '(10 + 5) * 2', '2 ** 10'"""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量: {node.value}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return allowed_operators[op_type](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return allowed_operators[op_type](_eval(node.operand))
        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        return str(result)
    except Exception as e:
        return f"计算 '{expression}' 时出错: {e}"


@tool
def web_search(query: str) -> str:
    """使用 DuckDuckGo 搜索网络信息。
    当用户询问时事新闻、你不确定的事实或任何需要最新互联网信息的问题时使用此工具。
    返回搜索结果摘要。"""
    try:
        url = "https://html.duckduckgo.com/html/"
        response = httpx.post(
            url,
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10.0,
        )
        response.raise_for_status()

        results = []
        snippets = re.findall(
            r'class="result__snippet">(.*?)</a>', response.text, re.DOTALL
        )
        for i, snippet in enumerate(snippets[:5]):
            clean = re.sub(r"<.*?>", "", snippet).strip()
            if clean:
                results.append(f"{i + 1}. {clean}")

        if results:
            return "\n".join(results)
        else:
            return f"未找到关于 '{query}' 的搜索结果"
    except Exception as e:
        return f"搜索出错: {e}"


ALL_TOOLS = [get_current_datetime, calculate, web_search]
