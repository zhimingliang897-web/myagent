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


@tool
def remember_user_fact(key: str, value: str) -> str:
    """记住关于用户的长期事实或偏好信息以供未来跨对话使用。
    当你了解到关于用户的新事实时（例如：用户的名字，职业，爱好，习惯等），务必调用此工具。
    参数说明：
    - key: 事实名称，尽量精简，如 "名字", "职业", "编程语言偏好"
    - value: 事实内容，如 "阿亮", "全栈工程师", "Python"
    """
    from agent.memory.profile import update_user_fact
    return update_user_fact(key, value)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气信息。
    当用户询问天气、气温、是否下雨等天气相关问题时使用此工具。
    参数说明：
    - city: 城市名（支持中文或英文，如 "北京"、"Shanghai"、"New York"）
    """
    try:
        # wttr.in 免费天气 API，无需 Key
        url = f"https://wttr.in/{city}?format=j1&lang=zh"
        response = httpx.get(url, timeout=10.0, headers={"User-Agent": "curl/7.0"})
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]
        area = data["nearest_area"][0]

        area_name = area["areaName"][0]["value"]
        country = area["country"][0]["value"]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]
        desc = current["lang_zh"][0]["value"] if current.get("lang_zh") else current["weatherDesc"][0]["value"]
        wind_speed = current["windspeedKmph"]
        wind_dir = current["winddir16Point"]
        visibility = current["visibility"]

        return (
            f"📍 {area_name}, {country}\n"
            f"🌡️ 当前气温: {temp_c}°C（体感 {feels_like}°C）\n"
            f"🌤️ 天气状况: {desc}\n"
            f"💧 湿度: {humidity}%\n"
            f"💨 风速: {wind_speed} km/h，风向: {wind_dir}\n"
            f"👁️ 能见度: {visibility} km"
        )
    except httpx.HTTPError as e:
        return f"天气查询网络错误: {e}"
    except (KeyError, IndexError, ValueError) as e:
        return f"天气数据解析失败（城市名可能不正确）: {e}"
    except Exception as e:
        return f"天气查询失败: {e}"


ALL_TOOLS = [get_current_datetime, calculate, web_search, remember_user_fact, get_weather]

