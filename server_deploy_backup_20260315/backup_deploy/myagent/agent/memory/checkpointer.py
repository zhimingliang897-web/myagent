import os
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# 默认数据库路径
DEFAULT_DB_PATH = "data/db/agent_memory.db"


def get_checkpointer_context(db_path: str = DEFAULT_DB_PATH):
    """
    Returns an async context manager for AsyncSqliteSaver.

    Usage:
        async with get_checkpointer_context() as checkpointer:
            agent = build_agent(..., checkpointer=checkpointer)
            # use agent

    Args:
        db_path (str): Path to the SQLite database file.

    Returns:
        AsyncContextManager[AsyncSqliteSaver]: Async context manager for the checkpointer.
    """
    # Ensure the directory exists
    directory = os.path.dirname(db_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    return AsyncSqliteSaver.from_conn_string(db_path)
