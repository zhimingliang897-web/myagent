import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

def get_checkpointer(db_path: str = "data/db/agent_memory.db"):
    """
    Creates and returns a SqliteSaver checkpointer.
    The database will be created if it doesn't exist and standard tables initialized.
    
    Args:
        db_path (str): Path to the SQLite database file.
        
    Returns:
        SqliteSaver: The configured checkpointer.
    """
    # Ensure the directory exists if a path is provided (handling relative paths too)
    directory = os.path.dirname(db_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    
    # check_same_thread=False is needed if the connection is shared across threads
    # though for this CLI it's mostly single threaded, it's safer.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    
    # SqliteSaver(conn) automatically creates the necessary tables if they don't exist
    memory = SqliteSaver(conn)
    
    return memory
