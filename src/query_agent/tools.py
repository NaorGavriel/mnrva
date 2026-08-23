import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from query_agent.schemas import FileReadRequest, FileReadResult, GrepMatch
