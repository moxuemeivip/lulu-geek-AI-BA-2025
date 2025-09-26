import os
from pathlib import Path
from typing import Union
from mcp.server.fastmcp import FastMCP

# 创建 MCP Server
mcp = FastMCP("桌面 TXT 文件统计器")

@mcp.tool()
def count_desktop_txt_files() -> int:
    """统计桌面上 .txt 文件的数量"""
    # 获取桌面路径
    desktop_path = Path(os.path.expanduser("~/OneDrive - Microsoft/Desktop"))

    print(desktop_path)

    # 统计 .txt 文件
    txt_files = list(desktop_path.glob("*.txt"))
    return len(txt_files)

@mcp.tool()
def list_desktop_txt_files() -> str:
    """获取桌面上所有 .txt 文件的列表"""
    # 获取桌面路径
    desktop_path = Path(os.path.expanduser("~/OneDrive - Microsoft/Desktop"))
    print(desktop_path)

    # 获取所有 .txt 文件
    txt_files = list(desktop_path.glob("*.txt"))

    # 返回文件名列表
    if not txt_files:
        return f"{desktop_path}上没有找到 .txt 文件。"

    # 格式化文件名列表
    file_list = "\n".join([f"- {file.name}" for file in txt_files])
    return f"在桌面上找到 {len(txt_files)} 个 .txt 文件：\n{file_list}"

@mcp.tool()
def read_txt_file(file_name: str) -> str:
    """读取指定 .txt 文件的内容
    
    参数:
        file_name: txt 文件的文件名（包含扩展名）
    
    返回:
        文件的内容字符串
    """
    # 获取桌面路径
    desktop_path = Path(os.path.expanduser("~/OneDrive - Microsoft/Desktop"))
    
    # 构建完整的文件路径
    file_path = desktop_path / file_name
    
    # 检查文件是否存在
    if not file_path.exists():
        return f"错误: 文件 '{file_name}' 不存在于桌面目录中。"
    
    # 检查文件是否为 .txt 文件
    if file_path.suffix.lower() != '.txt':
        return f"错误: '{file_name}' 不是一个 .txt 文件。"
    
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return f"文件 '{file_name}' 的内容：\n{content}"
    except Exception as e:
        return f"读取文件时发生错误: {str(e)}"

if __name__ == "__main__":
    # 初始化并运行服务器
    mcp.run()