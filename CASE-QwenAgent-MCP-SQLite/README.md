# 数据库助手 (SQLite) 项目说明

## 项目概述

这是一个基于 `qwen-agent` 实现的数据库助手应用，通过 Model Context Protocol (MCP) 与 SQLite 数据库进行交互。该应用提供了命令行测试、终端界面(TUI)和图形界面(GUI)三种使用模式，支持用户通过自然语言查询和操作 SQLite 数据库。

## 功能特点

- 🔍 **智能数据库查询**：通过自然语言提问查询数据库内容
- 📊 **多界面支持**：提供测试模式、终端界面和图形界面
- 🛠️ **MCP 协议集成**：使用 MCP 服务器与 SQLite 数据库交互
- 📁 **文件支持**：可以结合文件内容进行查询和分析

## 系统要求

- Python 3.8 或更高版本
- 需要安装以下依赖包

## 安装依赖

### 基本依赖安装

```bash
# 使用清华镜像源加速安装
pip install qwen-agent qwen-api -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### MCP 相关依赖安装

需要安装 MCP 服务器和 SQLite 扩展：

```bash
# 安装 MCP 服务器及其 SQLite 扩展
pip install mcp mcp-server-sqlite -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 UV 包管理器（用于运行 MCP 服务器）
pip install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 可选优化：设置超时时间

如果遇到网络连接超时问题，可以设置更长的超时时间：

```bash
# Windows 系统
set UV_HTTP_TIMEOUT=120

# Linux/Mac 系统
export UV_HTTP_TIMEOUT=120
```

## 代码结构说明

### 主要模块

1. **初始化代理服务** (`init_agent_service()`)
   - 配置 LLM 模型参数
   - 设置系统提示
   - 配置 MCP 服务器连接

2. **测试功能** (`test()`)
   - 快速测试数据库助手功能
   - 支持纯文本查询和文件上传

3. **终端用户界面** (`app_tui()`)
   - 交互式命令行界面
   - 支持连续对话

4. **图形用户界面** (`app_gui()`)
   - 基于 Web 的图形界面
   - 提供预设提示建议

### 核心配置

```python
# LLM 配置
llm_cfg = {
    'model': 'qwen3-max',
    'model_server': 'dashscope',
    'api_key': 'sk-3a10b27a251a46fe8aed54b6024d1e88',
    'generate_cfg': {
        'top_p': 0.8
    }
}

# MCP 服务器配置
tools = [{
    "mcpServers": {
        "sqlite" : {
            "command": "uvx",
            "args": [
                "mcp-server-sqlite",
                "--db-path",
                "test.db"
            ]
        }
    }
}]
```

## 使用方法

### 1. 准备数据库

确保当前目录下有一个有效的 SQLite 数据库文件 `test.db`。如果文件不存在，MCP 服务器会尝试创建它。

### 2. 运行应用

根据您的需要选择以下三种运行模式之一：

#### 测试模式

适用于快速验证功能：

```python
# 修改主函数部分，取消注释 test()
if __name__ == '__main__':
    test()
    # app_tui()
    # app_gui()
```

然后运行：
```bash
python assistant_mcp_sqlite_bot.py
```

#### 终端界面模式

适用于交互式命令行操作：

```python
# 修改主函数部分，取消注释 app_tui()
if __name__ == '__main__':
    # test()
    app_tui()
    # app_gui()
```

然后运行：
```bash
python assistant_mcp_sqlite_bot.py
```

#### 图形界面模式

适用于可视化操作：

```python
# 修改主函数部分，取消注释 app_gui()
if __name__ == '__main__':
    # test()
    # app_tui()
    app_gui()
```

然后运行：
```bash
python assistant_mcp_sqlite_bot.py
```

### 3. 输入查询示例

在运行过程中，可以尝试以下查询：

- 数据库里有几张表
- 表的结构是什么
- 查询所有学生信息
- 创建一个学生表包括学生的姓名、年龄
- 增加一个学生名字叫韩梅梅，今年6岁

## 常见问题与解决方案

### 1. MCP 服务器连接失败

如果遇到 `FileNotFoundError: [WinError 2]` 错误，确保：
- 已正确安装 uv 包管理器
- 已正确安装 mcp-server-sqlite
- 检查环境变量是否正确设置

### 2. 数据库操作超时

如果操作超时，尝试：
- 增加 UV_HTTP_TIMEOUT 环境变量的值
- 使用国内镜像源重新安装依赖

### 3. 其他问题

如果遇到其他问题，请参考：
- [Qwen-Agent 官方文档](https://github.com/QwenLM/Qwen-Agent)
- [MCP 协议文档](https://github.com/modelcontext/modelcontext)
