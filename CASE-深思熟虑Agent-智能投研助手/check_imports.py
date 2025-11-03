#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查导入问题的脚本
"""

print("开始检查导入...")

# 逐个尝试导入，以便找出可能的问题

try:
    from langchain_core.prompts import ChatPromptTemplate
    print("✓ langchain_core.prompts.ChatPromptTemplate 导入成功")
except ImportError as e:
    print(f"✗ langchain_core.prompts.ChatPromptTemplate 导入失败: {e}")

try:
    from langchain_community.llms import Tongyi
    print("✓ langchain_community.llms.Tongyi 导入成功")
except ImportError as e:
    print(f"✗ langchain_community.llms.Tongyi 导入失败: {e}")

try:
    from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
    print("✓ langchain_core.output_parsers 导入成功")
except ImportError as e:
    print(f"✗ langchain_core.output_parsers 导入失败: {e}")

try:
    from pydantic.v1 import BaseModel, Field, validator
    print("✓ pydantic.v1 导入成功")
except ImportError as e:
    print(f"✗ pydantic.v1 导入失败: {e}")

try:
    from langgraph.graph import StateGraph, END
    print("✓ langgraph.graph 导入成功")
except ImportError as e:
    print(f"✗ langgraph.graph 导入失败: {e}")

try:
    import dashscope
    print("✓ dashscope 导入成功")
except ImportError as e:
    print(f"✗ dashscope 导入失败: {e}")

print("\n导入检查完成。")