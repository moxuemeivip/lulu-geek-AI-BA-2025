import os
from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate

# 设置API密钥
DASHSCOPE_API_KEY = "sk-3a10b27a251a46fe8aed54b6024d1e88"
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

# 创建Tongyi实例并指定模型
llm = Tongyi(model_name="qwen-turbo", temperature=0.1)

# 定义提示模板
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

# 创建链式调用
chain = prompt | llm

# 提问
question = "9.9 和 9.11 哪个更大？"
response = chain.invoke({"question": question})

print(response)