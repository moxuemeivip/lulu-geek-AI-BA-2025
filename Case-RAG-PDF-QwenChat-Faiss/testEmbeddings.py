from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

DASHSCOPE_API_KEY = 'sk-3a10b27a251a46fe8aed54b6024d1e88'
# 创建嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

# 假设chunks是你的文本块列表
chunks = ["这是一个示例文本块。", "这是另一个示例文本块。"]

# 从文本块创建知识库
knowledgeBase = FAISS.from_texts(chunks, embeddings)