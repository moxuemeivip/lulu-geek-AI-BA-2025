from PyPDF2 import PdfReader
from langchain_community.callbacks import get_openai_callback
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# 使用阿里云的embedding和chat模型
from typing import List, Tuple, Dict, Any
from collections import Counter
import os

# 使用标准的 OpenAI 嵌入模型（暂时替代阿里云）
from langchain_openai import OpenAIEmbeddings

DASHSCOPE_API_KEY = "You_DashScope_API_Key"

def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:
    """
    从PDF中提取文本并记录每行文本对应的页码
    
    参数:
        pdf: PDF文件对象
    
    返回:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
    """
    text = ""
    page_numbers = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))
        else:
            print(f"警告: 第 {page_number} 页未找到文本。")

    return text, page_numbers


def process_text_with_splitter(text: str, page_numbers: List[int]) -> FAISS:
    """
    处理文本并创建向量存储
    
    参数:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
    
    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")
        
    # 使用阿里云的Qwen embedding模型
    try:
        # 使用 DashScopeEmbeddings 类创建嵌入模型
        from langchain_community.embeddings import DashScopeEmbeddings
        
        # 创建嵌入模型，使用 text-embedding-v4
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
        print("使用阿里云Qwen Embedding模型")
        
        # 测试embedding模型是否正常工作
        test_embedding = embeddings.embed_query("测试文本")
        print(f"Embedding维度: {len(test_embedding)}")
        
    except Exception as e:
        print(f"阿里云Qwen Embedding模型初始化失败: {e}")
        print("错误详情:", str(e))
        print("使用简单嵌入模型作为备选")
        
        # 使用简单嵌入作为备选
        from langchain_core.embeddings import Embeddings
        import hashlib
        
        class SimpleEmbeddings(Embeddings):
            def __init__(self, dimensions=1024):
                self.dimensions = dimensions
            
            def _get_text_vector(self, text):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                vector = []
                for i in range(0, len(text_hash), 2):
                    value = int(text_hash[i:i+2], 16) / 255.0 * 2 - 1
                    vector.append(value)
                    while len(vector) < self.dimensions:
                        vector.extend(vector)
                return vector[:self.dimensions]
            
            def embed_documents(self, texts):
                return [self._get_text_vector(text) for text in texts]
            
            def embed_query(self, text):
                return self._get_text_vector(text)
        
        embeddings = SimpleEmbeddings(dimensions=1024)
    
    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")
    
    # 创建一个与文本等长的页码列表，每个字符对应一个页码
    char_pages = []
    for page_num, page_text in zip(page_numbers, text.split('\n')):
        char_pages.extend([page_num] * len(page_text))
        char_pages.append(page_num)  # 为换行符添加页码

    # 为每个chunk找到最常见的页码
    chunk_page_info = {}
    for chunk in chunks:
        chunk_start = text.find(chunk)
        if chunk_start != -1 and chunk_start + len(chunk) <= len(char_pages):
            # 获取这个chunk中所有字符的页码
            chunk_pages = char_pages[chunk_start:chunk_start + len(chunk)]
            # 找出最常见的页码
            most_common_page = Counter(chunk_pages).most_common(1)[0][0]
            chunk_page_info[chunk] = most_common_page
        else:
            chunk_page_info[chunk] = "未知"

    knowledgeBase.page_info = chunk_page_info

    return knowledgeBase

# 读取PDF文件
pdf_reader = PdfReader('./浦发上海浦东发展银行西安分行个金客户经理考核办法.pdf')
# 提取文本和页码信息
text, page_numbers = extract_text_with_page_numbers(pdf_reader)
print(f"提取的文本长度: {len(text)} 个字符。")
    
# 处理文本并创建知识库
knowledgeBase = process_text_with_splitter(text, page_numbers)

# 设置查询问题
query = "客户经理被投诉了，投诉一次扣多少分"
#query = "客户经理每年评聘申报时间是怎样的？"
if query:
    # 执行相似度搜索，找到与查询相关的文档
    docs = knowledgeBase.similarity_search(query)

    # 使用阿里云的Chat模型
    try:
        from langchain_openai import ChatOpenAI
        # 创建阿里云的Chat模型（使用DashScope API调用Qwen）
        llm = ChatOpenAI(
            model="qwen-turbo",  # 使用Qwen Turbo模型
            openai_api_key=DASHSCOPE_API_KEY,
            openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.7,
            max_tokens=1024
        )
        print("使用阿里云Qwen Turbo模型")
    except Exception as e:
        print(f"阿里云Chat模型初始化失败: {e}")
        # 使用简单的模拟响应作为备选
        class SimpleLLM:
            def invoke(self, input):
                """模拟简单的问答系统"""
                if isinstance(input, dict) and 'question' in input:
                    question = input['question']
                else:
                    question = str(input)
                
                answer = f"根据文档内容，关于'{question}'的信息如下：\n\n"
                
                if isinstance(input, dict) and 'context' in input:
                    context = input['context']
                    if hasattr(context, 'page_content'):
                        content = context.page_content
                        answer += f"文档片段内容：{content[:200]}...\n"
                    elif isinstance(context, list):
                        for i, doc in enumerate(context[:2]):
                            if hasattr(doc, 'page_content'):
                                answer += f"文档{i+1}内容：{doc.page_content[:150]}...\n"
                
                return answer
        
        llm = SimpleLLM()
        print("使用简单模拟LLM作为备选")
    
    # 创建问答提示模板
    prompt = ChatPromptTemplate.from_template("""
                基于以下上下文信息回答问题：

                上下文信息：
                {context}

                问题：{question}

                请基于以上信息回答问题，如果上下文中没有相关信息，请说明。
            """)
    
    # 创建检索链
    def format_docs(docs):
        return "\n\n".join(doc if isinstance(doc, str) else doc.page_content for doc in docs)
    
    # 使用简单的链式调用
    rag_chain = (
        {
            "context": lambda x: format_docs(knowledgeBase.similarity_search(x)), 
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 使用回调函数跟踪API调用成本
    with get_openai_callback() as cost:
        # 执行问答链
        response = rag_chain.invoke(query)
        print(f"查询: {query}")
        print(f"查询已处理。成本: {cost}")
        print(f"答案: {response}")
        print("来源:")

    # 记录唯一的页码
    unique_pages = set()

    # 显示每个文档块的来源页码
    for doc in docs:
        text_content = getattr(doc, "page_content", "")
        source_page = knowledgeBase.page_info.get(
            text_content.strip(), "未知"
        )

        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"来源页码: {source_page}")
            # print(f"文本块内容: {text_content}")
            # print("-" * 50)  # 添加分隔线，使输出更清晰

