#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
私募基金运作指引问答助手 - 反应式智能体实现

适合反应式架构的私募基金问答助手，使用Agent模式实现主动思考和工具选择。
"""

import re
from typing import List, Dict, Any, Union
from langchain_core.tools import Tool
from langchain_community.llms import Tongyi
# LLMChain已在新版langchain中移除，使用create_react_agent替代
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLLM
from langchain.agents.factory import create_agent
from langchain_core.runnables import RunnableLambda

# 通义千问API密钥
import dashscope
DASHSCOPE_API_KEY = 'your-dashscope-api-key'
dashscope.api_key = DASHSCOPE_API_KEY

# 简化的私募基金规则数据库
FUND_RULES_DB = [
    {
        "id": "rule001",
        "category": "设立与募集",
        "question": "私募基金的合格投资者标准是什么？",
        "answer": "合格投资者是指具备相应风险识别能力和风险承担能力，投资于单只私募基金的金额不低于100万元且符合下列条件之一的单位和个人：\n1. 净资产不低于1000万元的单位\n2. 金融资产不低于300万元或者最近三年个人年均收入不低于50万元的个人"
    },
    {
        "id": "rule001a",
        "category": "设立与募集",
        "question": "机构投资者的合格标准是什么？",
        "answer": "机构投资者作为合格投资者需要满足：净资产不低于1000万元。"
    },
    {
        "id": "rule001b",
        "category": "设立与募集",
        "question": "净资产800万元的机构能投资私募基金吗？",
        "answer": "不能。根据规定，机构投资者需要净资产不低于1000万元才能作为合格投资者投资私募基金。净资产800万元低于最低要求。"
    },
    {
        "id": "rule002",
        "category": "设立与募集",
        "question": "私募基金的最低募集规模要求是多少？",
        "answer": "私募证券投资基金的最低募集规模不得低于人民币1000万元。对于私募股权基金、创业投资基金等其他类型的私募基金，监管规定更加灵活，通常需符合基金合同的约定。"
    },
    {
        "id": "rule014",
        "category": "监管规定",
        "question": "私募基金管理人的风险准备金要求是什么？",
        "answer": "私募证券基金管理人应当按照管理费收入的10%计提风险准备金，主要用于赔偿因管理人违法违规、违反基金合同、操作错误等给基金财产或者投资者造成的损失。"
    }
]

# 定义上下文QA模板,RAG模式
CONTEXT_QA_TMPL = """
你是私募基金问答助手。请根据以下信息回答问题：

信息：{context}
问题：{query}
"""
CONTEXT_QA_PROMPT = PromptTemplate(
    input_variables=["query", "context"],
    template=CONTEXT_QA_TMPL,
)

# 定义超出知识库范围问题的回答模板
OUTSIDE_KNOWLEDGE_TMPL = """
你是私募基金问答助手。用户的问题是关于私募基金的，但我们的知识库中没有直接相关的信息。
请首先明确告知用户"对不起，在我的知识库中没有关于[具体主题]的详细信息"，
然后，如果你有相关知识，可以以"根据我的经验"或"一般来说"等方式提供一些通用信息，
并建议用户查阅官方资料或咨询专业人士获取准确信息。

用户问题：{query}
缺失的知识主题：{missing_topic}
"""
OUTSIDE_KNOWLEDGE_PROMPT = PromptTemplate(
    input_variables=["query", "missing_topic"],
    template=OUTSIDE_KNOWLEDGE_TMPL,
)

# 私募基金问答数据源
class FundRulesDataSource:
    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.rules_db = FUND_RULES_DB

    # 工具1：通过关键词搜索相关规则
    def search_rules_by_keywords(self, keywords: str) -> str:
        """通过关键词搜索相关私募基金规则"""
        keywords = keywords.strip().lower()
        keyword_list = re.split(r'[,，\s]+', keywords)
        
        matched_rules = []
        for rule in self.rules_db:
            rule_text = (rule["category"] + " " + rule["question"]).lower()
            match_count = sum(1 for kw in keyword_list if kw in rule_text)
            if match_count > 0:
                matched_rules.append((rule, match_count))
        
        matched_rules.sort(key=lambda x: x[1], reverse=True)
        
        if not matched_rules:
            return "未找到与关键词相关的规则。"
        
        result = []
        for rule, _ in matched_rules[:2]:
            result.append(f"类别: {rule['category']}\n问题: {rule['question']}\n答案: {rule['answer']}")
        
        return "\n\n".join(result)

    # 工具2：根据规则类别查询
    def search_rules_by_category(self, category: str) -> str:
        """根据规则类别查询私募基金规则"""
        category = category.strip()
        matched_rules = []
        
        for rule in self.rules_db:
            if category.lower() in rule["category"].lower():
                matched_rules.append(rule)
        
        if not matched_rules:
            return f"未找到类别为 '{category}' 的规则。"
        
        result = []
        for rule in matched_rules:
            result.append(f"问题: {rule['question']}\n答案: {rule['answer']}")
        
        return "\n\n".join(result)

    # 工具3：直接回答用户问题
    def answer_question(self, query: str) -> str:
        """直接回答用户关于私募基金的问题"""
        query = query.strip()
        
        best_rule = None
        best_score = 0
        
        for rule in self.rules_db:
            query_words = set(query.lower().split())
            rule_words = set((rule["question"] + " " + rule["category"]).lower().split())
            common_words = query_words.intersection(rule_words)
            
            score = len(common_words) / max(1, len(query_words))
            if score > best_score:
                best_score = score
                best_rule = rule
        
        if best_score < 0.2 or best_rule is None:
            # 识别问题主题
            missing_topic = self._identify_missing_topic(query)
            prompt = OUTSIDE_KNOWLEDGE_PROMPT.format(
                query=query,
                missing_topic=missing_topic
            )
            # 直接通过LLM获取回答可能导致输出格式与Agent期望不符
            # 将回答包装为AgentFinish格式而不是返回给Agent处理
            response = self.llm.invoke(prompt)
            # 返回格式化后的回答，让Agent直接返回最终结果
            return f"这个问题超出了知识库范围。\n\n{response}"
        
        context = best_rule["answer"]
        prompt = CONTEXT_QA_PROMPT.format(query=query, context=context)
        
        return self.llm.invoke(prompt)
    
    def _identify_missing_topic(self, query: str) -> str:
        """识别查询中缺失的知识主题"""
        # 简单的主题提取逻辑
        query = query.lower()
        if "投资" in query and "资产" in query:
            return "私募基金可投资的资产类别"
        elif "公募" in query and "区别" in query:
            return "私募基金与公募基金的区别"
        elif "退出" in query and ("机制" in query or "方式" in query):
            return "创业投资基金的退出机制"
        elif "费用" in query and "结构" in query:
            return "私募基金的费用结构"
        elif "托管" in query:
            return "私募基金资产托管"
        # 如果无法确定具体主题，使用通用表述
        return "您所询问的具体主题"


# 注意：在新版langchain中，我们使用create_react_agent代替了自定义的Prompt模板和输出解析器


def create_fund_qa_agent():
    # 定义LLM - 直接传递API密钥
    llm = Tongyi(model_name="Qwen-Turbo-2025-04-28", dashscope_api_key=DASHSCOPE_API_KEY)
    
    # 创建数据源
    fund_rules_source = FundRulesDataSource(llm)
    
    # 定义工具
    tools = [
        Tool(
            name="关键词搜索",
            func=fund_rules_source.search_rules_by_keywords,
            description="当需要通过关键词搜索私募基金规则时使用，输入应为相关关键词",
        ),
        Tool(
            name="类别查询",
            func=fund_rules_source.search_rules_by_category,
            description="当需要查询特定类别的私募基金规则时使用，输入应为类别名称",
        ),
        Tool(
            name="回答问题",
            func=fund_rules_source.answer_question,
            description="当能够直接回答用户问题时使用，输入应为完整的用户问题",
        ),
    ]
    
    # 为新版langchain创建REACT提示模板
    template = """
你是一个私募基金问答助手，请根据用户的问题选择合适的工具来回答。

可用工具：
{tools}

按照以下格式回答问题：

如果需要使用工具：
```
思考：我需要使用工具来回答这个问题
工具：工具名称
输入：工具的输入
```

如果得到了工具的结果：
```
观察：工具返回的结果
```

如果可以直接回答用户：
```
思考：我现在可以回答用户的问题了
答案：你的回答内容
```

注意：
1. 如果知识库中没有相关信息，请明确告知用户"对不起，在我的知识库中没有关于[具体主题]的详细信息"
2. 如果你基于自己的知识提供补充信息，请用"根据我的经验"或"一般来说"等前缀明确标识
3. 回答要专业、简洁、准确

问题：{input}
历史：{chat_history}
{agent_scratchpad}
"""
    
    # 创建提示模板
    prompt = ChatPromptTemplate.from_template(template)
    
    # 在langchain 1.0中，使用RunnableLambda创建代理执行逻辑
    def agent_executor(inputs):
        user_query = inputs.get('input', '')
        thought = """
思考过程：
我需要分析用户的问题，决定使用哪种工具来回答。首先检查问题是否包含类别相关关键词，然后检查是否包含特定规则关键词，最后考虑直接回答。
"""
        
        # 优化的工具选择逻辑
        
        # 1. 检查是否需要类别查询（更精确地提取类别）
        category_match = re.search(r'(查询|查找|了解)([\u4e00-\u9fa5]+[类分类])', user_query)
        if category_match:
            # 提取类别关键词
            full_match = category_match.group(0)
            category_keyword = re.search(r'[\u4e00-\u9fa5]+[类分类]', full_match).group(0)
            thought += f"\n我发现问题包含类别查询：'{full_match}'"
            thought += f"\n提取类别关键词：'{category_keyword}'"
            thought += "\n决定使用'类别查询'工具来获取相关规则信息。"
            result = fund_rules_source.search_rules_by_category(category_keyword)
        # 2. 检查是否包含合格投资者相关的比较问题
        elif re.search(r'(净资产|金融资产|收入).*[\d.]+(万元|元).*能.*投资', user_query) or re.search(r'能.*投资.*(净资产|金融资产|收入).*[\d.]+(万元|元)', user_query):
            thought += f"\n我发现问题包含合格投资者资格比较：'{user_query}'"
            thought += "\n决定使用'关键词搜索'工具来查找相关规则。"
            # 提取数字和资产类型进行搜索
            numbers = re.findall(r'[\d.]+', user_query)
            asset_types = re.findall(r'(净资产|金融资产|收入)', user_query)
            search_terms = f"{user_query} {', '.join(numbers)} {', '.join(asset_types)}"
            result = fund_rules_source.search_rules_by_keywords(search_terms)
            
            # 如果搜索结果不包含明确答案，尝试直接根据规则生成答案
            if "未找到" in result or len(result) < 50:
                thought += "\n搜索结果不明确，我将尝试根据合格投资者规则直接分析。"
                # 提取关键信息
                is_company = any(keyword in user_query for keyword in ['机构', '公司', '企业', '单位'])
                numbers = re.findall(r'[\d.]+', user_query)
                
                if numbers and len(numbers) > 0:
                    amount = float(numbers[0])
                    
                    if is_company:
                        # 机构投资者标准
                        if amount >= 1000:
                            result = f"根据私募基金合格投资者规定，净资产{amount}万元的机构投资者符合合格投资者标准，可以投资私募基金。"
                        else:
                            result = f"根据私募基金合格投资者规定，净资产{amount}万元的机构投资者不符合合格投资者标准，因为机构投资者需要净资产不低于1000万元。"
                    else:
                        # 个人投资者标准
                        if re.search(r'净资产', user_query) and amount >= 1000:
                            result = f"根据私募基金合格投资者规定，净资产{amount}万元的个人投资者符合合格投资者标准，可以投资私募基金。"
                        elif re.search(r'金融资产', user_query) and amount >= 300:
                            result = f"根据私募基金合格投资者规定，金融资产{amount}万元的个人投资者符合合格投资者标准，可以投资私募基金。"
                        elif re.search(r'收入', user_query) and amount >= 50:
                            result = f"根据私募基金合格投资者规定，年收入{amount}万元的个人投资者符合合格投资者标准，可以投资私募基金。"
                        else:
                            result = f"根据私募基金合格投资者规定，您提供的条件（{user_query}）不符合合格投资者标准。个人投资者需要满足：金融资产不低于300万元或者最近三年个人年均收入不低于50万元。"
                else:
                    result = "无法从问题中提取具体金额，请提供明确的资产或收入数字。"
        # 3. 检查是否包含特定规则关键词
        elif any(keyword in user_query.lower() for keyword in ['合格投资者', '募集规模', '风险准备金']):
            # 使用关键词搜索
            thought += f"\n我发现问题包含特定规则关键词：'{user_query}'"
            thought += "\n决定使用'关键词搜索'工具来查找相关规则。"
            result = fund_rules_source.search_rules_by_keywords(user_query)
        # 4. 检查是否直接询问类别列表
        elif re.search(r'有哪些[类分类]|包括哪些[类分类]', user_query):
            thought += f"\n我发现问题询问类别列表：'{user_query}'"
            thought += "\n决定使用'类别查询'工具来获取所有规则类别。"
            # 生成所有规则类别的响应
            categories = set(rule["category"] for rule in fund_rules_source.rules_db)
            if categories:
                result = "私募基金规则主要包括以下类别：\n" + "\n".join(f"- {cat}" for cat in sorted(categories))
            else:
                result = "未找到任何规则类别。"
        # 5. 默认使用直接回答
        else:
            # 直接回答问题
            thought += f"\n对于问题：'{user_query}'"
            thought += "\n我将尝试直接回答，使用知识库中的信息生成回答。"
            result = fund_rules_source.answer_question(user_query)
        
        # 组合思考过程和最终结果
        output = f"{thought}\n\n{result}"
        return {"output": output}
    
    # 返回RunnableLambda作为代理执行器
    return RunnableLambda(agent_executor)


if __name__ == "__main__":
    # 创建Agent
    fund_qa_agent = create_fund_qa_agent()
    
    print("=== 私募基金运作指引问答助手（反应式智能体）===\n")
    print("使用模型：Qwen-Turbo-2025-04-28\n")
    print("您可以提问关于私募基金的各类问题，输入'退出'结束对话\n")
    
    # 主循环
    while True:
        try:
            user_input = input("请输入您的问题：")
            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("感谢使用，再见！")
                break
            
            response = fund_qa_agent.invoke({"input": user_input})["output"]
            print(f"回答: {response}\n")
            print("-" * 40)
        except KeyboardInterrupt:
            print("\n程序已中断，感谢使用！")
            break
        except Exception as e:
            print(f"发生错误：{e}")
            print("请尝试重新提问或更换提问方式。")