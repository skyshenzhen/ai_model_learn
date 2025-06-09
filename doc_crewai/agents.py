# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         agents
# Description:  智能体
# Author:       shaver
# Date:         2025/5/20
# -------------------------------------------------------------------------------
from textwrap import dedent
from crewai import Agent

from doc_crewai.llms import deepseek_llm
from doc_crewai.tools import Tools


class APIAgents:
    def __init__(self):
        self.tools = Tools()
        self.llm = deepseek_llm()

    # 需求分析智能体
    def requirements_api_agents(self):
        return Agent(
            role="软件测试需求分析工程师",
            goal="分析通过工具获取接口信息，分析其中的测试需求",
            backstory=dedent("""
            你是一位根据用户提供的Restful API JSON描述文件获取测试需求的代理，请让你的回答尽可能详细。
            """),
            allow_delegation=False,
            verbose=True,
            tools=[self.tools.scrapeWebsiteTool()],
            llm=self.llm
        )

    # 需求用例编写智能体
    def testcase_writer_agents(self):
        return Agent(
            role="需求用例编写工程师",
            goal="分析测试需求列表，编写高质量的测试用例",
            backstory=dedent("""
            你是一位专业的软件测试用例编写工程师，请让你的回答尽可能详细。
            """),
            allow_delegation=False,
            verbose=True,
            tools=[self.tools.scrapeWebsiteTool()],
            llm=self.llm
        )

    # 高级测试开发智能体
    def senior_engineer_agents(self):
        return Agent(
            role="高级软件测试开发工程师",
            goal="使用pytest编写高质量测试用例",
            backstory=dedent("""
            你是一位高级软件测试开发工程师。
            你在Python编程方面有专业知识，擅长使用pytest编写测试用例，并且致力于产生完美的代码。
            """),
            allow_delegation=False,
            verbose=True,
            tools=[self.tools.scrapeWebsiteTool()],
            llm=self.llm
        )

    # 质量智能体
    def qa_engineer_agents(self):
        return Agent(
            role="软件质量控制工程师",
            goal="通过分析给定的代码以查找错误，创建完美的代码",
            backstory=dedent("""
            你是一位专门检查代码错误的软件工程师。你对细节有敏锐的洞察力，并且擅长发现隐藏的bug。
            你会检查缺失的导入，变量声明，不匹配的括号和语法错误等内容。
            你还会检查安全漏洞和逻辑错误。
            """),
            allow_delegation=False,
            verbose=True,
            tools=[self.tools.scrapeWebsiteTool()],
            llm=self.llm
        )

    # 将代码保存智能体
    def python_file_writer_agents(self):
        return Agent(
            role="保存文件智能体",
            goal="对生成的代码进行全面检查，并将Python文件保存到指定文件中",
            backstory=dedent("""
            你是一位高级软件质量控制工程师，善于对代码进行全面的检查
            """),
            allow_delegation=False,
            verbose=True,
            tools=[self.tools.save_python_to_file],
            llm=self.llm
        )
