# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         generate_by_swagger
# Description:
# Author:       shaver
# Date:         2025/6/3
# -------------------------------------------------------------------------------

from crewai import LLM, Agent, Task, Crew
from textwrap import dedent

from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
from crewai.tools import tool

llm = LLM(
    model="deepseek-chat",
    temperature=0.7,
    base_url="https://api.deepseek.com/v1",
    api_key="sk-aa3a24ed00744632943daf833f298803"
)


class Tools:
    @staticmethod
    def scrapeWebsiteTool(swagger_url: str):
        return ScrapeWebsiteTool(website_url=swagger_url)


class APIAgents:
    def __init__(self):
        self.tools = Tools()
        self.llm = llm

    # 需求分析智能体
    def requirements_api_agents(self, swagger_url: str):
        return Agent(
            role="软件测试需求分析工程师",
            goal="分析通过工具获取接口信息，分析其中的测试需求",
            backstory=dedent("""
            你是一位根据用户提供的Restful API JSON描述文件获取测试需求的代理，请让你的回答尽可能详细。
            """),
            allow_delegation=False,
            verbose=True,
            tools=[self.tools.scrapeWebsiteTool(swagger_url)],
            llm=self.llm
        )

    # 需求用例编写智能体
    def testcase_writer_agents(self, swagger_url: str):
        return Agent(
            role="需求用例编写工程师",
            goal="分析测试需求列表，编写高质量的、结构化的测试用例",
            backstory=dedent("""
            你是一位专业的软件测试用例编写工程师，请严格按照以下格式输出测试用例：

            {
                "module": "模块名称",
                "test_cases": [
                    {
                        "title": "测试用例标题(明确表达测试目的)",
                        "steps": [
                            {"step": 1, "action": "操作描述"},
                            {"step": 2, "action": "操作描述"}
                        ],
                        expected_result: [
                            {"step": 1, "expected": "预期结果"},
                            {"step": 2, "expected": "预期结果"}
                        ],
                        "test_type": "手动/自动",
                        "case_type": "功能测试/性能测试/安全测试",
                        "priority": "P0/P1/P2",
                        "preconditions": "前置条件",
                        "estimated_time": "预估工时(可选)",
                        "remarks": "备注(可选)"
                    },
                    # 更多测试用例...
                ]
            }

            要求：
            1. 每个测试用例必须包含上述所有字段
            2. 步骤(steps中的action)和预期结果(expected_result中的expected)必须一一对应
            3. 使用规范的JSON格式输出，方便后续处理
            """),
            allow_delegation=False,
            verbose=True,
            tools=[self.tools.scrapeWebsiteTool(swagger_url)],
            llm=self.llm
        )


class APITasks:
    def requirements_task(self, agent):
        task = Task(
            description="调用工具获取需求信息，分解出详细的测试需求列表",
            expected_output="详细的需求列表",
            agent=agent,
        )
        return task

    def testcase_task(self, agent):
        task = Task(
            description="根据需求列表，编写测试用例，并进行测试",
            expected_output="详细的测试用例列表",
            agent=agent,
        )
        return task


class GenerateBySwagger:
    @staticmethod
    def create_crewai_setup(swagger_url=None):
        # 初始化智能体
        agents = APIAgents()
        requirement_api_agent = agents.requirements_api_agents(swagger_url)
        testcase_writer_agent = agents.testcase_writer_agents(swagger_url)

        # 设置任务
        tasks = APITasks()
        requirement_task = tasks.requirements_task(requirement_api_agent)
        testcase_task = tasks.testcase_task(testcase_writer_agent)

        # 创建并执行Crew
        crew = Crew(
            agents=[requirement_api_agent, testcase_writer_agent],
            tasks=[requirement_task, testcase_task],
            verbose=True,
        )

        crew_result = crew.kickoff()
        return crew_result
