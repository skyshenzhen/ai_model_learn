# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         generate_by_txt
# Description:
# Author:       shaver
# Date:         2025/6/3
# -------------------------------------------------------------------------------

from textwrap import dedent
import datetime
from crewai import LLM, Agent, Task, Crew

# 定义大模型
llm = LLM(
    model="deepseek-chat",
    temperature=0.7,
    base_url="https://api.deepseek.com/v1",
    api_key="sk-aa3a24ed00744632943daf833f298803"
)


class GenerateByTxt:

    # 构建crewai的步骤
    def create_crewai_setup(bytes_data=None):
        auto_test_requirement_agent = Agent(
            role="软件测试需求分析工程师",
            goal="通过用户提供描述，分析其中的测试需求",
            backstory=dedent(f"""
            你是一名拥有10年经验的软件测试需求分析专家，请根据用户提供的描述编写需求列表，且让你的回答尽可能详细。
            下面是用户提供描述：
                {bytes_data}
    
            """),
            allow_delegation=False,
            verbose=True,
            llm=llm
        )

        auto_test_case_agent = Agent(
            role="需求用例编写工程师",
            goal="分析测试需求列表，编写高质量的测试用例",
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
            llm=llm
        )

        auto_test_requirement_task = Task(
            description="根据用户提供的描述，分解出详细的测试需求列表",
            expected_output="详细的需求列表",
            agent=auto_test_requirement_agent,
        )

        auto_test_case_task = Task(
            description="根据需求列表，编写测试用例",
            expected_output="详细的测试用例",
            agent=auto_test_case_agent,
        )

        crew = Crew(
            agents=[
                auto_test_requirement_agent,
                auto_test_case_agent
            ],
            tasks=[
                auto_test_requirement_task,
                auto_test_case_task
            ],
            verbose=True,
        )

        crew_result = crew.kickoff()

        return crew_result

    @staticmethod
    def add_log_to_file(tag: str, result: str):
        # 追加 每次写一行
        # 格式：生成时间：2021-06-01 12:00:00 生成标记：tag 生成结果：{result}
        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now()} 生成标记：{tag} 生成结果：{result}\n")
