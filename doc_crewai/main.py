# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Name:         main
# Description:  主函数
# Author:       shaver
# Date:         2025/5/20
# -------------------------------------------------------------------------------
from crewai import Crew

from doc_crewai.agents import APIAgents
from doc_crewai.tasks import APITasks


#--------------------------智能体----------
agents = APIAgents()
requirement_api_agent = agents.requirements_api_agents()
testcase_writer_agent = agents.testcase_writer_agents()
senior_engineer_agent = agents.senior_engineer_agents()
qa_engineer_agent = agents.qa_engineer_agents()
python_file_writer_agent = agents.python_file_writer_agents()

#--------------------------任务--------------------
tasks = APITasks()
requirement_task = tasks.requirements_task(requirement_api_agent)
testcase_task = tasks.testcase_task(testcase_writer_agent)
senior_engineer_task = tasks.code_task(senior_engineer_agent)
qa_engineer_task = tasks.review_task(qa_engineer_agent)
file_generate_task = tasks.file_generate_task(python_file_writer_agent)

crew = Crew(
    agents=[
        requirement_api_agent,
        testcase_writer_agent,
        senior_engineer_agent,
        qa_engineer_agent,
        python_file_writer_agent,
    ],
    tasks=[
        requirement_task,
        testcase_task,
        senior_engineer_task,
        qa_engineer_task,
        file_generate_task,
    ],
    verbose=True,
)

game = crew.kickoff()

print("\n################")
print("## Here is the result")
print("################\n")
print("final code for the api testcase:")
print(game)