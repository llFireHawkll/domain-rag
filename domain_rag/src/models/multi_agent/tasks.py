from crewai import Task
from crewai.project import task

from domain_rag.src.models.multi_agent.agents import MultiAgentSystem


class MultiAgentTasks(MultiAgentSystem):
    @task
    def domain_qa_task_1(self) -> Task:
        return Task(
            config=self.tasks_config["domain_qa_task_1"],
            agent=self.domain_expert(),
        )

    @task
    def domain_qa_task_2(self) -> Task:
        return Task(
            config=self.tasks_config["domain_qa_task_2"],
            agent=self.domain_expert(),
        )

    @task
    def domain_qa_task_3(self) -> Task:
        return Task(
            config=self.tasks_config["domain_qa_task_3"],
            agent=self.domain_expert(),
        )

    @task
    def hallucination_grader_task(self) -> Task:
        return Task(
            config=self.tasks_config["hallucination_grader_task"],
            agent=self.hallucination_grader(),
            context=[
                self.domain_qa_task_1(),
                self.domain_qa_task_2(),
                self.domain_qa_task_3(),
            ],
        )

    @task
    def answer_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["answer_writing_task"],
            agent=self.short_answer_writer(),
            context=[
                self.domain_qa_task_1(),
                self.domain_qa_task_2(),
                self.domain_qa_task_3(),
                self.hallucination_grader_task(),
            ],
        )
