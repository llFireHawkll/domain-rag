from crewai import Task
from crewai.project import task

from domain_rag.src.models.single_agent.agents import SingleAgentSystem


class SingleAgentTasks(SingleAgentSystem):
    @task
    def domain_task(self) -> Task:
        return Task(
            config=self.tasks_config["domain_task"],
            agent=self.domain_expert(),
        )
