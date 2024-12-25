from pathlib import Path

import yaml
from crewai import Agent
from crewai.project import agent
from langchain_openai import AzureChatOpenAI

root = Path().absolute()
config_dir = root / "domain_rag/src/models/single_agent/config"


class SingleAgentSystem:
    def __init__(self, llm: AzureChatOpenAI) -> None:
        # Step-1: Get the LLM model
        self.llm = llm

        # Step-2: Load the configurations defined in the yaml
        # file to load the agents and their tasks
        self.agents_config = yaml.safe_load(open(config_dir / "agents.yaml"))
        self.tasks_config = yaml.safe_load(open(config_dir / "tasks.yaml"))

    @agent
    def domain_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["domain_expert"],
            llm=self.llm,
        )
