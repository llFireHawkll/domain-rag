import time
from typing import Dict

import numpy as np
from crewai import Crew
from crewai.process import Process
from langchain_openai import AzureChatOpenAI

from domain_rag.src.models.single_agent.agents import SingleAgentSystem
from domain_rag.src.models.single_agent.tasks import SingleAgentTasks


class SingleAgentCrew:
    def __init__(self, llm_config: Dict, vectorstore) -> None:
        # Step-1: Initiate the multi agentic crew
        self.llm = self._initiate_llm(llm_config=llm_config)
        self.agents = self._initiate_agents()
        self.tasks = self._initiate_tasks()
        self.crew = self._deploy_agentic_crew()
        self.vectorstore = vectorstore

        # Step-2: Attach all the required tasks into
        # the crew
        self._deploy_crew_tasks()

    def _initiate_llm(self, llm_config: Dict):
        return AzureChatOpenAI(**llm_config)

    def _initiate_agents(self):
        return SingleAgentSystem(llm=self.llm)

    def _initiate_tasks(self):
        return SingleAgentTasks(llm=self.llm)

    def _deploy_agentic_crew(self):
        return Crew(
            agents=[
                self.agents.domain_expert(),
            ],
            process=Process.sequential,
            verbose=0,
        )

    def _deploy_crew_tasks(self):
        self.crew.tasks = [self.tasks.domain_task()]

    def _generate_input_context(self, query: str) -> Dict:
        # Step-1: Search top 3 similar docs from vectorstore using FAISS
        documents = self.vectorstore.similarity_search(query, k=3)

        # Step-2: create the input-context
        context = {
            "query": query,
            "document_1": documents[0],
            "document_2": documents[1],
            "document_3": documents[2],
        }
        return context

    def get_query_response(self, row):
        # Step-1: We need to compute the response and time
        start_time = time.process_time()
        context = self._generate_input_context(query=row["question"])
        response = self.crew.kickoff(inputs=context)
        end_time = time.process_time()

        # Step-2: Also round the time in secs to 4 decimal places
        time_diff = np.round(end_time - start_time, 4)

        return (
            response,
            time_diff,
            self.crew.usage_metrics["total_tokens"],
            self.crew.usage_metrics["successful_requests"],
        )
