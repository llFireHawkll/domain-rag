from typing import Dict

from datasets import Dataset
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (answer_correctness, answer_relevancy,
                           answer_similarity, context_entity_recall,
                           context_precision, context_recall, faithfulness)
from ragas.metrics.critique import harmfulness
from ragas.run_config import RunConfig


class RAGExaminer:
    def __init__(
        self,
        llm_config: Dict,
        embedding_llm_config: Dict,
        max_workers=16,
        max_wait=60,
        timeout=60,
        max_retries=3,
    ) -> None:
        self.llm = AzureChatOpenAI(**llm_config)
        self.embedder = AzureOpenAIEmbeddings(**embedding_llm_config)
        self.run_config = RunConfig(
            timeout=timeout,
            max_retries=max_retries,
            max_wait=max_wait,  # default: 60
            max_workers=max_workers,  # default: 16
        )

    def examine_and_report_metrics(self, dataset_dict: Dict):
        # Step-1: Convert the given input dataset_dict
        # into a format which ragas understand
        evaluation_dataset = Dataset.from_dict(dataset_dict)

        # Step-2: Use the evaluate function to compute
        # the different metrics defined in the list
        score = evaluate(
            dataset=evaluation_dataset,
            llm=self.llm,
            embeddings=self.embedder,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                context_entity_recall,
                answer_similarity,
                answer_correctness,
                harmfulness,
            ],
            raise_exceptions=False,
            run_config=self.run_config,
        )

        # Step-3: Convert the scores into
        # score dataframes
        score_dataframe = score.to_pandas()

        return score_dataframe
