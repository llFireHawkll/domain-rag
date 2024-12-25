import os
from time import sleep
from typing import Dict, Literal

import pandas as pd
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from loguru import logger
from ragas.run_config import RunConfig
from ragas.testset.evolutions import (conditional, multi_context, reasoning,
                                      simple)
from ragas.testset.generator import TestsetGenerator

from domain_rag.src.utils.basic_utils import check_and_create_path


class QADatasetGenerator:
    def __init__(self, max_workers=8, max_wait=60, timeout=60, max_retries=3) -> None:
        # Define the RunConfig
        self.run_config = RunConfig(
            timeout=timeout,
            max_retries=max_retries,
            max_wait=max_wait,  # default: 60
            max_workers=max_workers,  # default: 16
        )

    def configure_llm(
        self,
        llm_type: Literal["azure"],
        generator_llm_config: Dict,
        critic_llm_config: Dict,
        embedding_llm_config: Dict,
    ):
        if llm_type == "azure":
            generator_llm = AzureChatOpenAI(**generator_llm_config)
            critic_llm = AzureChatOpenAI(**critic_llm_config)
            embedder_llm = AzureOpenAIEmbeddings(**embedding_llm_config)

        else:
            raise NotImplementedError(f"llm_type: {llm_type} not implemented!!!")

        return generator_llm, critic_llm, embedder_llm

    def configure_generator(
        self, generator_llm, critic_llm, embedder_llm, chunk_size=1024
    ):
        generator = TestsetGenerator.from_langchain(
            generator_llm=generator_llm,
            critic_llm=critic_llm,
            embeddings=embedder_llm,
            chunk_size=chunk_size,
        )

        return generator

    def _list_batcher(self, input_list, batch_size):
        # Step-1: Initialize an empty list to store the batches
        batches = []

        # Step-2: Loop through the list in steps of batch_size
        for i in range(0, len(input_list), batch_size):
            # Step-3: Slice the list from i to i+batch_size
            # and append to batches
            batch = input_list[i : i + batch_size]
            batches.append(batch)

        return batches

    def _generate_combine_dataframe(self, output_path: str, intermediate_path: str):
        try:
            # Step-1: List to hold DataFrames
            dataframes = []

            # Step-2: Iterate over all files in the directory
            for filename in os.listdir(intermediate_path):
                # Step-3: Check if the file is a CSV
                if filename.endswith(".csv"):
                    file_path = os.path.join(intermediate_path, filename)

                    # Step-4: Read the CSV file into a DataFrame
                    df = pd.read_csv(file_path)

                    # Step-5: Append the DataFrame to the list
                    dataframes.append(df)

            # Step-6: Concatenate all DataFrames
            combined_df = pd.concat(dataframes, ignore_index=True)

            # Step-7: Write the combined DataFrame to a CSV file
            output_file = output_path + "combined_dataframe.csv"
            combined_df.to_csv(output_file, index=False)

        except Exception as error:
            logger.error(
                f"Unable to generate combined dataframe due to the following error: {error}"
            )

    def generate_qa_dataset(
        self,
        generator,
        documents,
        output_path: str,
        batch_size: int = 100,
        no_of_questions: int = 10,
    ):

        # Step-0: Check if the output_path is present
        # or not else create the necessary path
        batch_df_path = output_path + "/intermediate/"
        combined_df_path = output_path + "/combined/"

        check_and_create_path(path=batch_df_path)
        check_and_create_path(path=combined_df_path)

        # Step-1: Get the document batches
        batches = self._list_batcher(input_list=documents, batch_size=batch_size)

        # Step-2: Iterate through the batches and then run
        # the qa generator
        for i, batch in enumerate(batches):
            logger.info(f"Processing Batch {i + 1}")

            try:
                if os.path.isfile(batch_df_path + f"/qa_batch_{i+1}.csv"):
                    logger.info(f"Skipping Already Processed Batch {i + 1}")
                else:
                    # Step-3: Generating the questions using custom ragas module
                    testset = generator.generate_with_langchain_docs(
                        batch,
                        test_size=no_of_questions * len(batch),
                        distributions={
                            simple: 0.15,
                            reasoning: 0.5,
                            conditional: 0.10,
                            multi_context: 0.25,
                        },
                        run_config=self.run_config,
                        is_async=True,
                        raise_exceptions=False,
                    )

                    # Step-4: Store the generated dataframe
                    # onto a given output path
                    questions_dataframe = testset.to_pandas()
                    file_path = batch_df_path + f"/qa_batch_{i+1}.csv"
                    questions_dataframe.to_csv(file_path, index=False)

                    # Step-5: Sleep for 20 seconds
                    sleep(20)

            except Exception as error:
                logger.error(
                    f"Unable to generate the dataset due to the following error: {error}"
                )

        # Step-5: Comnbine the all the dataframes into
        # a single massive dataframe
        self._generate_combine_dataframe(
            output_path=combined_df_path, intermediate_path=batch_df_path
        )
