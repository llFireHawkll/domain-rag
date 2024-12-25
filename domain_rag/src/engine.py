import os
import time
from textwrap import dedent
from typing import Dict, Literal

import pandas as pd
from langchain_community.document_loaders import JSONLoader
from langchain_openai import AzureOpenAIEmbeddings
from loguru import logger
from tqdm import tqdm

from domain_rag.src.data.loader.preprocessing import QADatasetGenerator
from domain_rag.src.data.vectorstore.base import CustomVectorStore
from domain_rag.src.evaluator.examiner import RAGExaminer
from domain_rag.src.evaluator.result_analyser import ResultAnalyser
from domain_rag.src.models.multi_agent.crew import MultiAgentCrew
from domain_rag.src.models.single_agent.crew import SingleAgentCrew
from domain_rag.src.utils.basic_utils import (
    check_and_create_path,
    create_concatenated_dataframe,
    read_yaml_file,
)

import gradio as gr

tqdm.pandas()


class DomainSpecificRAG:
    def __init__(self, config: Dict):
        # Step-1: Get configurations for different modules
        self.qa_config = config["qa_config"]
        self.vecstore_config = config["vecstore_config"]
        self.agents_config = config["agents_config"]
        self.evaluation_config = config["evaluation_config"]
        self.plot_config = config["plot_config"]

        # Step-2: Initalize internal custom packages
        self.custom_vs = CustomVectorStore()

    def create_qa_dataset_for_benchmarking(
        self, domain: Literal["clinical", "finance", "research"]
    ):
        # Step-1: Load the raw dataset related to domain
        if domain == "clinical":

            # Step-1a: Using JSONLoader to load clinical guidelines data
            logger.info("Starting data loading from JSONL data format...")
            loader = JSONLoader(
                file_path=self.qa_config["input_file_path"],
                jq_schema=".clean_text",
                text_content=True,
                json_lines=True,
            )

            # Step-1b: Load the data into memory
            raw_documents = loader.load()
            logger.success("Loading Complete...")

            # Step-1c: Keep only documents with given length
            logger.info(
                f"Filtering those documents which meet given document_size {self.qa_config['min_document_size']} characters criteria"
            )
            processed_documents = [
                doc
                for doc in raw_documents
                if len(doc.page_content) >= self.qa_config["min_document_size"]
            ]

            # Step-1d: Keep only specified numbers of documents
            logger.info(
                f"Selecting only top {self.qa_config['min_documents_required']} documents"
            )
            documents = processed_documents[: self.qa_config["min_documents_required"]]

        elif domain == "finance":
            pass
        elif domain == "research":
            pass
        else:
            raise NotImplementedError(
                "Only available domains are [clinical, finance, research]"
            )

        # Step-2: Initialize the QA Generator dataset
        qa_generator = QADatasetGenerator(max_workers=self.qa_config["max_workers"])

        # Step-3: Start configuring the LLM which needs to be used
        generator_llm, critic_llm, embedder_llm = qa_generator.configure_llm(
            llm_type=self.qa_config["llm_type"],
            generator_llm_config=self.qa_config["generator_llm_config"],
            critic_llm_config=self.qa_config["critic_llm_config"],
            embedding_llm_config=self.qa_config["embedding_llm_config"],
        )

        # Step-4: Create a testset generator which will generate the
        # QA dataset using the above LLMs
        generator = qa_generator.configure_generator(
            generator_llm=generator_llm,
            critic_llm=critic_llm,
            embedder_llm=embedder_llm,
            chunk_size=self.qa_config["chunk_size"],
        )

        # Step-5: Finally start the generator to generate QA pairs
        # with revelant context
        qa_generator.generate_qa_dataset(
            generator=generator,
            documents=documents,
            output_path=self.qa_config["output_path"],
            batch_size=self.qa_config["batch_size"],
            no_of_questions=self.qa_config["no_of_questions"],
        )

    def create_and_save_vectorstore(
        self, domain: Literal["clinical", "finance", "research"]
    ):
        # Step-1: Load the raw dataset related to domain
        if domain == "clinical":

            # Step-1a: Using JSONLoader to load clinical guidelines data
            logger.info("Starting data loading from JSONL data format...")
            loader = JSONLoader(
                file_path=self.vecstore_config["input_file_path"],
                jq_schema=".clean_text",
                text_content=True,
                json_lines=True,
            )

            # Step-1b: Load the data into memory
            raw_documents = loader.load()
            logger.success("Loading Complete...")

            # Step-1c: Keep only documents with given length
            logger.info(
                f"Filtering those documents which meet given document_size {self.vecstore_config['min_document_size']} characters criteria"
            )
            processed_documents = [
                doc
                for doc in raw_documents
                if len(doc.page_content) >= self.vecstore_config["min_document_size"]
            ]

            # Step-1d: Keep only specified numbers of documents
            logger.info(
                f"Selecting only top {self.vecstore_config['min_documents_required']} documents"
            )
            documents = processed_documents[
                : self.vecstore_config["min_documents_required"]
            ]

            # Step-1e: Use these documents and create a structure
            # for vectorstore usage
            texts = []

            for doc in documents:
                text = dedent(
                    f"""\
                ### START OF CLINICAL GUIDELINE PAGE
                {doc.page_content}
                ### END OF CLINICAL GUIDELINE PAGE"""
                )

                texts.append(text)

        elif domain == "finance":
            pass
        elif domain == "research":
            pass
        else:
            raise NotImplementedError(
                "Only available domains are [clinical, finance, research]"
            )

        # Step-2: Create vectorstore based on the choosen domain
        embedder_llm = AzureOpenAIEmbeddings(
            **self.vecstore_config["embedding_llm_config"]
        )
        base_vs = self.custom_vs.create_vectorstore(
            documents=texts,
            embedder_llm=embedder_llm,
            batch_size=self.vecstore_config["batch_size"],
            vectorstore_intermediate_location_path=self.vecstore_config[
                "vectorstore_intermediate_location_path"
            ],
        )

        # Step-3: Store the vectorstore in the defined location
        self.custom_vs.save_vectorstore(
            vectorstore=base_vs,
            save_location_path=self.vecstore_config["vectorstore_location_path"],
        )

    def load_vectorstore_in_memory(self):
        # Step-1: Load embedder LLM into the system memory
        embedder_llm = AzureOpenAIEmbeddings(
            **self.vecstore_config["embedding_llm_config"]
        )

        # Step-2: Load the specified vectorstore in memory
        vec_store = self.custom_vs.load_vectorstore(
            embedder_llm=embedder_llm,
            vectorstore_location_path=self.vecstore_config["vectorstore_location_path"],
        )

        return vec_store

    def get_query_response_via_agents(
        self,
        vectorstore,
        system_type: Literal[
            "single_agent", "multi_agent", "both_agentic_frameworks"
        ] = "multi_agent",
    ):
        # Step-1a: Load all the required configuration
        llm_config = self.agents_config["llm_config"]
        output_path = self.agents_config["output_path"]
        qa_dataframe = pd.read_csv(self.agents_config["qa_dataset_path"])

        # Step-1b: Check if the path exists or not
        check_and_create_path(path=output_path)
        check_and_create_path(path=output_path + "single_agent/")
        check_and_create_path(path=output_path + "multi_agent/")

        no_of_generated_questions = qa_dataframe.shape[0]
        batch_size = 10

        for index in tqdm(range(0, no_of_generated_questions, batch_size)):

            qa_batch_dataframe = qa_dataframe[index : index + batch_size]

            # Step-2: Check which system to initialize
            if system_type == "single_agent":
                # Step-3a: Initialize the single agent crew system
                single_agent_crew = SingleAgentCrew(
                    llm_config=llm_config, vectorstore=vectorstore
                )

                # Step-3b: Generate response of the questions
                qa_batch_dataframe[
                    ["answer", "response_time", "total_tokens", "successful_requests"]
                ] = qa_batch_dataframe.progress_apply(
                    single_agent_crew.get_query_response, axis=1, result_type="expand"
                )

                # Step-3c: Store the ouptut dataframe at the given location
                columns_to_keep = [
                    "question",
                    "ground_truth",
                    "answer",
                    "contexts",
                    "response_time",
                    "total_tokens",
                    "successful_requests",
                ]
                qa_batch_dataframe[columns_to_keep].to_csv(
                    output_path
                    + f"single_agent/single_agent_response_index_{index}.csv",
                    index=False,
                )

            elif system_type == "multi_agent":
                # Step-3a: Initialize the multi agent crew system
                multi_agent_crew = MultiAgentCrew(
                    llm_config=llm_config, vectorstore=vectorstore
                )

                # Step-3b: Generate response of the questions
                qa_batch_dataframe[
                    ["answer", "response_time", "total_tokens", "successful_requests"]
                ] = qa_batch_dataframe.progress_apply(
                    multi_agent_crew.get_query_response, axis=1, result_type="expand"
                )

                # Step-3c: Store the ouptut dataframe at the given location
                columns_to_keep = [
                    "question",
                    "ground_truth",
                    "answer",
                    "contexts",
                    "response_time",
                    "total_tokens",
                    "successful_requests",
                ]
                qa_batch_dataframe[columns_to_keep].to_csv(
                    output_path + f"multi_agent/multi_agent_response_index_{index}.csv",
                    index=False,
                )

            elif system_type == "both_agentic_frameworks":
                # Step-3a: Initialize the single and multi agent crew system
                single_agent_crew = SingleAgentCrew(
                    llm_config=llm_config, vectorstore=vectorstore
                )
                multi_agent_crew = MultiAgentCrew(
                    llm_config=llm_config, vectorstore=vectorstore
                )

                # Step-3b: Copy the original dataframe
                qa_dataframe_1 = qa_batch_dataframe.copy(deep=True)
                qa_dataframe_2 = qa_batch_dataframe.copy(deep=True)

                # Step-3b: Generate response of the questions
                qa_dataframe_1[
                    ["answer", "response_time", "total_tokens", "successful_requests"]
                ] = qa_dataframe.progress_apply(
                    single_agent_crew.get_query_response, axis=1, result_type="expand"
                )
                qa_dataframe_2[
                    ["answer", "response_time", "total_tokens", "successful_requests"]
                ] = qa_dataframe.progress_apply(
                    multi_agent_crew.get_query_response, axis=1, result_type="expand"
                )

                # Step-3c: Store the ouptut dataframe at the given location
                columns_to_keep = [
                    "question",
                    "ground_truth",
                    "answer",
                    "contexts",
                    "response_time",
                    "total_tokens",
                    "successful_requests",
                ]
                qa_dataframe_1[columns_to_keep].to_csv(
                    output_path
                    + f"single_agent/single_agent_response_index_{index}.csv",
                    index=False,
                )
                qa_dataframe_2[columns_to_keep].to_csv(
                    output_path + f"multi_agent/multi_agent_response_index_{index}.csv",
                    index=False,
                )

            else:
                raise NotImplementedError(
                    "Only available system_type are [single_agent, multi_agent, both_agentic_frameworks]"
                )

            time.sleep(30)

    def generate_agentic_rag_scores(
        self,
        system_type: Literal[
            "single_agent", "multi_agent", "both_agentic_frameworks"
        ] = "multi_agent",
    ):
        # Step-1: Load all the required configuration
        llm_config = self.evaluation_config["llm_config"]
        embedding_llm_config = self.evaluation_config["embedding_llm_config"]
        evaluation_data_path = self.evaluation_config["evaluation_data_path"]
        output_path = self.evaluation_config["output_path"]

        # Step-2: Initialize the RAGExaminer
        examiner = RAGExaminer(
            llm_config=llm_config, embedding_llm_config=embedding_llm_config
        )

        # Step-3: Check if the path exists or not
        check_and_create_path(path=output_path)
        check_and_create_path(path=output_path + "single_agent/")
        check_and_create_path(path=output_path + "multi_agent/")

        # Step-4: Check which system we will evaluate
        if system_type == "single_agent":
            for index, filename in enumerate(
                os.listdir(evaluation_data_path + "single_agent/")
            ):
                if os.path.isfile(
                    output_path
                    + f"single_agent/single_agent_scores_index_{index * 10}.csv"
                ):
                    logger.info(f"Skipping Already Processed File {index * 10}.csv")
                else:
                    if filename.endswith(".csv"):
                        # Step-4a: Load the evaluation dataset
                        eval_dataframe = pd.read_csv(
                            filepath_or_buffer=os.path.join(
                                evaluation_data_path + "single_agent/", filename
                            )
                        )

                        eval_dataframe["contexts"] = eval_dataframe["contexts"].apply(
                            lambda x: [x]
                        )

                        dataset_dict = {
                            "question": eval_dataframe["question"].tolist(),
                            "answer": eval_dataframe["answer"].tolist(),
                            "contexts": eval_dataframe["contexts"].tolist(),
                            "ground_truth": eval_dataframe["ground_truth"].tolist(),
                        }

                        # Step-4b: Examine results for single agent crew system
                        score_dataframe = examiner.examine_and_report_metrics(
                            dataset_dict=dataset_dict
                        )

                        # Step-4c: Store the ouptut dataframe at the given location
                        score_dataframe.to_csv(
                            output_path
                            + f"single_agent/single_agent_scores_index_{index * 10}.csv",
                            index=False,
                        )

                        time.sleep(30)

            sa_score_dataframe = create_concatenated_dataframe(
                directory_path=output_path + "single_agent/"
            )

            sa_score_dataframe.to_csv(
                output_path + "single_agent_scores.csv",
                index=False,
            )

        elif system_type == "multi_agent":
            for index, filename in enumerate(
                os.listdir(evaluation_data_path + "multi_agent/")
            ):
                if os.path.isfile(
                    output_path
                    + f"multi_agent/multi_agent_scores_index_{index * 10}.csv"
                ):
                    logger.info(f"Skipping Already Processed File {index * 10}.csv")
                else:
                    if filename.endswith(".csv"):
                        # Step-4a: Load the evaluation dataset
                        eval_dataframe = pd.read_csv(
                            filepath_or_buffer=os.path.join(
                                evaluation_data_path + "multi_agent/", filename
                            )
                        )

                        eval_dataframe["contexts"] = eval_dataframe["contexts"].apply(
                            lambda x: [x]
                        )

                        dataset_dict = {
                            "question": eval_dataframe["question"].tolist(),
                            "answer": eval_dataframe["answer"].tolist(),
                            "contexts": eval_dataframe["contexts"].tolist(),
                            "ground_truth": eval_dataframe["ground_truth"].tolist(),
                        }

                        # Step-4b: Initialize the single agent crew system
                        score_dataframe = examiner.examine_and_report_metrics(
                            dataset_dict=dataset_dict
                        )

                        # Step-4c: Store the ouptut dataframe at the given location
                        score_dataframe.to_csv(
                            output_path
                            + f"multi_agent/multi_agent_scores_index_{index * 10}.csv",
                            index=False,
                        )

                        time.sleep(20)

            # Step-: Final step to combine all the generated scores
            ma_score_dataframe = create_concatenated_dataframe(
                directory_path=output_path + "multi_agent/"
            )

            ma_score_dataframe.to_csv(
                output_path + "multi_agent_scores.csv",
                index=False,
            )

        elif system_type == "both_agentic_frameworks":
            # Step-4a: Load the evaluation dataset
            eval_dataframe_sa = pd.read_csv(
                evaluation_data_path + "single_agent_response.csv"
            )
            eval_dataframe_ma = pd.read_csv(
                evaluation_data_path + "multi_agent_response.csv"
            )

            eval_dataframe_sa["contexts"] = eval_dataframe_sa["contexts"].apply(
                lambda x: [x]
            )
            eval_dataframe_ma["contexts"] = eval_dataframe_ma["contexts"].apply(
                lambda x: [x]
            )

            dataset_dict_sa = {
                "question": eval_dataframe_sa["question"].tolist(),
                "answer": eval_dataframe_sa["answer"].tolist(),
                "contexts": eval_dataframe_sa["contexts"].tolist(),
                "ground_truth": eval_dataframe_sa["ground_truth"].tolist(),
            }
            dataset_dict_ma = {
                "question": eval_dataframe_ma["question"].tolist(),
                "answer": eval_dataframe_ma["answer"].tolist(),
                "contexts": eval_dataframe_ma["contexts"].tolist(),
                "ground_truth": eval_dataframe_ma["ground_truth"].tolist(),
            }

            # Step-4b: Initialize the single agent crew system
            score_dataframe_sa = examiner.examine_and_report_metrics(
                dataset_dict=dataset_dict_sa
            )
            score_dataframe_ma = examiner.examine_and_report_metrics(
                dataset_dict=dataset_dict_ma
            )

            # Step-4c: Store the ouptut dataframe at the given location
            score_dataframe_sa.to_csv(
                output_path + "single_agent_scores.csv", index=False
            )
            score_dataframe_ma.to_csv(
                output_path + "multi_agent_scores.csv", index=False
            )

        else:
            raise NotImplementedError(
                "Only available system_type are [single_agent, multi_agent, both_agentic_frameworks]"
            )

    def compare_and_evaluate_agentic_rag(self) -> None:
        # Step-1: Load the score dataframes for both the agent systems
        input_path = self.plot_config["score_input_path"]
        plots_output_path = self.plot_config["plots_output_path"]

        sa_score_df = pd.read_csv(input_path + "single_agent_scores.csv")
        ma_score_df = pd.read_csv(input_path + "multi_agent_scores.csv")

        # TODO: REMOVE THIS CHECK
        sa_score_df["execution_cost"] = 0.005
        sa_score_df["response_time"] = 0.24
        ma_score_df["execution_cost"] = 0.02
        ma_score_df["response_time"] = 1.02

        # Step-2: Check if the path exists or not
        check_and_create_path(path=plots_output_path)

        analyser = ResultAnalyser(sa_score_df=sa_score_df, ma_score_df=ma_score_df)
        analyser.analyse_and_compare_agent_systems(output_path=plots_output_path)

    def invoke_agentic_workflow_fn(
        self,
        query: str,
        system_type: Literal["single_agent", "multi_agent"],
    ):
        response = []
        llm_config = self.agents_config["llm_config"]

        if system_type == "single_agent":
            single_agent_crew = SingleAgentCrew(
                llm_config=llm_config, vectorstore=self.vectorstore
            )

            response = single_agent_crew.get_query_response(row={"question": query})

        elif system_type == "multi_agent":
            multi_agent_crew = MultiAgentCrew(
                llm_config=llm_config, vectorstore=self.vectorstore
            )

            response = multi_agent_crew.get_query_response(row={"question": query})

        else:
            raise NotImplementedError(
                "Only available system_type are [single_agent, multi_agent]"
            )

        return response


if __name__ == "__main__":
    # Step-1: Read the config file
    execution_config = read_yaml_file(
        file_path="/Users/I1598/Desktop/Personal/BITS-MTech/Dissertation/Midsem&Endsem/domain-rag/config/execution_conf.yaml"
    )

    # Step-2: Initialize the DomainSpecificRAG module
    domainrag = DomainSpecificRAG(config=execution_config)

    if execution_config["execution_flow"]["create_qa_data"]:
        # Step-3: Create QA Pairs dataset
        domainrag.create_qa_dataset_for_benchmarking(domain="clinical")

    if execution_config["execution_flow"]["create_vectorstore"]:
        domainrag.create_and_save_vectorstore(domain="clinical")

    if execution_config["execution_flow"]["load_vectorstore"]:
        vec_store = domainrag.load_vectorstore_in_memory()

        # Step-: Get the responses of the questions using agentic pipeline
        system_type = execution_config["agents_config"]["system_type"]
        domainrag.get_query_response_via_agents(
            vectorstore=vec_store, system_type=system_type
        )

    if execution_config["execution_flow"]["generate_scores"]:
        system_type = execution_config["evaluation_config"]["system_type"]
        domainrag.generate_agentic_rag_scores(system_type=system_type)

    if execution_config["execution_flow"]["plot_results"]:
        domainrag.compare_and_evaluate_agentic_rag()

    if execution_config["execution_flow"]["enable_gradio_app"]:
        domainrag.vectorstore = domainrag.load_vectorstore_in_memory()

        # Create the UI In Gradio
        iface = gr.Interface(
            fn=domainrag.invoke_agentic_workflow_fn,
            inputs=[
                gr.Textbox(value="Enter your query"),
                gr.Dropdown(
                    choices=["single_agent", "multi_agent"],
                    value="multi_agent",
                    label="Select an agent type",
                ),
            ],
            outputs=[
                gr.Textbox(label="Query Response"),
                gr.Number(label="Processing Time (seconds)"),
                gr.Number(label="Count of Output Tokens"),
                gr.Number(label="Count of API Requests"),
            ],
            title="Multi Agentic RAG Q&A Bot",
            examples=[
                [
                    "What are the benefits and harms associated with the use of fluoroquinolones in the treatment of MDR-TB?"
                ],
                [
                    "What criteria were used to select articles for the alemtuzumab systematic review in CLL patients, and what were the DSG recommendations for alemtuzumab use in CLL?"
                ],
                [
                    "What's the vaccination schedule for hepatitis B in Texas prisons and what are the risk factors for routine hepatitis C testing?",
                ],
                [
                    "What are the potential sources of rabies exposure for vets and clients in the US, and how can it be prevented?"
                ],
            ],
            theme=gr.themes.Soft(),
            allow_flagging="never",
        )

        iface.launch(share=False)  # put share equal to True for public URL
