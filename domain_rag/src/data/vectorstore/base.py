import os
from time import sleep
from typing import List

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm


class CustomVectorStore:
    def __init__(self) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=500,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
        )

    def create_vectorstore(
        self,
        documents: List[str],
        embedder_llm,
        vectorstore_intermediate_location_path: str,
        batch_size: int = 100,
    ):
        # Step-1: Defining the variables used to store and iterate
        # vectorstore
        vectorstore = []
        max_idx = int(np.ceil(len(documents) / batch_size) * batch_size)

        # Step-2a: Create embeddings vectorstores based on the defined
        # batch_size
        for index in tqdm(range(0, max_idx, batch_size)):

            counter = int(np.ceil(index / batch_size))
            # Step-: Check if index is already stored
            index_path = vectorstore_intermediate_location_path + f"{counter}/"

            if os.path.exists(index_path):
                pass
            else:
                chunked_documents = self.text_splitter.create_documents(
                    texts=documents[index : index + batch_size]
                )

                vs = FAISS.from_texts(
                    texts=[doc.page_content for doc in chunked_documents],
                    embedding=embedder_llm,
                )

                self.save_vectorstore(
                    vectorstore=vs,
                    save_location_path=vectorstore_intermediate_location_path
                    + f"{counter}/",
                )

                if counter % 10 == 0:
                    # Step-2b: Sleep for 60 secs after every 10 batches
                    sleep(60)
                else:
                    # Step-2b: Sleep for 20 secs for each batch
                    sleep(20)

        for index in tqdm(range(0, max_idx, batch_size)):
            index_path = (
                vectorstore_intermediate_location_path
                + f"{int(np.ceil(index/batch_size))}/"
            )

            vs = self.load_vectorstore(
                embedder_llm=embedder_llm, vectorstore_location_path=index_path
            )

            vectorstore.append(vs)
            logger.info(
                f"Succefully appended {int(np.ceil(index/batch_size))} vectorstore"
            )

        # Step-3: Merge all vectorstores
        base_vs = vectorstore[0]
        for vs in vectorstore[1:]:
            base_vs.merge_from(vs)

        logger.success(f"Succefully merged all the chucked vectorstore")

        # Step-4: Return the merged vectorstore
        return base_vs

    def load_vectorstore(
        self,
        embedder_llm,
        vectorstore_location_path: str,
        allow_dangerous_deserialization: bool = True,
    ):
        try:
            vectorstore = FAISS.load_local(
                folder_path=vectorstore_location_path,
                embeddings=embedder_llm,
                allow_dangerous_deserialization=allow_dangerous_deserialization,
            )
        except Exception as error:
            logger.error(f"Unable to load vectorstore due to error: {error}")
            raise Exception

        return vectorstore

    def save_vectorstore(self, vectorstore, save_location_path: str):
        try:
            vectorstore.save_local(save_location_path)
        except Exception as error:
            logger.error(f"Unable to save vectorstore due to error: {error}")
            raise Exception

    def merge_vectorstore(sel, vectorstores: List):
        # Step-1: Get the base vectorstore
        base_vs = vectorstores[0]

        # Step-2: Start merging the all the remaining
        # vectorstore into base vectorstore
        for index in range(len(vectorstores) - 1):
            base_vs.merge_from(vectorstores[index + 1])

        return base_vs
