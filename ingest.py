import logging
from haystack.nodes import EmbeddingRetriever, PreProcessor, AnswerParser
from haystack.utils import convert_files_to_docs
from haystack.pipelines import Pipeline
from haystack.document_stores import WeaviateDocumentStore
print("Imports done")

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
print("Logging done")

all_docs = convert_files_to_docs(dir_path="data/")

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True
)

pre_processed_docs = preprocessor.process(all_docs)
