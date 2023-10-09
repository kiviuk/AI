import logging
from haystack.nodes import PreProcessor, PDFToTextConverter, EmbeddingRetriever
from haystack.document_stores import WeaviateDocumentStore

print("Imports done")

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
print("Logging done")

converter = PDFToTextConverter(remove_numeric_tables=True, multiprocessing=False)
all_docs = converter.convert(file_path="data/DatabricksAcademyCourseCatalog.pdf")

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=False,
    clean_header_footer=True,
    split_by="word",
    split_length=100,
    split_respect_sentence_boundary=True
)

pre_processed_docs = preprocessor.process(all_docs)
print(pre_processed_docs)
print("Preprocessing done")

document_store = WeaviateDocumentStore(host='http://localhost',
                                       port=8080,
                                       embedding_dim=384)
print("Document store done")

document_store.write_documents(pre_processed_docs)

retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2")

print("Retriever: ", retriever)

document_store.update_embeddings(retriever)

print("Embeddings Done.")
