import logging
from haystack.nodes import PreProcessor, PDFToTextConverter

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

