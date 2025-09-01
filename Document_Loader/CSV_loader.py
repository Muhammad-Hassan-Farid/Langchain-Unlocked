from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='./Document_Loader/data/pharmaceutical_word_reconstruction_dataset.csv', encoding='utf-8')

doc = loader.load()

print(doc)