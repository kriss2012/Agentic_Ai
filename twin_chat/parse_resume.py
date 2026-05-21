from langchain_community.document_loaders import PyPDFLoader
import sys

def parse_pdf(file_path, output_path):
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        with open(output_path, "w", encoding="utf-8") as f:
            for page in pages:
                f.write(page.page_content)
                f.write("\n\n")
        print("Success")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parse_pdf(sys.argv[1], sys.argv[2])
