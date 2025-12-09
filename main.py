from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

def main():
    print("Advanced RAG!")
    print(app.invoke(
        input = {
            "question": "What is agent memory?"
        }
    ))


if __name__ == "__main__":
    main()
