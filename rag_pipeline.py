import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Loading the environment variables
load_dotenv()
# Initializing Cerebras LLM via LangChain
llm = ChatCerebras(
    model="gpt-oss-120b",
    api_key=os.getenv("CEREBRAS_API_KEY"),
    temperature=0.3,
    max_tokens=1024,
)

# Initializing Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_web(query: str) -> str:
    """Search the web using Tavily and format results as context."""
    response = tavily_client.search(query=query, max_results=5)

    if not response["results"]:
        return "No search results found."

    context_parts = []
    for i, result in enumerate(response["results"], 1):
        context_parts.append(
            f"[Source {i}] {result['title']}\n"
            f"URL: {result['url']}\n"
            f"Content: {result['content']}\n"
        )

    return "\n".join(context_parts)

def classify_question(question: str) -> str:
    """Classify whether a question needs web search or can be answered directly."""
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a query classifier. Determine if the user's question "
            "requires fresh web data or can be answered from general knowledge.\n\n"
            "Respond with exactly one word:\n"
            "- 'web_search' if the question asks about recent events, current "
            "prices, latest news, live data, or anything time-sensitive\n"
            "- 'direct_answer' if the question is about general knowledge, "
            "definitions, concepts, or well-established facts"
        )),
        ("human", "Question: {question}"),
    ])
    
    classify_chain = classify_prompt | llm | StrOutputParser()
    result = classify_chain.invoke({"question": question}).strip().lower()
    
    if "web_search" in result:
        return "web_search"
    return "direct_answer"
# Defineing the RAG prompt template
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful research assistant. Answer the user's question "
        "based on the provided web search results. Always cite your sources "
        "using [Source N] notation. If the search results don't contain "
        "relevant information, say so clearly."
    )),
    ("human", (
        "Web Search Results:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a comprehensive answer with citations:"
    )),
]) 

# Define the direct answer prompt
direct_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant. Answer the user's question clearly "
        "and concisely from your general knowledge."
    )),
    ("human", "{question}"),
])


# Build chains
output_parser = StrOutputParser()
rag_chain = rag_prompt | llm | output_parser
direct_chain = direct_prompt | llm | output_parser

def ask(question: str) -> str:
    """Run the RAG pipeline with intelligent query routing."""
    route = classify_question(question)
    print(f"\nRoute: {route}")
    
    if route == "web_search":
        print(f"Searching the web for: {question}")
        context = search_web(question)
        
        print("Generating answer...\n")
        answer = rag_chain.invoke({
            "context": context,
            "question": question,
        })
    else:
        print("Answering from general knowledge...\n")
        answer = direct_chain.invoke({"question": question})
    
    return answer

def main():
    """Interactive loop for asking questions."""
    print("=" * 60)
    print("Real-Time RAG Pipeline with Query Routing")
    print("Ask any question and get a cited answer from the live web!")
    print("Type 'quit' to exit.")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = ask(question)
            print(f"\nAnswer:\n{answer}")
        except Exception as e:
            print(f"\nError: {e}")
            print("Please check your API keys and try again.")


if __name__ == "__main__":
    main()
