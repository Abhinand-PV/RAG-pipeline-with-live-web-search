import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# Loading the environment variables
load_dotenv()


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = (os.getenv(name) or "").strip()
    if not value or value.startswith("your_"):
        raise RuntimeError(
            f"Missing or placeholder value for {name}. "
            "Copy .env.example to .env and set your API keys."
        )
    return value


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
    try:
        response = tavily_client.search(query=query, max_results=5)
    except Exception as e:
        return f"Error performing web search: {e}"

    results = response.get("results") or []
    if not results:
        return "No search results found."

    context_parts = []
    for i, result in enumerate(results, 1):
        title = result.get("title") or "Untitled"
        url = result.get("url") or "N/A"
        content = result.get("content") or "No snippet available."
        context_parts.append(
            f"[Source {i}] {title}\n"
            f"URL: {url}\n"
            f"Content: {content}\n"
        )

    return "\n".join(context_parts)


def classify_question(question: str, chat_history: list) -> str:
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
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {question}"),
    ])

    classify_chain = classify_prompt | llm | StrOutputParser()
    result = classify_chain.invoke({
        "question": question,
        "chat_history": chat_history
    }).strip().lower()

    if "web_search" in result:
        return "web_search"
    return "direct_answer"


def rewrite_search_query(question: str, chat_history: list) -> str:
    """Turn follow-up questions into a standalone web search query."""
    if not chat_history:
        return question

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Rewrite the user's latest question as a standalone web search query. "
            "Resolve pronouns and references using chat history. "
            "Return only the search query, with no quotes or extra text."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = rewrite_chain.invoke({
        "question": question,
        "chat_history": chat_history,
    }).strip()
    return rewritten or question

# Defining the RAG prompt template
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful research assistant. Answer the user's question "
        "based on the provided web search results. Always cite your sources "
        "using [Source N] notation. If the search results don't contain "
        "relevant information, say so clearly."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
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
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


# Build chains
output_parser = StrOutputParser()
rag_chain = rag_prompt | llm | output_parser
direct_chain = direct_prompt | llm | output_parser

def ask(question: str, chat_history: list) -> str:
    """Run the RAG pipeline with intelligent query routing and history."""
    route = classify_question(question, chat_history)
    print(f"\nRoute: {route}")

    if route == "web_search":
        search_query = rewrite_search_query(question, chat_history)
        print(f"Searching the web for: {search_query}")
        context = search_web(search_query)

        print("Generating answer...\n")
        answer = rag_chain.invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history,
        })
    else:
        print("Answering from general knowledge...\n")
        answer = direct_chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })

    return answer

def main():
    """Interactive loop for asking questions."""
    try:
        require_env("CEREBRAS_API_KEY")
        require_env("TAVILY_API_KEY")
    except RuntimeError as e:
        print(f"\n{e}")
        return

    print("=" * 60)
    print("Real-Time RAG Pipeline with Query Routing (with Conversational Memory)")
    print("Ask any question and get a cited answer from the live web!")
    print("Type 'quit' to exit, or 'clear' to reset conversation memory.")
    print("=" * 60)

    chat_history = []

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if question.lower() == "clear":
            chat_history = []
            print("Conversation memory cleared.")
            continue

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = ask(question, chat_history)
            print(f"\nAnswer:\n{answer}")

            # Maintain sliding window of conversation history (keep last 5 turns / 10 messages)
            chat_history.append(HumanMessage(content=question))
            chat_history.append(AIMessage(content=answer))
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
        except Exception as e:
            print(f"\nError: {e}")
            print("Please check your API keys and try again.")


if __name__ == "__main__":
    main()
