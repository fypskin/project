import os
import time
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from tabulate import tabulate
import nltk
nltk.download("punkt")

GROQ_API_KEY = "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"
SEARCH_API_KEY = "AIzaSyCy9IcbiTsry740Pks0j7CyqJ9E1lW_56k"
SEARCH_ENGINE_ID = "d67aa710f894f402d"
PDF_FILES = ["book.pdf", "book2.pdf", "book3.pdf"]

MODELS = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "mistral-7b",
    "llama2-70b-4096"
]

def extractTextFromPdfs(pdfFiles):
    rawText = ""
    for pdfPath in pdfFiles:
        if os.path.exists(pdfPath):
            print(f"Processing: {pdfPath}")
            pdfReader = PdfReader(pdfPath)
            for page in pdfReader.pages:
                text = page.extract_text()
                if text:
                    rawText += text + "\n"
        else:
            print(f"Warning: {pdfPath} not found.")
    return rawText

def chunkText(text, chunkSize=1000, chunkOverlap=200):
    textSplitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunkSize,
        chunk_overlap=chunkOverlap,
        length_function=len
    )
    return textSplitter.split_text(text)

def createFaissIndex(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)

def retrieveRelevantChunks(query, vectorStorage, k=3):
    retrievedChunks = vectorStorage.similarity_search(query, k=k)
    return [chunk.page_content for chunk in retrievedChunks]

def queryGroqWithBooksMultiModel(query, retrievedChunks, chatHistory, modelName):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    context = "\n".join(retrievedChunks)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Use the provided context to answer queries."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]

    for q, a in chatHistory:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    payload = {
        "model": modelName,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }

    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload)
    elapsed_time = time.time() - start_time

    result = response.json()

    if "choices" in result and result["choices"]:
        content = result["choices"][0]["message"]["content"]
    else:
        content = "No response."

    bleu = sentence_bleu([query.split()], content.split())

   
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(query, content)
    rouge_l = rouge_scores['rougeL'].fmeasure

    return {
        "model": modelName,
        "response": content,
        "response_time": elapsed_time,
        "bleu_score": bleu,
        "rouge_l": rouge_l
    }

def runModelComparisons(query, retrievedChunks, chatHistory):
    results = []

    print("\n🔍 Running model comparisons...\n")

    for model in MODELS:
        print(f"Querying model: {model}")
        result = queryGroqWithBooksMultiModel(query, retrievedChunks, chatHistory, model)
        results.append(result)

    headers = ["Model", "BLEU Score", "ROUGE-L", "Response Time (s)", "Response"]
    table = [
        [
            res["model"],
            f"{res['bleu_score']:.4f}",
            f"{res['rouge_l']:.4f}",
            f"{res['response_time']:.2f}",
            res["response"][:100] + "…" if len(res["response"]) > 100 else res["response"]
        ]
        for res in results
    ]

    print("\n📊 Comparison Table:")
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))

    best_model = max(results, key=lambda x: x["bleu_score"])
    print(f"\n🏆 Best Model Based on BLEU Score: {best_model['model']}")
    return best_model["response"]

def searchWeb(query):
    searchUrl = f"https://www.googleapis.com/customsearch/v1?q={query}&key={SEARCH_API_KEY}&cx={SEARCH_ENGINE_ID}"
    response = requests.get(searchUrl)
    results = response.json()

    if "items" in results:
        webSnippets = [item["snippet"] for item in results["items"][:3]]
        return " ".join(webSnippets)
    else:
        return "No relevant web results found."

def queryGroqWithWeb(query, webData):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Use the provided web data to answer queries."},
        {"role": "user", "content": f"Web Data: {webData}\n\nQuestion: {query}"}
    ]

    payload = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    return result["choices"][0]["message"]["content"] if "choices" in result and result["choices"] else "No response."

def main():
    print("\n📚 Loading and Processing PDFs...")
    extractedText = extractTextFromPdfs(PDF_FILES)
    textChunks = chunkText(extractedText)
    vectorStorage = createFaissIndex(textChunks)

    print("\n🤖 Chatbot is ready! Type 'exit' to quit.")

    chatHistory = []

    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == "exit":
            print("👋 Exiting chatbot. Have a great day!")
            break

        retrievedChunks = retrieveRelevantChunks(query, vectorStorage, k=3)
        bookAnswer = runModelComparisons(query, retrievedChunks, chatHistory)

        webData = searchWeb(query)
        webAnswer = queryGroqWithWeb(query, webData)

        print("\n🌐 Web Answer:")
        print(webAnswer)

        chatHistory.append((query, bookAnswer))

        print(f"\n💡 Final Book-Based Answer: {bookAnswer}")

if __name__ == "__main__":
    main()
