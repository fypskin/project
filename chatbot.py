import os
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.translate.bleu_score import sentence_bleu

GROQ_API_KEY = "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" 
SEARCH_API_KEY = "AIzaSyCy9IcbiTsry740Pks0j7CyqJ9E1lW_56k" 
SEARCH_ENGINE_ID = "d67aa710f894f402d" 
PDF_FILES = ["book.pdf"]

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

def queryGroqWithBooks(query, retrievedChunks, chatHistory):
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
        "model": "llama3-8b-8192",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }

    response = requests.post(url, headers=headers, json=payload)
    result = response.json()

    return result["choices"][0]["message"]["content"] if "choices" in result and result["choices"] else "No response."

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

def decideBestAnswer(bookAnswer, webAnswer, query):
    bookSimilarity = sentence_bleu([query.split()], bookAnswer.split())
    webSimilarity = sentence_bleu([query.split()], webAnswer.split())

    print("\nBook-Based Answer:")
    print(bookAnswer)
    print(f"Book Similarity Score: {bookSimilarity:.4f}")

    print("\nWeb-Based Answer:")
    print(webAnswer)
    print(f"Web Similarity Score: {webSimilarity:.4f}")

    if bookSimilarity > webSimilarity:
        print("\nFinal Answer (Books Preferred):")
        return bookAnswer
    elif webSimilarity > bookSimilarity:
        print("\nFinal Answer (Web Preferred):")
        return webAnswer
    else:
        print("\nFinal Answer (Combined Response):")
        return f"Book Answer: {bookAnswer}\nWeb Answer: {webAnswer}"

def main():
    print("\nLoading and Processing PDFs...")
    extractedText = extractTextFromPdfs(PDF_FILES)
    textChunks = chunkText(extractedText)
    vectorStorage = createFaissIndex(textChunks)

    print("\nChatbot is ready! Type 'exit' to quit.")

    chatHistory = []

    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == "exit":
            print("Exiting chatbot. Have a great day!")
            break

        retrievedChunks = retrieveRelevantChunks(query, vectorStorage, k=3)
        bookAnswer = queryGroqWithBooks(query, retrievedChunks, chatHistory)

        webData = searchWeb(query)
        webAnswer = queryGroqWithWeb(query, webData)

        finalAnswer = decideBestAnswer(bookAnswer, webAnswer, query)

        chatHistory.append((query, finalAnswer))

        print(f"\n💡 Final Chatbot Answer: {finalAnswer}")

if __name__ == "__main__":
    main()
