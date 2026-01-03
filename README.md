# 👋 Welcome to My RAG Experiments!

Hi there! Welcome to my playground for Retrieval-Augmented Generation (RAG). I built this repo to explore different ways to make RAG pipelines smarter, more reliable, and just plain better.

If you're looking to build your own RAG system or just curious about how things like *Agentic Chunking* or *Reciprocal Rank Fusion* work, you've come to the right place!

## 🚀 What's Inside?

This isn't just a standard "load and chat" repo. I've broken down the key components so you can play with them individually:

*   **Ingestion Pipeline**: A robust way to load your text documents and get them ready for search.
*   **Retrieval Pipeline**: The classic "ask a question, get an answer" flow.
*   **Advanced Stuff**:
    *   **Agentic Chunking**: Using an LLM to smartly split text based on *meaning* rather than just character counts.
    *   **Multi-Query Retrieval**: Why ask once when you can ask 3 times in different ways?
    *   **Reciprocal Rank Fusion (RRF)**: A fancy way to combine results from multiple searches to get the absolute best answers.

## 🛠️ Getting Started

First things first, let's get you set up. You'll need Python installed (I recommend 3.10+).

1.  **Clone the repo** (but you probably already did that).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up your environment**:
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY=sk-your-key-here
    ```

    > **Note:** If you want to use local embeddings (free!), you can also set `USE_LOCAL_EMBEDDINGS=true` in your `.env`.

## 🏃‍♀️ How to Run It

### 1. Load Your Documents
Before you can search, you need something to search *through*.
1.  Put your `.txt` files in the `docs/` folder.
2.  Run the ingestion script:
    ```bash
    python ingestion_pipeline.py
    ```
    This will chunk your text and save it to a local Vector Database (`db/chroma_db`).

### 2. Ask Questions
Once your data is loaded, you can run a simple retrieval:
```bash
python retrieval_pipeline.py
```
Check the code in `retrieval_pipeline.py` to change the query to whatever you want!

### 3. Try the "Cool Stuff"
Want to see how Multi-Query Retrieval works?
```bash
python multi_query_retrieval.py
```

Or how about merging results with Reciprocal Rank Fusion?
```bash
python reciprocal_rank_fusion.py
```

## 🧠 Why I Built This
I wanted to move beyond the basic tutorials and see how RAG handles real-world messiness. Things like:
*   "What if the user asks a vague question?" (solved with Multi-Query)
*   "What if splitting by 1000 characters cuts a paragraph in half?" (solved with Agentic Chunking)

Feel free to poke around, break things, and make them better. Happy coding! 💻✨
