import os
import glob
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pypandoc
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import markdown

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 環境変数の読み込み
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPIアプリ
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 入力データのモデル
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

# ドキュメント読み込み
def load_documents(directory="documents"):
    texts = []
    try:
        for file in glob.glob(f"{directory}/*.md") + glob.glob(f"{directory}/*.txt")+ glob.glob(f"{directory}/*.adoc"):
            logger.info(f"読み込み中: {file}")
            if file.endswith(".md"):
                text = pypandoc.convert_file(file, "plain")
            else:
                with open(file, "r", encoding="utf-8") as f:
                    text = f.read()
            chunks = [text[i:i+500] for i in range(0, len(text), 500)]
            texts.extend(chunks)
        if not texts:
            logger.warning("ドキュメントが見つかりません")
        return texts
    except Exception as e:
        logger.error(f"ドキュメント読み込みエラー: {str(e)}")
        raise

# FAISSインデックス構築
def build_index(texts):
    try:
        #model = SentenceTransformer("all-MiniLM-L6-v2")
        #model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
        model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")    
        embeddings = model.encode(texts, show_progress_bar=True)
        dimension = embeddings.shape[1]
        #index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.add(embeddings)
        logger.info("FAISSインデックス構築完了")
        return index, model, texts
    except Exception as e:
        logger.error(f"インデックス構築エラー: {str(e)}")
        raise

# 初期化
try:
    documents = load_documents()
    index, model, texts = build_index(documents)
except Exception as e:
    logger.error(f"初期化エラー: {str(e)}")
    raise

# RAG推論
def rag_query(query: str, top_k: int = 3):
    try:
        logger.info(f"クエリ処理: {query}")
        query_embedding = model.encode([query])[0]
        distances, indices = index.search(np.array([query_embedding]), top_k)
        context = "\n".join([texts[i] for i in indices[0]])
        logger.info(f"取得コンテキスト: {context[:100]}...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the following context to answer the query:\n" + context},
                {"role": "user", "content": query}
            ],
            max_tokens=500
        )
        md_content = response.choices[0].message.content
        # Markdown→HTML変換
        html_content = markdown.markdown(md_content, extensions=['fenced_code', 'tables'])
        return html_content
    except Exception as e:
        logger.error(f"RAG推論エラー: {str(e)}")
        raise

# UIエンドポイント
@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# APIエンドポイント
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        answer_html = rag_query(request.query, top_k=5)
        return {"query": request.query, "answer": answer_html}
    except Exception as e:
        logger.error(f"APIエラー: {str(e)}")
        raise HTTPException(status_code=750, detail=str(e))

# サーバー起動
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)