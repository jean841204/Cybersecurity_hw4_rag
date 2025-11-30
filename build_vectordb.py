"""
向量資料庫建立腳本
使用 FAISS + HuggingFace Embeddings（完全免費）
"""

import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def load_books_data():
    """載入書籍資料"""
    with open('data/books_raw.json', 'r', encoding='utf-8') as f:
        books = json.load(f)
    print(f"📚 載入 {len(books)} 本書籍")
    return books

def prepare_documents(books):
    """準備文檔格式"""
    documents = []

    for book in books:
        # 組合成完整文字
        text = f"""書名：{book['title']}
作者：{', '.join(book['authors'])}
出版社：{book['publisher']}
出版日期：{book['published_date']}
類別：{', '.join(book['categories'])}
頁數：{book['page_count']}

簡介：
{book['description']}
"""

        # 建立 Document 物件
        doc = Document(
            page_content=text,
            metadata={
                'title': book['title'],
                'authors': ', '.join(book['authors']),
                'categories': ', '.join(book['categories']),
                'id': book['id'],
                'preview_link': book['preview_link'],
                'thumbnail': book['thumbnail']
            }
        )
        documents.append(doc)

    return documents

def build_vectordb():
    """建立 FAISS 向量資料庫"""

    print("=" * 60)
    print("🚀 開始建立向量資料庫")
    print("=" * 60)

    # 1. 載入資料
    print("\n[1/4] 📚 載入書籍資料...")
    books = load_books_data()

    # 2. 準備文檔
    print("\n[2/4] 📝 準備文檔...")
    documents = prepare_documents(books)
    print(f"準備了 {len(documents)} 個文檔")

    # 3. 文字切割
    print("\n[3/4] ✂️  切割文字...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"切割成 {len(splits)} 個片段")

    # 4. 建立 Embeddings（本地執行，免費）
    print("\n[4/4] 🔄 建立向量資料庫...")
    print("使用模型：sentence-transformers/all-MiniLM-L6-v2")
    print("（本地執行，完全免費，輕量快速）")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 建立 FAISS 向量資料庫
    vectordb = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

    # 5. 儲存
    print("\n💾 儲存向量資料庫...")
    save_path = "vectordb/faiss_index"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(save_path)

    print("\n" + "=" * 60)
    print("✅ 向量資料庫建立完成！")
    print(f"📁 儲存位置：{save_path}")
    print(f"📊 文檔數量：{len(splits)}")
    print("=" * 60)

    return vectordb

if __name__ == "__main__":
    build_vectordb()
