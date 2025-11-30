好的！我幫你寫一個完整的執行計劃給 Claude Code，並且**全部使用免費資源**！

## 📋 給 Claude Code 的專案執行計劃

```markdown
# 專案：免費 Google Books RAG 推薦系統

## 專案目標
建立一個書籍推薦系統，使用 RAG (Retrieval-Augmented Generation) 技術，
完全使用免費資源，包括：
- Google Books API（免費）
- HuggingFace Inference API（免費）
- 本地向量資料庫（FAISS，免費）
- Streamlit（免費部署）

---

## 技術棧（全免費）

### LLM 選擇
1. **HuggingFace Inference API**（推薦）
   - 模型：`google/flan-t5-xxl` 或 `mistralai/Mistral-7B-Instruct-v0.2`
   - 免費額度：有 rate limit 但足夠使用
   - 支援繁體中文

2. **Google Gemini API**（備選）
   - 模型：`gemini-pro`
   - 免費額度：每分鐘 60 requests
   - 中文支援好

### Embedding 模型
- HuggingFace `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- 本地執行，完全免費
- 支援中文

### 向量資料庫
- FAISS（Facebook AI Similarity Search）
- 本地儲存，不需要雲端服務
- 輕量且快速

---

## 專案結構

```
books-rag-system/
├── data_collection.py       # 步驟1：收集 Google Books 資料
├── build_vectordb.py        # 步驟2：建立向量資料庫
├── app.py                   # 步驟3：Streamlit 應用程式
├── requirements.txt         # Python 套件
├── .gitignore              
├── README.md               
├── data/
│   └── books_raw.json      # 原始書籍資料
└── vectordb/
    └── faiss_index/        # FAISS 向量資料庫
```

---

## 詳細實作步驟

### 步驟 0：環境設定

**requirements.txt**
```
streamlit==1.29.0
langchain==0.1.0
langchain-community==0.0.10
sentence-transformers==2.2.2
faiss-cpu==1.7.4
requests==2.31.0
google-generativeai==0.3.2
huggingface-hub==0.20.0
python-dotenv==1.0.0
```

**環境變數設定（.env）**
```
# 選擇一個即可

# 方案 A：HuggingFace（推薦）
HUGGINGFACE_API_KEY=hf_your_key_here

# 方案 B：Google Gemini
GOOGLE_API_KEY=your_google_api_key
```

**如何取得免費 API Key：**

1. **HuggingFace Token（推薦）**
   - 註冊：https://huggingface.co/join
   - 前往：https://huggingface.co/settings/tokens
   - 建立 "Read" token（免費）
   - 免費額度充足

2. **Google Gemini API**
   - 前往：https://makersuite.google.com/app/apikey
   - 建立免費 API Key
   - 每月 60 requests/分鐘免費

---

### 步驟 1：資料收集腳本

**檔案：data_collection.py**

```python
"""
Google Books 資料收集腳本
功能：從 Google Books API 收集書籍資料
"""

import requests
import json
import time
from pathlib import Path

def search_books(query, max_results=40, language='zh-TW'):
    """
    從 Google Books API 搜尋書籍
    
    Args:
        query: 搜尋關鍵字
        max_results: 最多結果數
        language: 語言限制
    
    Returns:
        書籍列表
    """
    all_books = []
    
    for start_index in range(0, max_results, 40):
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {
            'q': query,
            'langRestrict': language,
            'maxResults': min(40, max_results - start_index),
            'startIndex': start_index,
            'printType': 'books'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            all_books.extend(data.get('items', []))
            print(f"已收集 {len(all_books)} 本書...")
            time.sleep(1)  # 避免 rate limit
        except Exception as e:
            print(f"錯誤：{e}")
            continue
    
    return all_books

def extract_book_info(book):
    """提取書籍重要資訊"""
    volume_info = book.get('volumeInfo', {})
    
    return {
        'id': book.get('id'),
        'title': volume_info.get('title', '無標題'),
        'authors': volume_info.get('authors', ['未知作者']),
        'publisher': volume_info.get('publisher', '未知出版社'),
        'published_date': volume_info.get('publishedDate', '未知'),
        'description': volume_info.get('description', '無描述'),
        'categories': volume_info.get('categories', ['未分類']),
        'page_count': volume_info.get('pageCount', 0),
        'language': volume_info.get('language', 'zh'),
        'preview_link': volume_info.get('previewLink', ''),
        'thumbnail': volume_info.get('imageLinks', {}).get('thumbnail', ''),
    }

def collect_books_data(categories=None, books_per_category=30):
    """
    收集多個類別的書籍
    
    Args:
        categories: 類別列表
        books_per_category: 每個類別收集數量
    """
    if categories is None:
        categories = [
            '小說',
            '科幻',
            '推理',
            '愛情',
            '歷史',
            '科普',
            '商業',
            '自我成長',
            '哲學',
            '心理學'
        ]
    
    all_books = []
    
    print(f"📚 開始收集書籍資料...")
    print(f"類別數量：{len(categories)}")
    print(f"每類別：{books_per_category} 本")
    print("-" * 50)
    
    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] 收集「{category}」類別...")
        books = search_books(category, max_results=books_per_category)
        processed_books = [extract_book_info(book) for book in books]
        all_books.extend(processed_books)
        time.sleep(1)
    
    # 去重（根據 ID）
    unique_books = {book['id']: book for book in all_books}.values()
    unique_books = list(unique_books)
    
    # 儲存資料
    Path('data').mkdir(exist_ok=True)
    output_file = 'data/books_raw.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unique_books, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 50)
    print(f"✅ 收集完成！")
    print(f"📊 總共收集：{len(unique_books)} 本書籍")
    print(f"💾 儲存位置：{output_file}")
    print("=" * 50)
    
    return unique_books

if __name__ == "__main__":
    books = collect_books_data()
```

---

### 步驟 2：建立向量資料庫

**檔案：build_vectordb.py**

```python
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
    print("使用模型：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("（本地執行，完全免費，支援中文）")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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
```

---

### 步驟 3：Streamlit 應用程式

**檔案：app.py**

```python
"""
Google Books RAG 推薦系統
使用免費資源：FAISS + HuggingFace/Gemini
"""

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== 配置選項 ====================

# 選擇使用哪個 LLM（二選一）
USE_LLM = "huggingface"  # 或 "gemini"

# ==================== LLM 設定 ====================

def get_llm():
    """根據配置返回對應的 LLM"""
    
    if USE_LLM == "huggingface":
        from langchain_community.llms import HuggingFaceHub
        
        api_key = os.getenv("HUGGINGFACE_API_KEY") or st.secrets.get("HUGGINGFACE_API_KEY")
        
        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # 或 "google/flan-t5-xxl"
            huggingfacehub_api_token=api_key,
            model_kwargs={
                "temperature": 0.7,
                "max_length": 512
            }
        )
        return llm
    
    elif USE_LLM == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7
        )
        return llm

# ==================== 快取函數 ====================

@st.cache_resource
def load_vectordb():
    """載入向量資料庫（只載入一次）"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectordb = FAISS.load_local(
        "vectordb/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vectordb

@st.cache_resource
def create_qa_chain(_vectordb):
    """建立問答鏈"""
    
    template = """你是一個專業的書籍推薦助手。根據以下書籍資訊回答問題。

相關書籍資訊：
{context}

問題：{question}

請用繁體中文回答，並且：
1. 推薦 2-3 本最相關的書籍
2. 說明推薦理由
3. 提供書名、作者和簡短介紹

回答："""

    PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    llm = get_llm()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=_vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain

# ==================== Streamlit UI ====================

def main():
    st.set_page_config(
        page_title="📚 AI 書籍推薦助手",
        page_icon="📚",
        layout="wide"
    )
    
    # 標題
    st.title("📚 AI 書籍推薦助手")
    st.markdown("基於 Google Books 資料，使用 RAG 技術推薦好書 | 🆓 完全免費")
    
    # 顯示使用的技術
    st.caption(f"💡 使用技術：FAISS + {USE_LLM.upper()} | 本地 Embeddings")
    
    # 載入資源
    try:
        with st.spinner("載入書籍資料庫..."):
            vectordb = load_vectordb()
            qa_chain = create_qa_chain(vectordb)
        st.success("✅ 資料庫載入完成！")
    except Exception as e:
        st.error(f"❌ 載入失敗：{e}")
        st.info("請確認：\n1. 已執行 build_vectordb.py\n2. API Key 已設定")
        return
    
    # 側邊欄
    with st.sidebar:
        st.header("💡 使用說明")
        st.markdown("""
        輸入您的問題，AI 會推薦相關書籍
        
        **範例問題：**
        - 推薦科幻小說
        - 有什麼商業書籍？
        - 適合初學者的心理學書
        - 推薦經典文學作品
        """)
        
        st.divider()
        
        st.header("⚙️ 系統資訊")
        st.write(f"**LLM**：{USE_LLM}")
        st.write(f"**向量資料庫**：FAISS")
        st.write(f"**Embeddings**：本地模型")
        
        # 顯示統計
        if vectordb:
            st.metric("資料庫文檔數", vectordb.index.ntotal)
    
    # 主要區域
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 提出問題")
        question = st.text_input(
            "請輸入您的問題：",
            placeholder="例如：推薦適合新手的科幻小說",
            label_visibility="collapsed"
        )
    
    with col2:
        st.write("")  # 空白對齊
        submit = st.button("🚀 獲取推薦", type="primary", use_container_width=True)
    
    # 範例問題按鈕
    st.markdown("**快速範例：**")
    col1, col2, col3, col4 = st.columns(4)
    
    example_questions = {
        "科幻小說": "推薦經典科幻小說",
        "商業理財": "有什麼商業或理財書？",
        "心理學": "推薦心理學相關書籍",
        "歷史": "有什麼歷史類的好書？"
    }
    
    for col, (label, q) in zip([col1, col2, col3, col4], example_questions.items()):
        with col:
            if st.button(label, use_container_width=True):
                question = q
                submit = True
    
    # 處理問答
    if submit and question:
        with st.spinner("🤔 思考中..."):
            try:
                result = qa_chain({"query": question})
                
                # 顯示答案
                st.markdown("### 💬 推薦結果")
                st.write(result['result'])
                
                # 顯示參考書籍
                st.markdown("### 📚 參考書籍")
                
                for i, doc in enumerate(result['source_documents'][:3], 1):
                    with st.expander(f"📖 書籍 {i}：{doc.metadata.get('title', '未知')}"):
                        col_a, col_b = st.columns([1, 3])
                        
                        with col_a:
                            if doc.metadata.get('thumbnail'):
                                st.image(doc.metadata['thumbnail'], width=120)
                        
                        with col_b:
                            st.write(f"**作者**：{doc.metadata.get('authors', '未知')}")
                            st.write(f"**類別**：{doc.metadata.get('categories', '未知')}")
                            if doc.metadata.get('preview_link'):
                                st.markdown(f"[📱 預覽連結]({doc.metadata['preview_link']})")
                
            except Exception as e:
                st.error(f"發生錯誤：{e}")
                st.info("請檢查 API Key 是否正確設定")
    
    elif submit and not question:
        st.warning("⚠️ 請輸入問題！")

if __name__ == "__main__":
    main()
```

---

## 執行順序

### 本地測試

```bash
# 1. 安裝套件
pip install -r requirements.txt

# 2. 設定 API Key（擇一）
# 建立 .env 檔案，加入：
# HUGGINGFACE_API_KEY=hf_xxxx
# 或
# GOOGLE_API_KEY=xxxx

# 3. 收集資料
python data_collection.py

# 4. 建立向量資料庫（只需執行一次，約需 5-10 分鐘）
python build_vectordb.py

# 5. 啟動應用程式
streamlit run app.py
```

### 部署到 Streamlit Cloud

1. 將專案推到 GitHub（不要包含 .env）
2. 在 Streamlit Cloud 建立新應用
3. 在 Settings → Secrets 加入 API Key：
```toml
HUGGINGFACE_API_KEY = "hf_your_key"
# 或
GOOGLE_API_KEY = "your_key"
```

---

## 注意事項

1. **第一次執行 build_vectordb.py 會下載模型**
   - 約 400MB
   - 只需下載一次
   - 之後都是本地執行

2. **HuggingFace 免費限制**
   - 每小時約 1000 次請求
   - 足夠個人使用或 Demo

3. **檔案大小**
   - vectordb/ 資料夾約 100-200MB
   - 可以推到 GitHub
   - 或在部署時重新建立

4. **效能**
   - 載入資料庫：3-5 秒
   - 單次問答：5-10 秒
   - Streamlit Cloud 不會 timeout

---

## 進階功能建議

1. **加入書籍封面顯示**
2. **匯出推薦清單為 PDF**
3. **加入使用者評分功能**
4. **多語言支援（中英切換）**
5. **顯示相似書籍推薦**

---

## 疑難排解

### Q: HuggingFace 模型下載太慢？
A: 可以先在本地下載，再部署

### Q: Streamlit Cloud 上傳失敗？
A: vectordb 太大，可以在 GitHub Actions 自動建立

### Q: 回答品質不佳？
A: 可以調整 Prompt 或換用 Gemini

---

請按照這個計劃執行，有任何問題隨時回報！
```

---

## 📝 額外補充：如何取得免費 API Key

### HuggingFace Token（推薦！）

1. 前往 https://huggingface.co/join 註冊
2. 登入後到 https://huggingface.co/settings/tokens
3. 點選 "New token"
4. Token 類型選 "Read"
5. 複製 Token（格式：`hf_xxxxxx`）

**優點：**
- 完全免費
- 額度充足
- 支援多種模型

### Google Gemini API

1. 前往 https://makersuite.google.com/app/apikey
2. 登入 Google 帳號
3. 點選 "Create API Key"
4. 複製 API Key

**優點：**
- 每分鐘 60 requests
- 中文表現好
- 回應速度快

---

這樣你就有一個**完全免費**的 RAG 系統了！需要我解釋哪個部分嗎？