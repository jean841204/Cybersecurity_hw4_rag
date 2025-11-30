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

# 選擇使用哪個 LLM
USE_LLM = "groq"  # groq (推薦)

# ==================== LLM 設定 ====================

def get_llm():
    """根據配置返回對應的 LLM"""

    if USE_LLM == "groq":
        from langchain_groq import ChatGroq

        # 優先從環境變數讀取，失敗則嘗試 streamlit secrets
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                api_key = None

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=api_key,
            temperature=0.7,
            max_tokens=512
        )
        return llm

# ==================== 快取函數 ====================

@st.cache_resource
def load_vectordb():
    """載入向量資料庫（只載入一次）"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectordb = FAISS.load_local(
        "vectordb/faiss_index",
        embeddings,
        allow_dangerous_deserialization=True  # 安全：我們自己建立的資料庫
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
        retriever=_vectordb.as_retriever(search_kwargs={"k": 10}),  # 增加到 10 本
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
        st.write(f"**LLM**：{USE_LLM.upper()}")
        if USE_LLM == "groq":
            st.write(f"**模型**：llama-3.3-70b-versatile")

        st.write(f"**向量資料庫**：FAISS v1.7.4")
        st.write(f"**Embedding 模型**：")
        st.write(f"- all-MiniLM-L6-v2")
        st.write(f"- sentence-transformers v2.2.2")
        st.write(f"- 本地運行 (CPU)")

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

                for i, doc in enumerate(result['source_documents'][:6], 1):  # 顯示前6本
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
