# 📚 Google Books RAG 推薦系統

## 🔗 Live Demo
**👉 [https://cybersecurityhw4rag-jean.streamlit.app/](https://cybersecurityhw4rag-jean.streamlit.app/)**

基於 RAG (Retrieval-Augmented Generation) 技術的智慧書籍推薦系統，使用完全免費的資源建構。

## ✨ 功能特色

- 🔍 智慧書籍搜尋與推薦
- 🤖 使用 RAG 技術提供精準推薦
- 💰 完全免費的技術棧
- 🌏 支援繁體中文
- 🚀 可免費部署至 Streamlit Cloud

## 🛠️ 技術架構

| 組件 | 技術選擇 | 說明 |
|------|---------|------|
| **LLM** | HuggingFace / Google Gemini | 免費 API |
| **Embedding** | sentence-transformers | 本地執行 |
| **向量資料庫** | FAISS | 本地儲存 |
| **資料來源** | Google Books API | 免費 |
| **前端** | Streamlit | 免費部署 |

## 📂 專案結構

```
books-rag-system/
├── data_collection.py      # 從 Google Books API 收集資料
├── build_vectordb.py       # 建立 FAISS 向量資料庫
├── app.py                  # Streamlit 應用程式
├── requirements.txt        # Python 套件
├── .env.example           # 環境變數範例
├── .gitignore
├── README.md
├── data/
│   └── books_raw.json     # 原始書籍資料
└── vectordb/
    └── faiss_index/       # 向量資料庫
```

## 🚀 快速開始

### 1. 環境設定

```bash
# 安裝套件
pip install -r requirements.txt

# 設定 API Key
cp .env.example .env
# 編輯 .env 檔案，填入你的 API Key
```

### 2. 取得免費 API Key

#### 方案 A：HuggingFace（推薦）

1. 註冊帳號：https://huggingface.co/join
2. 前往設定：https://huggingface.co/settings/tokens
3. 建立 "Read" token
4. 複製 token（格式：`hf_xxxxx`）

**優點：**
- 完全免費
- 額度充足
- 支援多種模型

#### 方案 B：Google Gemini

1. 前往：https://makersuite.google.com/app/apikey
2. 建立 API Key
3. 複製 API Key

**優點：**
- 每分鐘 60 requests
- 中文支援好
- 回應速度快

### 3. 收集書籍資料

```bash
python data_collection.py
```

這會從 Google Books API 收集約 200-300 本書籍資料，包含以下類別：
- 小說、科幻、推理、愛情
- 歷史、科普、商業
- 自我成長、哲學、心理學

### 4. 建立向量資料庫

```bash
python build_vectordb.py
```

**注意：**
- 第一次執行會下載 embedding 模型（約 400MB）
- 只需執行一次
- 執行時間約 5-10 分鐘

### 5. 啟動應用程式

```bash
streamlit run app.py
```

應用程式會在 http://localhost:8501 啟動。

## 💡 使用方式

1. 在輸入框輸入問題，例如：
   - "推薦科幻小說"
   - "有什麼商業書籍？"
   - "適合初學者的心理學書"

2. 點選「獲取推薦」按鈕

3. AI 會：
   - 推薦 2-3 本相關書籍
   - 說明推薦理由
   - 顯示書籍封面、作者、類別等資訊
   - 提供預覽連結

## 🌐 部署到 Streamlit Cloud

### 步驟 1：準備專案

```bash
# 確保所有檔案都已提交
git add .
git commit -m "Add RAG book recommendation system"
git push
```

### 步驟 2：部署

1. 前往 https://share.streamlit.io/
2. 登入 GitHub 帳號
3. 選擇你的 repository
4. Main file path: `app.py`
5. 點選 Deploy

### 步驟 3：設定 Secrets

在 Streamlit Cloud 專案設定中，加入：

```toml
HUGGINGFACE_API_KEY = "hf_your_key"
# 或
GOOGLE_API_KEY = "your_key"
```

## ⚙️ 配置選項

在 [app.py](app.py#L19) 中可以切換 LLM：

```python
# 選擇使用哪個 LLM（二選一）
USE_LLM = "huggingface"  # 或 "gemini"
```

### HuggingFace 模型選項

在 [app.py](app.py#L31) 中可以更換模型：

```python
repo_id="mistralai/Mistral-7B-Instruct-v0.2"  # 或 "google/flan-t5-xxl"
```

## 📊 系統資訊

- **向量資料庫大小**：約 100-200MB
- **載入時間**：3-5 秒
- **問答回應時間**：5-10 秒
- **免費額度**：
  - HuggingFace：每小時約 1000 次請求
  - Gemini：每分鐘 60 次請求

## 🔧 疑難排解

### Q: 建立向量資料庫時記憶體不足？

A: 可以在 [data_collection.py](data_collection.py#L82) 中減少收集的書籍數量：

```python
books_per_category=20  # 改為較小的數字
```

### Q: HuggingFace 模型下載太慢？

A: 可以換用較小的模型：

```python
repo_id="google/flan-t5-base"  # 較小但速度快
```

### Q: Streamlit Cloud 部署失敗？

A: 確認：
1. `vectordb/` 資料夾是否太大（可以在部署後重新建立）
2. API Key 是否正確設定在 Secrets
3. requirements.txt 中的套件版本是否相容

### Q: 回答品質不理想？

A: 可以調整 prompt 模板或切換到 Gemini：

```python
USE_LLM = "gemini"  # Gemini 中文支援較好
```

## 📝 開發紀錄

- 專案建立：2025-11-29
- 使用技術：RAG, FAISS, LangChain, Streamlit
- 目標：完全免費的書籍推薦系統

## 📄 授權

MIT License

## 🙏 致謝

- Google Books API
- HuggingFace
- LangChain
- Streamlit
