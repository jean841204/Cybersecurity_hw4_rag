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

def collect_books_data(categories=None, books_per_category=40):
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
            '心理學',
            '藝術',
            '音樂',
            '旅遊',
            '料理',
            '運動',
            '科技',
            '醫學',
            '教育', 
            '投資',
            '股票'
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
