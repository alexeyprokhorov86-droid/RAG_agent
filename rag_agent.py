"""
🤖 RAG-агент | Кондитерская Прохорова
Интеллектуальный помощник с доступом к данным компании

Возможности:
- Поиск по закупкам, продажам, номенклатуре, клиентам
- Аналитические вопросы по данным
- История диалога в рамках сессии
"""

import streamlit as st
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
import anthropic
import json
import re

# ============================================================
# НАСТРОЙКИ
# ============================================================

st.set_page_config(
    page_title="AI-помощник | Кондитерская Прохорова",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ПОДКЛЮЧЕНИЕ К БД
# ============================================================

@st.cache_resource
def get_db_connection():
    return psycopg2.connect(
        host=st.secrets["postgres"]["host"],
        port=st.secrets["postgres"]["port"],
        database=st.secrets["postgres"]["database"],
        user=st.secrets["postgres"]["user"],
        password=st.secrets["postgres"]["password"],
    )


def execute_query(query: str, params: tuple = None) -> pd.DataFrame:
    """Выполняет SQL-запрос и возвращает DataFrame"""
    try:
        conn = get_db_connection()
        df = pd.read_sql(query, conn, params=params)
        return df
    except Exception as e:
        st.error(f"Ошибка БД: {e}")
        return pd.DataFrame()


# ============================================================
# ФУНКЦИИ ПОИСКА ПО ДАННЫМ
# ============================================================

def search_purchases(query: str = None, supplier: str = None, 
                     date_from: str = None, date_to: str = None,
                     limit: int = 20) -> dict:
    """Поиск по закупкам"""
    conditions = ["1=1"]
    params = []
    
    if query:
        conditions.append("(nomenclature_name ILIKE %s OR contractor_name ILIKE %s)")
        params.extend([f"%{query}%", f"%{query}%"])
    
    if supplier:
        conditions.append("contractor_name ILIKE %s")
        params.append(f"%{supplier}%")
    
    if date_from:
        conditions.append("doc_date >= %s")
        params.append(date_from)
    
    if date_to:
        conditions.append("doc_date <= %s")
        params.append(date_to)
    
    sql = f"""
        SELECT doc_date, doc_number, contractor_name, nomenclature_name,
               quantity, price, sum_total
        FROM purchase_prices
        WHERE {' AND '.join(conditions)}
        ORDER BY doc_date DESC
        LIMIT {limit}
    """
    
    df = execute_query(sql, tuple(params) if params else None)
    
    # Статистика
    stats_sql = f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT contractor_name) as suppliers,
            COUNT(DISTINCT nomenclature_name) as products,
            COALESCE(SUM(sum_total), 0) as total_sum,
            MIN(doc_date) as min_date,
            MAX(doc_date) as max_date
        FROM purchase_prices
        WHERE {' AND '.join(conditions)}
    """
    stats_df = execute_query(stats_sql, tuple(params) if params else None)
    stats = stats_df.iloc[0].to_dict() if not stats_df.empty else {}
    
    return {
        "type": "purchases",
        "data": df.to_dict('records') if not df.empty else [],
        "stats": stats,
        "query_used": sql
    }


def search_sales(query: str = None, client: str = None,
                 doc_type: str = None, date_from: str = None, 
                 date_to: str = None, limit: int = 20) -> dict:
    """Поиск по продажам"""
    conditions = ["1=1"]
    params = []
    
    if query:
        conditions.append("(nomenclature_name ILIKE %s OR client_name ILIKE %s)")
        params.extend([f"%{query}%", f"%{query}%"])
    
    if client:
        conditions.append("client_name ILIKE %s")
        params.append(f"%{client}%")
    
    if doc_type:
        conditions.append("doc_type = %s")
        params.append(doc_type)
    
    if date_from:
        conditions.append("doc_date >= %s")
        params.append(date_from)
    
    if date_to:
        conditions.append("doc_date <= %s")
        params.append(date_to)
    
    sql = f"""
        SELECT doc_type, doc_date, doc_number, client_name, 
               nomenclature_name, quantity, price, sum_with_vat
        FROM sales
        WHERE {' AND '.join(conditions)}
        ORDER BY doc_date DESC
        LIMIT {limit}
    """
    
    df = execute_query(sql, tuple(params) if params else None)
    
    stats_sql = f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT client_name) as clients,
            COALESCE(SUM(CASE WHEN doc_type = 'Реализация' THEN sum_with_vat ELSE 0 END), 0) as sales_sum,
            COALESCE(SUM(CASE WHEN doc_type = 'Корректировка' THEN sum_with_vat ELSE 0 END), 0) as corrections_sum
        FROM sales
        WHERE {' AND '.join(conditions)}
    """
    stats_df = execute_query(stats_sql, tuple(params) if params else None)
    stats = stats_df.iloc[0].to_dict() if not stats_df.empty else {}
    
    return {
        "type": "sales",
        "data": df.to_dict('records') if not df.empty else [],
        "stats": stats
    }


def search_nomenclature(query: str = None, limit: int = 30) -> dict:
    """Поиск по номенклатуре"""
    conditions = ["n.is_folder = false"]
    params = []
    
    if query:
        conditions.append("(n.name ILIKE %s OR n.article ILIKE %s)")
        params.extend([f"%{query}%", f"%{query}%"])
    
    sql = f"""
        SELECT n.name, n.article, n.code, nt.name as type_name
        FROM nomenclature n
        LEFT JOIN nomenclature_types nt ON n.type_id = nt.id
        WHERE {' AND '.join(conditions)}
        ORDER BY n.name
        LIMIT {limit}
    """
    
    df = execute_query(sql, tuple(params) if params else None)
    
    return {
        "type": "nomenclature",
        "data": df.to_dict('records') if not df.empty else [],
        "total_found": len(df)
    }


def search_clients(query: str = None, limit: int = 30) -> dict:
    """Поиск по клиентам"""
    conditions = ["1=1"]
    params = []
    
    if query:
        conditions.append("(name ILIKE %s OR inn ILIKE %s)")
        params.extend([f"%{query}%", f"%{query}%"])
    
    sql = f"""
        SELECT name, inn
        FROM clients
        WHERE {' AND '.join(conditions)}
        ORDER BY name
        LIMIT {limit}
    """
    
    df = execute_query(sql, tuple(params) if params else None)
    
    return {
        "type": "clients",
        "data": df.to_dict('records') if not df.empty else [],
        "total_found": len(df)
    }


def get_price_dynamics(nomenclature: str) -> dict:
    """Динамика цен на номенклатуру"""
    sql = """
        SELECT doc_date, contractor_name, price, quantity
        FROM purchase_prices
        WHERE nomenclature_name ILIKE %s
        ORDER BY doc_date
    """
    df = execute_query(sql, (f"%{nomenclature}%",))
    
    if df.empty:
        return {"type": "price_dynamics", "data": [], "message": "Данные не найдены"}
    
    stats = {
        "min_price": float(df['price'].min()),
        "max_price": float(df['price'].max()),
        "avg_price": float(df['price'].mean()),
        "first_price": float(df.iloc[0]['price']),
        "last_price": float(df.iloc[-1]['price']),
        "price_change_pct": round((df.iloc[-1]['price'] - df.iloc[0]['price']) / df.iloc[0]['price'] * 100, 1)
    }
    
    return {
        "type": "price_dynamics",
        "nomenclature": nomenclature,
        "data": df.to_dict('records'),
        "stats": stats
    }


def get_top_clients(date_from: str = None, date_to: str = None, limit: int = 10) -> dict:
    """Топ клиентов по продажам"""
    conditions = ["doc_type = 'Реализация'"]
    params = []
    
    if date_from:
        conditions.append("doc_date >= %s")
        params.append(date_from)
    if date_to:
        conditions.append("doc_date <= %s")
        params.append(date_to)
    
    sql = f"""
        SELECT client_name, 
               SUM(sum_with_vat) as total_sum,
               COUNT(DISTINCT doc_number) as orders_count
        FROM sales
        WHERE {' AND '.join(conditions)}
        GROUP BY client_name
        ORDER BY total_sum DESC
        LIMIT {limit}
    """
    
    df = execute_query(sql, tuple(params) if params else None)
    
    return {
        "type": "top_clients",
        "data": df.to_dict('records') if not df.empty else []
    }


def get_top_products(date_from: str = None, date_to: str = None, limit: int = 10) -> dict:
    """Топ продуктов по продажам"""
    conditions = ["doc_type = 'Реализация'"]
    params = []
    
    if date_from:
        conditions.append("doc_date >= %s")
        params.append(date_from)
    if date_to:
        conditions.append("doc_date <= %s")
        params.append(date_to)
    
    sql = f"""
        SELECT nomenclature_name,
               SUM(quantity) as total_qty,
               SUM(sum_with_vat) as total_sum
        FROM sales
        WHERE {' AND '.join(conditions)}
        GROUP BY nomenclature_name
        ORDER BY total_sum DESC
        LIMIT {limit}
    """
    
    df = execute_query(sql, tuple(params) if params else None)
    
    return {
        "type": "top_products",
        "data": df.to_dict('records') if not df.empty else []
    }


def get_summary_stats() -> dict:
    """Общая статистика по базе"""
    purchases = execute_query("""
        SELECT COUNT(*) as cnt, COALESCE(SUM(sum_total), 0) as total,
               MIN(doc_date) as min_date, MAX(doc_date) as max_date
        FROM purchase_prices
    """)
    
    sales = execute_query("""
        SELECT COUNT(*) as cnt, 
               COALESCE(SUM(CASE WHEN doc_type='Реализация' THEN sum_with_vat ELSE 0 END), 0) as sales_total,
               COALESCE(SUM(CASE WHEN doc_type='Корректировка' THEN sum_with_vat ELSE 0 END), 0) as corr_total,
               MIN(doc_date) as min_date, MAX(doc_date) as max_date
        FROM sales
    """)
    
    nom_count = execute_query("SELECT COUNT(*) as cnt FROM nomenclature WHERE is_folder = false")
    clients_count = execute_query("SELECT COUNT(*) as cnt FROM clients")
    
    return {
        "type": "summary",
        "purchases": {
            "records": int(purchases.iloc[0]['cnt']),
            "total_sum": float(purchases.iloc[0]['total']),
            "period": f"{purchases.iloc[0]['min_date']} — {purchases.iloc[0]['max_date']}"
        },
        "sales": {
            "records": int(sales.iloc[0]['cnt']),
            "sales_sum": float(sales.iloc[0]['sales_total']),
            "corrections_sum": float(sales.iloc[0]['corr_total']),
            "period": f"{sales.iloc[0]['min_date']} — {sales.iloc[0]['max_date']}"
        },
        "nomenclature_count": int(nom_count.iloc[0]['cnt']),
        "clients_count": int(clients_count.iloc[0]['cnt'])
    }


# ============================================================
# ОПРЕДЕЛЕНИЕ ИНСТРУМЕНТОВ ДЛЯ LLM
# ============================================================

TOOLS = [
    {
        "name": "search_purchases",
        "description": "Поиск по закупкам (документы 'Приобретение товаров и услуг'). Используй для вопросов о закупках, поставщиках, закупочных ценах.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос (название товара или поставщика)"},
                "supplier": {"type": "string", "description": "Фильтр по поставщику"},
                "date_from": {"type": "string", "description": "Начало периода (YYYY-MM-DD)"},
                "date_to": {"type": "string", "description": "Конец периода (YYYY-MM-DD)"},
                "limit": {"type": "integer", "description": "Максимум записей", "default": 20}
            }
        }
    },
    {
        "name": "search_sales",
        "description": "Поиск по продажам (реализации и корректировки). Используй для вопросов о продажах, клиентах, выручке.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Поисковый запрос"},
                "client": {"type": "string", "description": "Фильтр по клиенту"},
                "doc_type": {"type": "string", "enum": ["Реализация", "Корректировка"], "description": "Тип документа"},
                "date_from": {"type": "string", "description": "Начало периода (YYYY-MM-DD)"},
                "date_to": {"type": "string", "description": "Конец периода (YYYY-MM-DD)"},
                "limit": {"type": "integer", "description": "Максимум записей", "default": 20}
            }
        }
    },
    {
        "name": "search_nomenclature",
        "description": "Поиск по справочнику номенклатуры. Используй для поиска товаров, продуктов, материалов.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Название или артикул товара"},
                "limit": {"type": "integer", "description": "Максимум записей", "default": 30}
            }
        }
    },
    {
        "name": "search_clients",
        "description": "Поиск по справочнику клиентов (партнёров).",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Название или ИНН клиента"},
                "limit": {"type": "integer", "description": "Максимум записей", "default": 30}
            }
        }
    },
    {
        "name": "get_price_dynamics",
        "description": "Получить динамику цен на конкретную номенклатуру. Показывает историю закупочных цен.",
        "input_schema": {
            "type": "object",
            "properties": {
                "nomenclature": {"type": "string", "description": "Название номенклатуры"}
            },
            "required": ["nomenclature"]
        }
    },
    {
        "name": "get_top_clients",
        "description": "Получить топ клиентов по сумме продаж.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "Начало периода (YYYY-MM-DD)"},
                "date_to": {"type": "string", "description": "Конец периода (YYYY-MM-DD)"},
                "limit": {"type": "integer", "description": "Количество в топе", "default": 10}
            }
        }
    },
    {
        "name": "get_top_products",
        "description": "Получить топ продуктов по сумме продаж.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "Начало периода (YYYY-MM-DD)"},
                "date_to": {"type": "string", "description": "Конец периода (YYYY-MM-DD)"},
                "limit": {"type": "integer", "description": "Количество в топе", "default": 10}
            }
        }
    },
    {
        "name": "get_summary_stats",
        "description": "Получить общую статистику по базе данных: количество записей, суммы, периоды.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Выполняет инструмент и возвращает результат"""
    try:
        if tool_name == "search_purchases":
            result = search_purchases(**tool_input)
        elif tool_name == "search_sales":
            result = search_sales(**tool_input)
        elif tool_name == "search_nomenclature":
            result = search_nomenclature(**tool_input)
        elif tool_name == "search_clients":
            result = search_clients(**tool_input)
        elif tool_name == "get_price_dynamics":
            result = get_price_dynamics(**tool_input)
        elif tool_name == "get_top_clients":
            result = get_top_clients(**tool_input)
        elif tool_name == "get_top_products":
            result = get_top_products(**tool_input)
        elif tool_name == "get_summary_stats":
            result = get_summary_stats()
        else:
            result = {"error": f"Неизвестный инструмент: {tool_name}"}
        
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


# ============================================================
# РАБОТА С LLM
# ============================================================

SYSTEM_PROMPT = """Ты — интеллектуальный помощник компании "Кондитерская Прохорова".

У тебя есть доступ к данным компании из 1С:
- Закупки (поставщики, цены, номенклатура)
- Продажи (клиенты, реализации, корректировки)
- Справочник номенклатуры (товары, продукция)
- Справочник клиентов

Правила:
1. Используй инструменты для поиска данных перед ответом
2. Отвечай на русском языке
3. Форматируй числа с разделителями тысяч (1 234 567)
4. Суммы указывай в рублях
5. Если данных нет — так и скажи, не выдумывай
6. При аналитических вопросах показывай статистику и выводы
7. Будь кратким, но информативным

Текущая дата: {current_date}
"""


def get_ai_response(user_message: str, chat_history: list) -> tuple[str, list]:
    """Получает ответ от Claude с использованием инструментов"""
    
    client = anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])
    
    # Формируем сообщения
    messages = []
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    
    system = SYSTEM_PROMPT.format(current_date=datetime.now().strftime("%Y-%m-%d"))
    
    tools_used = []
    
    # Цикл обработки с инструментами
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages
        )
        
        # Проверяем нужно ли вызвать инструменты
        if response.stop_reason == "tool_use":
            # Собираем все вызовы инструментов
            tool_calls = [block for block in response.content if block.type == "tool_use"]
            
            # Добавляем ответ ассистента в историю
            messages.append({"role": "assistant", "content": response.content})
            
            # Выполняем инструменты и добавляем результаты
            tool_results = []
            for tool_call in tool_calls:
                tool_result = execute_tool(tool_call.name, tool_call.input)
                tools_used.append({
                    "name": tool_call.name,
                    "input": tool_call.input,
                    "output_preview": tool_result[:500] + "..." if len(tool_result) > 500 else tool_result
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": tool_result
                })
            
            messages.append({"role": "user", "content": tool_results})
        else:
            # Финальный ответ
            final_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_response += block.text
            
            return final_response, tools_used


# ============================================================
# ИНТЕРФЕЙС
# ============================================================

def main():
    st.title("🤖 AI-помощник")
    st.caption("Кондитерская Прохорова | Данные из 1С")
    
    # Проверка API ключа
    if "anthropic" not in st.secrets or "api_key" not in st.secrets["anthropic"]:
        st.error("❌ Не настроен API ключ Anthropic")
        st.info("Добавьте в secrets.toml:\n```\n[anthropic]\napi_key = \"sk-ant-...\"\n```")
        return
    
    # Проверка подключения к БД
    try:
        get_db_connection()
    except Exception as e:
        st.error(f"❌ Ошибка подключения к БД: {e}")
        return
    
    # Инициализация истории чата
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "tools_log" not in st.session_state:
        st.session_state.tools_log = []
    
    # Боковая панель
    with st.sidebar:
        st.header("ℹ️ О помощнике")
        st.markdown("""
        Я могу помочь с:
        - 📦 Поиск по закупкам и ценам
        - 💰 Анализ продаж
        - 👥 Информация о клиентах
        - 📊 Статистика и аналитика
        """)
        
        st.divider()
        
        st.header("💡 Примеры вопросов")
        examples = [
            "Какая общая статистика по базе?",
            "Топ-10 клиентов по продажам",
            "Покажи закупки сахара за последний месяц",
            "Какая динамика цен на муку?",
            "Найди клиента Магнит",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.pending_message = ex
                st.rerun()
        
        st.divider()
        
        if st.button("🗑️ Очистить историю", use_container_width=True):
            st.session_state.messages = []
            st.session_state.tools_log = []
            st.rerun()
        
        # Лог инструментов
        if st.session_state.tools_log:
            with st.expander("🔧 Использованные инструменты"):
                for i, tool in enumerate(st.session_state.tools_log[-10:]):
                    st.markdown(f"**{tool['name']}**")
                    st.json(tool['input'])
    
    # Отображение истории чата
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Поле ввода чата (всегда видимое)
    user_input = st.chat_input("Задайте вопрос...")
    
    # Обработка примера из боковой панели
    if "pending_message" in st.session_state:
        user_input = st.session_state.pending_message
        del st.session_state.pending_message
    
    # Обработка ввода
    if user_input:
        # Добавляем сообщение пользователя
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Получаем ответ
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                try:
                    response, tools_used = get_ai_response(
                        user_input, 
                        st.session_state.messages[:-1]  # без последнего сообщения
                    )
                    st.markdown(response)
                    
                    # Сохраняем
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.tools_log.extend(tools_used)
                    
                except Exception as e:
                    error_msg = f"❌ Ошибка: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
