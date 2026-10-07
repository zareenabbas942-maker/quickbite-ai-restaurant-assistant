import json
import random
import sqlite3
from datetime import datetime

import gradio as gr
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# =========================================================
# QUICKBITE AI RESTAURANT ASSISTANT
# =========================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
DATABASE = "restaurant_orders.db"


# =========================================================
# MENU
# =========================================================

menu = [
    {"id": "B01", "name": "Zinger Burger", "price": 450},
    {"id": "B02", "name": "Chicken Burger", "price": 400},
    {"id": "P01", "name": "Chicken Pizza", "price": 700},
    {"id": "F01", "name": "Fries", "price": 200},
    {"id": "D01", "name": "Cola", "price": 150},
]


# =========================================================
# CART
# =========================================================

cart = []


def add_to_cart(item_id, quantity=1):
    item = next(
        (item for item in menu if item["id"] == item_id),
        None
    )

    if item is None:
        return "Item not found."

    cart.append({
        "id": item["id"],
        "name": item["name"],
        "price": item["price"],
        "quantity": quantity
    })

    return f"{quantity} x {item['name']} added to cart."


def calculate_total():
    total = 0

    for item in cart:
        total += item["price"] * item["quantity"]

    return total


def show_cart():
    if not cart:
        return "🛒 Cart is empty."

    lines = ["🛒 YOUR CART\n"]

    for item in cart:
        subtotal = item["price"] * item["quantity"]

        lines.append(
            f"{item['id']} | "
            f"{item['quantity']} x {item['name']} "
            f"= PKR {subtotal}"
        )

    lines.append("\n-------------------------")
    lines.append(f"💰 TOTAL: PKR {calculate_total()}")

    return "\n".join(lines)


# =========================================================
# DATABASE
# =========================================================

def setup_database():
    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id TEXT UNIQUE,
        token TEXT,
        items TEXT,
        total INTEGER,
        status TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

    print("✅ Database is ready!")


# =========================================================
# AI ORDER UNDERSTANDING
# =========================================================

def understand_order(customer_message):

    menu_text = "\n".join(
        f"{item['id']} | {item['name']} | PKR {item['price']}"
        for item in menu
    )

    prompt = f"""
You are a restaurant order assistant.

Available menu:

{menu_text}

Convert the customer's order into JSON.

Rules:

1. Use only item IDs from the menu.
2. Include the quantity for each item.
3. Do not calculate prices.
4. Do not add items that are not in the menu.
5. Return ONLY valid JSON.

Format:

{{
  "items": [
    {{"id": "B01", "quantity": 2}},
    {{"id": "F01", "quantity": 1}}
  ]
}}

Customer:

{customer_message}
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            return ""

        return response.json()["response"].strip()

    except Exception:
        return ""


# =========================================================
# ADD AI ORDER TO CART
# =========================================================

def add_ai_order_to_cart(order_json):

    try:
        data = json.loads(order_json)

    except json.JSONDecodeError:
        return "❌ AI response valid JSON nahi hai."

    for item in data.get("items", []):

        item_id = item.get("id")
        quantity = item.get("quantity", 0)

        menu_item = next(
            (m for m in menu if m["id"] == item_id),
            None
        )

        if menu_item is None:
            return f"❌ Invalid menu item: {item_id}"

        if not isinstance(quantity, int) or quantity <= 0:
            return f"❌ Invalid quantity for {item_id}"

        add_to_cart(item_id, quantity)

    return "✅ Order items successfully added to cart."


# =========================================================
# ORDER CONFIRMATION
# =========================================================

def confirm_order():

    if not cart:
        return "Cart is empty."

    order_id = f"ORD-{random.randint(1000, 9999)}"

    token = f"T{random.randint(100, 999)}"

    total = calculate_total()

    order = {
        "order_id": order_id,
        "token": token,
        "items": cart.copy(),
        "total": total,
        "status": "Pending",
        "created_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    return order


# =========================================================
# PLACE ORDER
# =========================================================

def place_order():

    if not cart:
        return "🛒 Cart is empty."

    order = confirm_order()

    if isinstance(order, str):
        return order

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR REPLACE INTO orders (
        order_id,
        token,
        items,
        total,
        status,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        order["order_id"],
        order["token"],
        json.dumps(order["items"]),
        order["total"],
        order["status"],
        order["created_at"]
    ))

    conn.commit()
    conn.close()

    cart.clear()

    return (
        f"✅ ORDER CONFIRMED\n\n"
        f"🆔 Order ID: {order['order_id']}\n"
        f"🎟️ Token: {order['token']}\n"
        f"💰 Total: PKR {order['total']}\n"
        f"📦 Status: {order['status']}"
    )


# =========================================================
# ORDER TRACKING
# =========================================================

def check_order_status(order_id):

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
    SELECT order_id, token, total, status, created_at
    FROM orders
    WHERE order_id = ?
    """, (order_id,))

    result = cursor.fetchone()

    conn.close()

    if result is None:
        return "❌ Order not found."

    return (
        f"📦 ORDER STATUS\n\n"
        f"Order ID: {result[0]}\n"
        f"Token: {result[1]}\n"
        f"Total: PKR {result[2]}\n"
        f"Status: {result[3]}\n"
        f"Time: {result[4]}"
    )


# =========================================================
# KITCHEN MANAGEMENT
# =========================================================

def update_kitchen_status(order_id, new_status):

    valid_statuses = [
        "Pending",
        "Preparing",
        "Ready",
        "Completed"
    ]

    if new_status not in valid_statuses:
        return (
            "❌ Invalid status. Use: "
            + ", ".join(valid_statuses)
        )

    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
    UPDATE orders
    SET status = ?
    WHERE order_id = ?
    """, (new_status, order_id))

    conn.commit()

    if cursor.rowcount == 0:
        result = "❌ Order not found."
    else:
        result = (
            f"✅ {order_id} updated to: "
            f"{new_status}"
        )

    conn.close()

    return result


# =========================================================
# RAG KNOWLEDGE BASE
# =========================================================

restaurant_knowledge = """
QuickBite Restaurant is an AI-powered restaurant ordering system.

MENU:

B01 - Zinger Burger - PKR 450
B02 - Chicken Burger - PKR 400
P01 - Chicken Pizza - PKR 700
F01 - Fries - PKR 200
D01 - Cola - PKR 150

ORDERING:

Customers can place orders using natural language.
The AI assistant understands food items and quantities.
Customers can view their cart and calculate the total bill.

ORDER CONFIRMATION:

After confirmation, the system generates an Order ID and Token.
The order initially has Pending status.

ORDER TRACKING:

Customers can track an order using its Order ID.
Order status can be Pending, Preparing, Ready, or Completed.

KITCHEN:

Kitchen staff can update the order status.
"""


knowledge_chunks = [
    chunk.strip()
    for chunk in restaurant_knowledge.split("\n\n")
    if chunk.strip()
]


# =========================================================
# FAISS VECTOR STORE
# =========================================================

print("⏳ Loading AI knowledge base...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
    knowledge_chunks,
    convert_to_numpy=True
)

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(
    np.array(embeddings).astype("float32")
)

print("✅ RAG knowledge base ready!")


# =========================================================
# RAG SEARCH
# =========================================================

def search_knowledge(query, top_k=2):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for i in indices[0]:

        if i < len(knowledge_chunks):
            results.append(
                knowledge_chunks[i]
            )

    return results


# =========================================================
# RAG QUESTION ANSWERING
# =========================================================

def ask_rag_question(question):

    try:

        results = search_knowledge(
            question,
            top_k=2
        )

        context = "\n\n".join(results)

        prompt = f"""
You are QuickBite Restaurant AI Assistant.

Answer the customer's question using only the restaurant information below.

Restaurant Information:

{context}

Customer Question:

{question}

Give a short and helpful answer.
"""

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        if response.status_code != 200:
            return "AI service is currently unavailable."

        data = response.json()

        return data.get(
            "response",
            "Sorry, I could not generate a response."
        ).strip()

    except Exception as e:

        return f"AI service error: {str(e)}"


# =========================================================
# MAIN CHAT FUNCTION
# =========================================================

def quickbite_chat(message):

    if not message.strip():
        return "Please enter your order or question."

    try:

        order_json = understand_order(
            message
        )

        data = json.loads(order_json)

        if "items" in data and data["items"]:

            cart.clear()

            result = add_ai_order_to_cart(
                order_json
            )

            if result.startswith("✅"):

                return (
                    f"{result}\n\n"
                    f"🛒 CART TOTAL: "
                    f"PKR {calculate_total()}\n\n"
                    "Please confirm your order "
                    "using the Confirm Order button."
                )

            return result

    except (json.JSONDecodeError, TypeError):

        pass

    return ask_rag_question(message)


# =========================================================
# GRADIO FUNCTIONS
# =========================================================

def refresh_cart():
    return show_cart()


def track_order(order_id):

    if not order_id.strip():
        return "Please enter your Order ID."

    return check_order_status(
        order_id.strip()
    )


def kitchen_status(order_id, status):

    if not order_id.strip():
        return "Please enter your Order ID."

    return update_kitchen_status(
        order_id.strip(),
        status
    )


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

setup_database()


# =========================================================
# GRADIO UI
# =========================================================

with gr.Blocks(
    title="QuickBite Restaurant"
) as final_demo:

    gr.Markdown(
        "# 🍔 QuickBite Restaurant"
    )

    gr.Markdown(
        "### 🤖 AI-Powered Restaurant Ordering System"
    )

    with gr.Row():

        # =================================================
        # CUSTOMER SIDE
        # =================================================

        with gr.Column():

            gr.Markdown(
                "## 🤖 AI Restaurant Assistant"
            )

            message = gr.Textbox(
                label="Your message",
                placeholder=(
                    "Example: I want 2 Zinger "
                    "Burgers and 1 Fries"
                )
            )

            assistant = gr.Textbox(
                label="Assistant",
                lines=6
            )

            send = gr.Button(
                "Send",
                variant="primary"
            )

            send.click(
                fn=quickbite_chat,
                inputs=message,
                outputs=assistant
            )

            gr.Markdown(
                "## 🛒 Your Cart"
            )

            cart_output = gr.Textbox(
                label="Cart",
                lines=8
            )

            refresh = gr.Button(
                "🔄 Refresh Cart"
            )

            refresh.click(
                fn=refresh_cart,
                inputs=None,
                outputs=cart_output
            )

            confirm = gr.Button(
                "✅ Confirm Order",
                variant="primary"
            )

            confirmation = gr.Textbox(
                label="Order Confirmation",
                lines=7
            )

            confirm.click(
                fn=place_order,
                inputs=None,
                outputs=confirmation
            )

        # =================================================
        # RESTAURANT SIDE
        # =================================================

        with gr.Column():

            gr.Markdown(
                "## 🍔 Menu"
            )

            for item in menu:

                gr.Markdown(
                    f"**{item['id']} — "
                    f"{item['name']}**  \n"
                    f"Price: PKR {item['price']}"
                )

            gr.Markdown(
                "## 📦 Order Tracking"
            )

            order_id_input = gr.Textbox(
                label="Order ID",
                placeholder="Example: ORD-4590"
            )

            track_button = gr.Button(
                "🔎 Check Order Status"
            )

            status_output = gr.Textbox(
                label="Order Status",
                lines=7
            )

            track_button.click(
                fn=track_order,
                inputs=order_id_input,
                outputs=status_output
            )

            gr.Markdown(
                "## 👨‍🍳 Kitchen Management"
            )

            kitchen_order_id = gr.Textbox(
                label="Order ID",
                placeholder="Example: ORD-4590"
            )

            kitchen_status_dropdown = gr.Dropdown(
                choices=[
                    "Pending",
                    "Preparing",
                    "Ready",
                    "Completed"
                ],
                value="Preparing",
                label="New Kitchen Status"
            )

            kitchen_button = gr.Button(
                "👨‍🍳 Update Kitchen Status"
            )

            kitchen_output = gr.Textbox(
                label="Kitchen Update",
                lines=4
            )

            kitchen_button.click(
                fn=kitchen_status,
                inputs=[
                    kitchen_order_id,
                    kitchen_status_dropdown
                ],
                outputs=kitchen_output
            )

    gr.Markdown("---")

    gr.Markdown(
        "### QuickBite • AI Restaurant Ordering System"
    )


# =========================================================
# LAUNCH
# =========================================================

final_demo.launch(
    server_name="127.0.0.1",
    server_port=7860
)