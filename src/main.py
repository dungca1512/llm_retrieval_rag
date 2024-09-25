from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# Mô hình dữ liệu
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

# Giả lập cơ sở dữ liệu
items_db = {}

@app.get("/")
async def root():
    return {"message": "Welcome to the Item API"}

@app.post("/items/")
async def create_item(item: Item):
    item_id = len(items_db) + 1
    items_db[item_id] = item
    return {"item_id": item_id, **item.dict()}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, **items_db[item_id].dict()}

@app.get("/items/")
async def list_items(skip: int = 0, limit: int = 10):
    return list(items_db.values())[skip : skip + limit]

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    items_db[item_id] = item
    return {"item_id": item_id, **item.dict()}

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    if item_id not in items_db:
        raise HTTPException(status_code=404, detail="Item not found")
    del items_db[item_id]
    return {"message": "Item deleted successfully"}

@app.get("/items/search/")
async def search_items(q: str = Query(None, min_length=3, max_length=50)):
    results = [item for item in items_db.values() if q.lower() in item.name.lower()]
    return results

# Để chạy ứng dụng, sử dụng lệnh:
# uvicorn main:app --reload