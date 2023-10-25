from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from datetime import datetime

# fastapi is like light weight flask, but with more features
# other alternatives: flask, django

api = FastAPI()


@api.get("/")
def read_root():
    return {"message": "This is main page"}


@api.get("/right")
def say_something_wise():
    return "something wise: wife is always right, that's why husband is left"


@api.get("/page1")
def page1():
    with open("page1.html", "r") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)


@api.get("/now")
def now():
    return {"now": datetime.now()}


@api.get("/duplicate/{name}/{number}")
def duplicate(name: str, number: int):
    return {"duplicate": name * number}


class Fruit(BaseModel):
    name: str


@api.post("/add_fruit/")
async def add_fruit_func(fruit: Fruit):
    print(f"fruit added: {fruit.name}")
    return {"fruit": f"fruit added: {fruit.name}", "status": "success"}


@api.get("/search_amazing/{sentence}")
def search_amazing(sentence: str):
    return {"results": [f"{sentence} is amazing"]}
