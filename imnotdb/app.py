from fastapi import FastAPI
from pydantic import BaseModel
import hashlib
from datetime import datetime
import psycopg2


app = FastAPI()

SALT = "a2b530C3"


class Database:
    def __init__(
        self,
        user,
        password,
        database="jamiebest",
        host="localhost",
        port="5432",
    ):
        self.user = user
        self.password = password
        self.database = database
        self.host = host
        self.port = port

    def execute_query(self, query: str):
        conn = psycopg2.connect(
            dbname=self.database, user=self.user, password=self.password, host=self.host, port=self.port
        )
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        conn.close()


db = Database(
    user="jamiebest",
    password="jamiebest",
    database="jamiebest",
)


def hash_the_password(password, salt):
    return hashlib.sha256(salt.encode() + password.encode()).hexdigest()[:30]


def new_user_to_db(
    db,
    username: str,
    password: str,
    email: str,
):
    hashed = hash_the_password(password, SALT)

    db.execute_query(
        f"""
        INSERT INTO "user" (username, password, email, created_on)
        VALUES ('{username}', '{hashed}', '{email}', '{datetime.now()}');
        """
    )
    return True


class NewUser(BaseModel):
    username: str
    password: str
    email: str


@app.post("/add_user/")
def add_user(user: NewUser):
    new_user_to_db(db, user.username, user.password, user.email)
    return {"status": "success"}
