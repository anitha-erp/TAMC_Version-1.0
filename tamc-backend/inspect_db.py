import json
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", 3306))

conn = pymysql.connect(
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    db=DB_NAME,
    port=DB_PORT,
)

cursor = conn.cursor(pymysql.cursors.DictCursor)

data = {}

cursor.execute("SELECT DISTINCT commodity_name FROM lots_new")
data["unique_commodities"] = cursor.fetchall()

cursor.execute("SELECT DISTINCT amc_name, district, mandal FROM lots_new")
data["unique_amcs"] = cursor.fetchall()

with open("db_debug_dump.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print("db_debug_dump.json created successfully!")

cursor.close()
conn.close()
