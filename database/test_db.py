from db import get_db
conn = get_db()
cursor = conn.cursor()

cursor.execute("SELECT * FROM users")
print(cursor.fetchall())
