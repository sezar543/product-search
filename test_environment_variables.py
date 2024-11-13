import os
from dotenv import load_dotenv
load_dotenv()

print("DB_USER:", os.getenv("POSTGRES_USER"))
print("DB_PASSWORD:", os.getenv("POSTGRES_PASSWORD"))
print("DB_HOST:", os.getenv("DB_HOST"))
print("DB_PORT:", os.getenv("DB_PORT"))