import os
from dotenv import load_dotenv
import psycopg
import pandas as pd


def load_data_from_db():
    load_dotenv()

    connection = {"sslmode": "require", "target_session_attrs": "read-write"}
    postgres_credentials = {
        "host": os.getenv("DB_DESTINATION_HOST"),
        "port": os.getenv("DB_DESTINATION_PORT"),
        "dbname": os.getenv("DB_DESTINATION_NAME"),
        "user": os.getenv("DB_DESTINATION_USER"),
        "password": os.getenv("DB_DESTINATION_PASSWORD"),
    }

    connection.update(postgres_credentials)

    with psycopg.connect(**connection) as conn:
        # напишите код здесь
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM clean_flats_prices")
            data = cur.fetchall()
            columns = [desc[0] for desc in cur.description]

    df = pd.DataFrame(data, columns=columns)
    return df


if __name__ == "__main__":
    df = load_data_from_db()
    print(df.head())
