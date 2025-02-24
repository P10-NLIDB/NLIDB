import psycopg2
from psycopg2 import sql
from psycopg2 import pool
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum

# Store connection pools for multiple databases
connection_pools = {}

# Default database config (used for creating new databases)
DEFAULT_DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "yourpassword",
    "host": "localhost",
    "port": "5432"
}

USER_INTERACTIONS_DB = "user_interactions_db"

class SimilarityMethod(Enum):
    L2 = "<->"             # Euclidean Distance
    INNER_PRODUCT = "<#>"  # Negative Inner Product
    COSINE = "<=>"         # Cosine Distance
    L1 = "<+>"             # Manhattan Distance
    HAMMING = "<~>"        # Hamming Distance (binary vectors)
    JACCARD = "<%>"        # Jaccard Distance (binary vectors)

def create_database(db_name, reset=False):
    """
    Creates a new database. If reset=True, it will drop and recreate it.
    """
    with psycopg2.connect(**DEFAULT_DB_CONFIG) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            if reset:
                cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))
                print(f"Dropped database '{db_name}'.")
            cursor.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), (db_name,))
            exists = cursor.fetchone()
            if not exists:
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
                print(f"Database '{db_name}' created successfully.")
            else:
                print(f"Database '{db_name}' already exists.")

def setup_connection_pool(db_name, db_user="postgres", db_password="yourpassword", db_host="localhost", db_port="5432"):
    """
    Sets up a connection pool for a given database.
    """
    if db_name not in connection_pools:
        connection_pools[db_name] = psycopg2.pool.SimpleConnectionPool(
            minconn=1, maxconn=5,
            dbname=db_name, user=db_user, password=db_password, host=db_host, port=db_port
        )

def get_pool(db_name):
    """
    Retrieves a connection pool for the specified database.
    """
    return connection_pools.get(db_name)

@contextmanager
def get_connection(db_name):
    """
    Gets a connection from the pool and ensures it is released after use.
    """
    pool = get_pool(db_name)
    if not pool:
        raise ValueError(f"Database {db_name} is not registered.")

    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

def close_all_pools():
    """
    Closes all database connection pools.
    """
    for pool in connection_pools.values():
        pool.closeall()

def execute_query(db_name, query, params=None, fetch=False):
    """
    Executes a SQL query on the specified database.
    """
    with get_connection(db_name) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
            if fetch:
                return cursor.fetchall()
            conn.commit()

def init_user_interactions_db():
    """
    Enables pgvector and initializes the user_interactions table with an HNSW index.
    """
    db_name = USER_INTERACTIONS_DB
    with get_connection(db_name) as conn:
        with conn.cursor() as cursor:
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create user interactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_interactions (
                    id SERIAL PRIMARY KEY,
                    database_name TEXT NOT NULL,  -- Spider dataset is multi-db so we also need to track the Spider database associated with the query
                    query_text TEXT NOT NULL,
                    query_embedding VECTOR(768),  -- (BERT: 768)
                    sql_query TEXT NOT NULL,
                    user_feedback INTEGER CHECK (user_feedback BETWEEN 1 AND 5) NULL, -- Satisfaction score (1-5) or not set
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create HNSW index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS query_embedding_hnsw_idx
                ON user_interactions USING hnsw (query_embedding);
            """)

            conn.commit()
            print(f"Database '{db_name}' initialized with user_interactions table and HNSW index.")

def insert_user_interaction(database_name, query_text, query_embedding, sql_query, user_feedback=None):
    """
    Inserts a new user interaction into the shared user_interactions table.
    The database_name column tracks which Spider database the query belongs to.
    """
    db_name = USER_INTERACTIONS_DB
    query = """
        INSERT INTO user_interactions (database_name, query_text, query_embedding, sql_query, user_feedback, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING id
    """
    params = (database_name, query_text, query_embedding, sql_query, user_feedback, datetime.now(timezone.utc))
    return execute_query(db_name, query, params, fetch=True)[0][0]

def search_similar_queries(vector, method: SimilarityMethod = SimilarityMethod.COSINE, limit=5):
    """
    Finds similar user queries using a specified vector similarity method.
    """
    db_name = USER_INTERACTIONS_DB
    query = f"""
        SELECT id, query_text, sql_query, user_feedback, timestamp, query_embedding {method.value} %s AS distance
        FROM user_interactions
        ORDER BY distance
        LIMIT %s
    """
    return execute_query(db_name, query, (vector, limit), fetch=True)

def get_all_interactions():
    """
    Retrieves all stored user interactions.
    """
    db_name = USER_INTERACTIONS_DB
    query = "SELECT * FROM user_interactions ORDER BY timestamp DESC"
    return execute_query(db_name, query, fetch=True)


def initialize_system(databases, reset=False):
    """
    Initializes the entire system:
    1. Creates the user interactions database (if needed)
    2. Sets up connection pools for Spider databases
    3. Initializes the user interactions table
    """
    print("Initializing System...")
    
    # Create the main user interactions database
    create_database(USER_INTERACTIONS_DB, reset=reset)
    setup_connection_pool(USER_INTERACTIONS_DB)

    # Initialize the user interactions table
    init_user_interactions_db()

    # Setup connection pools for Spider databases
    # Maybe change this part so that we dont have a billion database connections because of spiders 100s of databases...
    for db in databases:
        setup_connection_pool(db)
    
    print("\System Initialization Complete\n")

# Example usage
if __name__ == "__main__":
    # Define Spider dataset databases
    # Update this to dynamically load all databases
    spider_dbs = ["restaurants"] 

    # Initialize the system (set reset=True to start fresh)
    initialize_system(spider_dbs, reset=False)

    # Insert a new user interaction into a Spider database (Example: restaurants)
    interaction_id = insert_user_interaction(
        "restaurants",
        "Find all restaurants with a rating above 4.5",
        [0.1] * 768,
        "SELECT name FROM restaurants WHERE rating > 4.5;",
        5
    )
    print(f"Inserted interaction with ID: {interaction_id}")

    # Search for similar queries
    query_vector = [0.1] * 768
    similar_queries = search_similar_queries(query_vector, SimilarityMethod.COSINE)

    print("\nSimilar Queries:")
    for row in similar_queries:
        print(row)

    # Close all connection pools
    close_all_pools()