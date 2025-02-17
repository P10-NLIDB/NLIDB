import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import json
import os
import re
import nltk
from transformers import BertTokenizer, BertModel

# Download necessary NLTK data
nltk.download('punkt')

# Load tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def load_spider_schema(spider_db_dir):
    """Loads schema from Spider dataset."""
    print("Loading database schemas...")
    schema = {}
    tables_path = os.path.join(spider_db_dir, "tables.json")

    with open(tables_path, "r") as f:
        tables_data = json.load(f)

    for db in tables_data:
        db_name = db["db_id"]
        schema[db_name] = {
            "tables": db["table_names_original"],
            "columns": {},
            "foreign_keys": db["foreign_keys"]
        }

        for col_id, (table_id, col_name) in enumerate(db["column_names_original"]):
            if table_id == -1:
                continue
            table_name = db["table_names_original"][table_id]
            if table_name not in schema[db_name]["columns"]:
                schema[db_name]["columns"][table_name] = []
            schema[db_name]["columns"][table_name].append(col_name)

    print(f"Loaded {len(schema)} database schemas.")
    return schema

def load_spider_data(spider_db_dir):
    """Loads Spider dataset (NL-SQL pairs)."""
    print("Loading Spider dataset...")
    train_path = os.path.join(spider_db_dir, "train_spider.json")

    with open(train_path, "r") as f:
        data = json.load(f)

    dataset = [(item["question"], item["query"], item["db_id"]) for item in data]
    print(f"Loaded {len(dataset)} training samples.")
    return dataset

def extract_ground_truth_links(sql_query, schema):
    """Extracts schema elements referenced in a SQL query."""
    sql_tokens = re.findall(r"[A-Za-z_]+", sql_query.lower())  
    ground_truth_schema = set()

    for table in schema["tables"]:
        if table.lower() in sql_tokens:
            ground_truth_schema.add(f"TABLE:{table}")

    for table, columns in schema["columns"].items():
        for column in columns:
            if column.lower() in sql_tokens:
                ground_truth_schema.add(f"COLUMN:{table}.{column}")

    return list(ground_truth_schema)

def preprocess_NL_question(question):
    """Tokenizes a natural language question."""
    return nltk.word_tokenize(question.lower())

def extract_schema(schema):
    """Extracts tables and columns as schema nodes."""
    schema_nodes = []
    for table in schema["tables"]:
        schema_nodes.append(f"TABLE:{table}")
    for table, columns in schema["columns"].items():
        for column in columns:
            schema_nodes.append(f"COLUMN:{table}.{column}")
    return schema_nodes

def compute_similarity(token, schema_element):
    """Computes similarity using BERT embeddings."""
    inputs = tokenizer([token, schema_element], return_tensors="pt", padding=True, truncation=True)
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return F.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()

def create_edges(NL_tokens, schema_nodes, threshold=0.5):
    """Creates edges between NL tokens and schema nodes."""
    edges = []
    for i, token in enumerate(NL_tokens):
        for j, schema_element in enumerate(schema_nodes):
            similarity = compute_similarity(token, schema_element)
            if similarity > threshold:
                edges.append((i, len(NL_tokens) + j)) 
    return edges

def build_graph(NL_tokens, schema_nodes, edges):
    """Constructs the full PyG graph before training."""
    num_nodes = len(NL_tokens) + len(schema_nodes)
    node_features = torch.rand((num_nodes, 128))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=node_features, edge_index=edge_index)

