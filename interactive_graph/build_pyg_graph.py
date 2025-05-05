import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import tqdm
from transformers import BertTokenizerFast, BertModel, BertForSequenceClassification
from utils.safe_file import safe_pickle_load, safe_pickle_save
from .process_dataset import generate_preprocessed_relational_data

"""
As a reminder, this is the data-flow

+-------------------------------------+
|    Question, Schema                 |
+-------------------------------------+
          |
          v
+-------------------------------------+
|  Pretrained Linker Model            |
| (predicts relevance of each         |
|  table/column given the question)   |
+-------------------------------------+
          |
          v
+--------------------------------------------------------------------+
|  Build BERT embeddings:                                            |
|  [CLS] question [SEP] schema_elem [SEP] relevance_token [SEP]      |
|   (relevance_token = 'relevant' / 'irrelevant')                    |
+--------------------------------------------------------------------+
          |
          v
+----------------------------------------------------------------+
|  Build Graph (nodes = embeddings, edges = schema relations)    |
+----------------------------------------------------------------+
          |
          v
+---------------------------------------+
|      GNN Classifier (Ambiguous / Not)  |
+---------------------------------------+
"""

def load_linker_model(linker_model_path="./linker_out/"):
    linker_model = BertForSequenceClassification.from_pretrained(linker_model_path)
    linker_tokenizer = BertTokenizerFast.from_pretrained(linker_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    linker_model.to(device)
    linker_model.eval()
    return linker_model, linker_tokenizer, device


def load_bert_encoder():
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    bert_model.eval()
    return bert_model, bert_tokenizer, device


@torch.no_grad()
def batch_predict_relevance(question: str, schema_elements, linker_model, linker_tokenizer, device, threshold=0.5):
    inputs = linker_tokenizer(
        [question] * len(schema_elements),
        schema_elements,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors='pt'
    ).to(device)

    logits = linker_model(**inputs).logits.squeeze(-1)  # (batch_size,)
    probs = torch.sigmoid(logits)
    relevance_flags = (probs >= threshold).tolist()  # List of True/False
    return relevance_flags

@torch.no_grad()
def batch_build_embeddings(question: str, schema_elements, relevance_flags, bert_model, bert_tokenizer, device):
    input_texts = [
        f"[CLS] {question} [SEP] {schema_element} [SEP] {'relevant' if is_relevant else 'irrelevant'} [SEP]"
        for schema_element, is_relevant in zip(schema_elements, relevance_flags)
    ]

    inputs = bert_tokenizer(
        input_texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors='pt'
    ).to(device)

    outputs = bert_model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
    return cls_embeddings  # (batch_size, hidden_dim)

def get_schema_node_map(db):
    """
    Creates an ordered mapping of schema elements to indices.
    Returns:
        node_to_idx: dict mapping schema node (str) -> index (int)
        idx_to_node: list of node names in index order
    """
    schema_nodes = db['processed_table_names'] + db['processed_column_names']
    node_to_idx = {name: idx for idx, name in enumerate(schema_nodes)}
    return node_to_idx, schema_nodes


@torch.no_grad()
def create_graph_from_schema(db, entry):
    node_to_idx, idx_to_node = get_schema_node_map(db)
    num_nodes = len(idx_to_node)

    edge_index = []
    schema_rel = db['relations']

    for i in range(len(schema_rel)):
        for j in range(len(schema_rel[0])):
            rel = schema_rel[i][j]
            if rel != 'none' and rel != '' and not rel.endswith("-generic"):
                edge_index.append([i, j])

    edge_index += [[j, i] for i, j in edge_index]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = torch.zeros((num_nodes, 1))  # dummy features, will be changed to the embedding
    
    is_ambiguous = bool(entry.get("is_ambiguous", False))
    y = torch.tensor([1.0 if is_ambiguous else 0.0], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.node_to_idx = node_to_idx 
    data.idx_to_node = idx_to_node  # reverse map

    return data


@torch.no_grad()
def generate_node_embeddings(
    entry,
    db,
    node_to_idx,
    linker_model,
    linker_tokenizer,
    linker_device,
    bert_model,
    bert_tokenizer,
    bert_device,
    validate_alignment=True,
):
    question = " ".join(entry['processed_question_toks'])
    schema_nodes = list(node_to_idx.keys())  # ordered list of schema node names, should assure schema-node-embedding alignment

    relevance_flags = batch_predict_relevance(
        question, schema_nodes, linker_model, linker_tokenizer, linker_device
    )

    cls_embeddings = batch_build_embeddings(
        question, schema_nodes, relevance_flags, bert_model, bert_tokenizer, bert_device
    )

    # Validation: ensure index i matches node_to_idx[schema_nodes[i]]
    if validate_alignment:
        for i, node_name in enumerate(schema_nodes):
            expected_idx = node_to_idx[node_name]
            if expected_idx != i:
                raise ValueError(
                    f"Embedding mismatch: node '{node_name}' is at index {i} "
                    f"but expected index {expected_idx} from node_to_idx."
                )

    return cls_embeddings  # [num_nodes, hidden_dim]


def load_or_build_graphs(
    data_base_dir,
    dataset_name,
    mode="train",
    used_coref=False,
    use_dependency=False,
    batch_size=32,
    overwrite=False,
):
    """
    Loads or builds PyG graphs with node alignment guaranteed via node_to_idx dictionary.
    """
    graph_file_path = os.path.join(
        data_base_dir, "preprocessed_dataset", dataset_name, f"{mode}_graphs.pkl"
    )

    if os.path.exists(graph_file_path) and not overwrite:
        print(f"Loading graphs from {graph_file_path}...")
        graph_dataset = safe_pickle_load(graph_file_path)

        dataset_file_path = os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, f"{mode}.pkl")
        tables_file_path = os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "tables.pkl")

        dataset = safe_pickle_load(dataset_file_path)
        tables = safe_pickle_load(tables_file_path)

    else:
        print("Preprocessing graphs from scratch...")
        dataset, tables = generate_preprocessed_relational_data(
            data_base_dir, dataset_name, mode,
            used_coref,
            use_dependency,
            overwrite
        )

        linker_model, linker_tokenizer, linker_device = load_linker_model()
        bert_model, bert_tokenizer, bert_device = load_bert_encoder()

        graph_dataset = []
        for entry in tqdm(dataset, desc="Building graphs"):
            db = tables[entry["db_id"]]

            graph = create_graph_from_schema(db, entry)
            node_embeddings = generate_node_embeddings(
                entry, db,
                graph.node_to_idx,
                linker_model, linker_tokenizer, linker_device,
                bert_model, bert_tokenizer, bert_device,
                validate_alignment=True
            )
            graph.x = node_embeddings
            graph_dataset.append(graph)


        safe_pickle_save(graph_dataset, graph_file_path)
        print(f"Saved {len(graph_dataset)} graphs to {graph_file_path}")

    loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True)
    return loader, graph_dataset, dataset, tables
