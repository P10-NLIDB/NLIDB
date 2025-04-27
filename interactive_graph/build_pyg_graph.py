import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import BertTokenizerFast, BertModel, BertForSequenceClassification
from utils.safe_file import safe_pickle_load, safe_pickle_save
from .process_dataset import generate_preprocessed_relational_data


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
def predict_relevance(question: str, schema_element: str, linker_model, linker_tokenizer, device) -> bool:
    enc = linker_tokenizer(
        question, schema_element,
        truncation=True,
        padding='max_length',
        max_length=64,
        return_tensors='pt'
    ).to(device)
    logits = linker_model(**enc).logits.squeeze(-1)
    prob = torch.sigmoid(logits).item()
    return prob >= 0.5  # Threshold here


@torch.no_grad()
def build_embedding(question: str, schema_element: str, is_relevant: bool, bert_model, bert_tokenizer, device):
    relevance_token = "relevant" if is_relevant else "irrelevant"
    input_text = f"[CLS] {question} [SEP] {schema_element} [SEP] {relevance_token} [SEP]"
    inputs = bert_tokenizer(
        input_text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    ).to(device)
    outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token
    return cls_embedding.squeeze(0)  # (hidden_dim,)


def build_pyg_graph(
    entry,
    db,
    linker_model,
    linker_tokenizer,
    linker_device,
    bert_model,
    bert_tokenizer,
    bert_device,
    mode='train',
    use_syntax=False,
    label=False
):
    question = " ".join(entry['processed_question_toks'])  # Preprocessed question
    schema_elements = db['processed_table_names'] + db['processed_column_names']

    # --- Build node features ---
    node_features = []
    for schema_element in schema_elements:
        is_relevant = predict_relevance(
            question, schema_element, linker_model, linker_tokenizer, linker_device
        )
        embedding = build_embedding(
            question, schema_element, is_relevant, bert_model, bert_tokenizer, bert_device
        )
        # Possibly soft max this...
        node_features.append(embedding)

    x = torch.stack(node_features)  # [num_nodes, hidden_dim]

    # --- Build edge_index ---
    edge_index = []

    schema_rel = db['relations']
    for i in range(len(schema_rel)):
        for j in range(len(schema_rel[0])):
            rel = schema_rel[i][j]
            if rel != 'none' and rel != "" and not rel.endswith("-generic"):
                edge_index.append([i, j])

    # Make graph undirected
    edge_index += [[j, i] for i, j in edge_index]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # --- Graph label: ambiguous or not ---
    y = torch.tensor([1] if label else [0], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)


def load_or_build_graphs(
    data_base_dir,
    dataset_name,
    mode="train",
    used_coref=False,
    use_dependency=False,
    batch_size=32,
    overwrite=False,
    build_fn=None
):
    """
    Loads preprocessed graphs for a specified dataset from disk, or generates them from scratch 
    if they do not exist or if overwrite is True. It returns a DataLoader suitable for training
    Graph Neural Networks (GNNs), along with the individual graphs, preprocessed dataset entries, 
    and schema information.

    Args:
        data_base_dir (str): Base directory path containing raw and processed data.
        dataset_name (str): Name of the dataset (e.g., 'spider', 'cosql', 'ambiQT').
        mode (str, optional): Mode of the dataset ('train', 'dev', or 'test'). Defaults to "train".
        used_coref (bool, optional): Whether coreference resolution preprocessing is applied. Defaults to False.
        use_dependency (bool, optional): Whether dependency parsing information is included. Defaults to False.
        batch_size (int, optional): Batch size for the PyTorch Geometric DataLoader. Defaults to 32.
        overwrite (bool, optional): If True, forces regeneration of graphs even if cached versions exist. Defaults to False.
        build_fn (callable, required): Function used to build individual graphs from preprocessed entries. 
                                       Must accept arguments (entry, db, turn) and return a PyG Data object.

    Returns:
        tuple:
            - loader (DataLoader): PyTorch Geometric DataLoader containing graph batches for training.
            - graph_dataset (list[Data]): List of PyG graph objects for individual dataset entries.
            - dataset (list[dict]): Preprocessed dataset entries containing questions and annotations.
            - tables (dict): Processed database schema information.
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

        if build_fn is None:
            raise ValueError("Must provide build_fn function to build graphs.")
        
        linker_model, linker_tokenizer, linker_device = load_linker_model()
        bert_model, bert_tokenizer, bert_device = load_bert_encoder()

        true_or_negative_label = True if dataset_name == "ambiQT" else False

        graph_dataset = [
            build_fn(
                entry, tables[entry["db_id"]],
                linker_model, linker_tokenizer, linker_device,
                bert_model, bert_tokenizer, bert_device,
                mode=mode, label=true_or_negative_label
            )
            for entry in dataset
        ]

        safe_pickle_save(graph_dataset, graph_file_path)
        print(f"Saved {len(graph_dataset)} graphs to {graph_file_path}")

    loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True)
    return loader, graph_dataset, dataset, tables