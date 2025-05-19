import gc
import os
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
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
    linker_model = BertForSequenceClassification.from_pretrained(
        linker_model_path)
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

    outputs = bert_model.bert(**inputs)
    # [batch_size, hidden_dim]
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings  # (batch_size, hidden_dim)


def get_schema_node_map(db):
    """
    Returns
        node_to_idx : dict  (schema node -> unique index)
        idx_to_node : list  (unique nodes in stable order)
    """
    schema_nodes = db['processed_table_names'] + db['processed_column_names']

    node_to_idx = {}
    idx_to_node = []
    for name in schema_nodes:
        if name not in node_to_idx:
            node_to_idx[name] = len(idx_to_node)
            idx_to_node.append(name)

    return node_to_idx, idx_to_node


@torch.no_grad()
def create_graph_from_schema(db, entry):
    raw_nodes = db['processed_table_names'] + db['processed_column_names']

    node_to_idx, unique_nodes = get_schema_node_map(db)

    edges = []
    for i, row in enumerate(db['relations']):
        for j, rel in enumerate(row):
            if rel not in ('none', '', None) and not str(rel).endswith('-generic'):
                ui = node_to_idx[raw_nodes[i]]
                uj = node_to_idx[raw_nodes[j]]
                edges.extend(([ui, uj], [uj, ui]))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # ── sanity (should never fail now)
    if edge_index.numel() and edge_index.max() >= len(unique_nodes):
        raise RuntimeError("ghost node still present – mapping bug")

    x = torch.zeros((len(unique_nodes), 1))          # placeholder
    y = torch.tensor([float(entry.get('is_ambiguous', 0.0))])

    return Data(x=x, edge_index=edge_index, y=y), node_to_idx, unique_nodes


def clean_graph(g: Data):
    for attr in ('node_to_idx', 'idx_to_node'):
        if not hasattr(g, attr):
            setattr(g, attr, {} if attr.endswith('_idx') else [])
    if g.x.size(1) != 768:
        raise ValueError(f"x has width {g.x.size(1)}; expected 768")
    return g


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
    idx_to_node,
    validate_alignment=True,
):
    question = " ".join(entry['processed_question_toks'])
    # ordered list of schema node names, should assure schema-node-embedding alignment
    # schema_nodes = list(node_to_idx.keys())
    schema_nodes = idx_to_node
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

class ShardedGraphDataset(Dataset):
    def __init__(self, index_path):
        self.paths = torch.load(index_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.load(self.paths[idx])


def load_or_build_graphs(
    data_base_dir,
    dataset_name,
    mode="train",
    used_coref=False,
    use_dependency=False,
    batch_size=32,
    overwrite=False,
    test=False,
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
    Returns:
        tuple:
            - loader (DataLoader): PyTorch Geometric DataLoader containing graph batches for training.
            - graph_dataset (list[Data]): List of PyG graph objects for individual dataset entries.
            - dataset (list[dict]): Preprocessed dataset entries containing questions and annotations.
            - tables (dict): Processed database schema information.
    """
    folder_name = "test-data-fitted" if test else "train-data-fitted"
    graph_dir = os.path.join(
        data_base_dir, "preprocessed_dataset", dataset_name, folder_name, f"{mode}_graphs"
    )
    index_path = os.path.join(graph_dir, "index.pt")

    dataset_file_path = os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, f"{mode}.pkl")
    tables_file_path = os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "tables.pkl")

    if not os.path.exists(index_path) or overwrite:
        print("Building graphs from scratch...")
        dataset, tables = generate_preprocessed_relational_data(
            data_base_dir, dataset_name, mode,
            used_coref,
            use_dependency,
            overwrite
        )

        linker_model, linker_tokenizer, linker_device = load_linker_model()
        bert_model, bert_tokenizer, bert_device = load_bert_encoder()

        os.makedirs(graph_dir, exist_ok=True)
        index_file = []

        for i, entry in enumerate(tqdm(dataset, desc="Creating and saving graphs")):
            db = tables[entry["db_id"]]
            graph, node_to_idx, idx_to_node = create_graph_from_schema(db, entry)

            node_embeddings = generate_node_embeddings(
                entry, db,
                node_to_idx,
                linker_model, linker_tokenizer, linker_device,
                linker_model, linker_tokenizer, linker_device, idx_to_node,
                validate_alignment=True,
            )

            graph.x = node_embeddings
            graph = clean_graph(graph)
            graph.example_index = i  # to link with dataset entry

            graph_path = os.path.join(graph_dir, f"graph_{i}.pt")
            torch.save(graph, graph_path)
            index_file.append(graph_path)

        torch.save(index_file, index_path)
        print(f"Saved {len(index_file)} graphs to {graph_dir}")

    graph_dataset = ShardedGraphDataset(index_path)

    dataset = safe_pickle_load(dataset_file_path)
    tables = safe_pickle_load(tables_file_path)

    loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True)
    return loader, graph_dataset, dataset, tables
