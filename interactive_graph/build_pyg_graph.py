import gc
import os
import random
from matplotlib import pyplot as plt
import networkx as nx
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

    outputs = bert_model(**inputs)
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


def create_graph_with_random_edges(db, entry, edge_multiplier=1.0):
    """
    edge_multiplier = total_edges / num_nodes
        - edge_multiplier = 0.5 → 1 edge per 2 nodes
        - edge_multiplier = 2.0 → 2 edges per node
    """
    raw_nodes = db['processed_table_names'] + db['processed_column_names']
    node_to_idx, unique_nodes = get_schema_node_map(db)

    num_nodes = len(unique_nodes)
    max_possible_edges = num_nodes * (num_nodes - 1)  # directed

    # Each edge counts as 2 for undirected-ness (both directions)
    desired_edge_count = int(edge_multiplier * num_nodes)
    desired_edge_count = min(desired_edge_count, max_possible_edges // 2)

    edge_set = set()
    while len(edge_set) < desired_edge_count:
        i = random.randint(0, num_nodes - 1)
        j = random.randint(0, num_nodes - 1)
        if i != j:
            edge_set.add((i, j))
            edge_set.add((j, i))  # bidirectional edge

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()

    x = torch.zeros((num_nodes, 1))
    y = torch.tensor([float(entry.get('is_ambiguous', 0.0))])
    return Data(x=x, edge_index=edge_index, y=y), node_to_idx, unique_nodes


@torch.no_grad()
def create_graph_from_schema(db, entry, bidirectional: bool, show_plot = False):
    raw_nodes = db['processed_table_names'] + db['processed_column_names']
    node_to_idx, unique_nodes = get_schema_node_map(db)

    edges = []
    for i, row in enumerate(db['relations']):
        for j, rel in enumerate(row):
            if (
                rel in ('none', '', None)
                or str(rel).startswith('question')
                or str(rel).endswith('-generic')
                or rel == 'column-column-sametable'
            ):
                continue

            ui = node_to_idx[raw_nodes[i]]
            uj = node_to_idx[raw_nodes[j]]

            if bidirectional:
                edges.append([ui, uj])
                edges.append([uj, ui])
            else:
                if rel in {
                    'table-column-has', 'table-column-pk',
                    'table-table-fk', 'column-column-fk'
                }:
                    edges.append([ui, uj])
                elif rel in {
                    'column-table-has', 'column-table-pk',
                    'table-table-fkr', 'column-column-fkr'
                }:
                    continue
                elif rel in {
                    'table-table-fkb', 'column-column-fkb'
                }:
                    edges.append([ui, uj])
                    edges.append([uj, ui])
                elif rel.endswith('-identity'):
                    if ui < uj:
                        edges.append([ui, uj])
                else:
                    edges.append([ui, uj])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    if edge_index.numel() and edge_index.max() >= len(unique_nodes):
        raise RuntimeError("ghost node still present – mapping bug")

    x = torch.zeros((len(unique_nodes), 1))  # placeholder node features
    y = torch.tensor([float(entry.get('is_ambiguous', 0.0))])


    if show_plot:
        G = nx.DiGraph() if not bidirectional else nx.Graph()
        G.add_nodes_from(range(len(unique_nodes)))
        G.add_edges_from(edge_index.t().tolist())

        # Define node colors: tables = red, columns = blue
        table_count = len(db['processed_table_names'])
        colors = ['red' if i < table_count else 'blue' for i in range(len(unique_nodes))]

        plt.figure(figsize=(12, 10))
        nx.draw_networkx(
            G,
            labels={i: name for i, name in enumerate(unique_nodes)},
            node_color=colors,
            font_size=8,
            node_size=800,
            arrows=not bidirectional,
            edge_color='gray'
        )
        plt.title("Schema Graph with Table (Red) and Column (Blue) Nodes")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return Data(x=x, edge_index=edge_index, y=y), node_to_idx, unique_nodes


@torch.no_grad()
def create_graph_from_schema_with_edge_type(db, entry, bidirectional: bool, show_plot=False):
    import networkx as nx
    import matplotlib.pyplot as plt

    raw_nodes = db['processed_table_names'] + db['processed_column_names']
    node_to_idx, unique_nodes = get_schema_node_map(db)

    edge_tuples = []
    edge_type_to_idx = {}

    for i, row in enumerate(db['relations']):
        for j, rel in enumerate(row):
            if (
                rel in ('none', '', None)
                or rel.startswith('question')
                or rel.endswith('-generic')
                or rel == 'column-column-sametable'
            ):
                continue

            if raw_nodes[i] not in node_to_idx or raw_nodes[j] not in node_to_idx:
                continue  # Skip ghost or unmatched nodes

            ui = node_to_idx[raw_nodes[i]]
            uj = node_to_idx[raw_nodes[j]]

            if bidirectional:
                edge_tuples.append((ui, uj, rel))
                edge_tuples.append((uj, ui, rel))
            else:
                if rel in {'table-column-has', 'table-column-pk', 'table-table-fk', 'column-column-fk'}:
                    edge_tuples.append((ui, uj, rel))
                elif rel in {'table-table-fkb', 'column-column-fkb'}:
                    edge_tuples.append((ui, uj, rel))
                    edge_tuples.append((uj, ui, rel))
                elif rel.endswith('-identity'):
                    if ui < uj:
                        edge_tuples.append((ui, uj, rel))
                elif rel in {'column-table-has', 'column-table-pk', 'table-table-fkr', 'column-column-fkr'}:
                    continue  # skip reverse-only relations
                else:
                    edge_tuples.append((ui, uj, rel))  # fallback

    edges = []
    edge_types = []

    for ui, uj, rel in edge_tuples:
        edges.append([ui, uj])
        if rel not in edge_type_to_idx:
            edge_type_to_idx[rel] = len(edge_type_to_idx)
        edge_types.append(edge_type_to_idx[rel])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_types, dtype=torch.long)

    if edge_index.numel() and edge_index.max() >= len(unique_nodes):
        raise RuntimeError("ghost node still present – mapping bug")

    x = torch.zeros((len(unique_nodes), 1))  # placeholder node features
    y = torch.tensor([float(entry.get('is_ambiguous', 0.0))])

    if show_plot:
        G = nx.DiGraph() if not bidirectional else nx.Graph()
        G.add_nodes_from(range(len(unique_nodes)))
        G.add_edges_from(edge_index.t().tolist())

        table_count = len(db['processed_table_names'])
        colors = ['red' if i < table_count else 'blue' for i in range(len(unique_nodes))]

        plt.figure(figsize=(12, 10))
        nx.draw_networkx(
            G,
            labels={i: name for i, name in enumerate(unique_nodes)},
            node_color=colors,
            font_size=8,
            node_size=800,
            arrows=not bidirectional,
            edge_color='gray'
        )
        plt.title("Schema Graph with Table (Red) and Column (Blue) Nodes")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y)
    return data, node_to_idx, unique_nodes

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
    data_base_dir: str,
    dataset_name: str,
    mode: str = "train",  # "train", "dev", "test"
    used_coref: bool = False,
    use_dependency: bool = False,
    batch_size: int = 32,
    overwrite: bool = False,
    bidirectional: bool = True,
    edge_type_aware: bool = False
) -> tuple:
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
    graph_dir_parts = [mode, "graphs"]
    if bidirectional:
        graph_dir_parts.append("bidirectional")
    else:
        graph_dir_parts.append("monodirectional")
    if edge_type_aware:
        graph_dir_parts.append("rel_type")

    graph_dir_name = "_".join(graph_dir_parts)
    graph_dir = os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, graph_dir_name)
    index_path = os.path.join(graph_dir, "index.pt")

    dataset_file_path = os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, f"{mode}.pkl")
    tables_file_path = os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "tables.pkl")

    if not os.path.exists(index_path) or overwrite:
        print(f"Building graphs from scratch for mode '{mode}' (bidirectional={bidirectional})...")
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
            if edge_type_aware:
                graph, node_to_idx, idx_to_node = create_graph_from_schema_with_edge_type(db, entry, bidirectional)
            else:
                graph, node_to_idx, idx_to_node = create_graph_from_schema(db, entry, bidirectional)
            node_embeddings = generate_node_embeddings(
                entry, db,
                node_to_idx,
                linker_model, linker_tokenizer, linker_device,
                bert_model, bert_tokenizer, bert_device, idx_to_node,
                validate_alignment=True,
            )

            graph.x = node_embeddings
            graph = clean_graph(graph)
            graph.example_index = i  # preserve link to dataset

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

def load_or_build_random_graphs(
    data_base_dir: str,
    dataset_name: str,
    mode: str = "train",
    edge_multiplier: float = 1.0,
    batch_size: int = 32,
    overwrite: bool = False
):
    em_str = str(edge_multiplier).replace('.', '_')
    graph_dir = os.path.join(
        data_base_dir,
        "preprocessed_dataset",
        dataset_name,
        f"{mode}_random_graphs_em_{em_str}" 
    )
    index_path = os.path.join(graph_dir, "index.pt")

    dataset_file_path = os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, f"{mode}.pkl")
    tables_file_path = os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "tables.pkl")

    if not os.path.exists(index_path) or overwrite:
        print(f"Building random-edge graphs (EM={edge_multiplier}) for mode '{mode}'...")
        dataset, tables = generate_preprocessed_relational_data(
            data_base_dir, dataset_name, mode,
            used_coref=False, use_dependency=False, overwrite=overwrite
        )

        linker_model, linker_tokenizer, linker_device = load_linker_model()
        bert_model, bert_tokenizer, bert_device = load_bert_encoder()

        os.makedirs(graph_dir, exist_ok=True)
        index_file = []

        for i, entry in enumerate(tqdm(dataset, desc="Creating random graphs")):
            db = tables[entry["db_id"]]
            graph, node_to_idx, idx_to_node = create_graph_with_random_edges(db, entry, edge_multiplier)

            node_embeddings = generate_node_embeddings(
                entry, db,
                node_to_idx,
                linker_model, linker_tokenizer, linker_device,
                bert_model, bert_tokenizer, bert_device, idx_to_node,
                validate_alignment=True,
            )

            graph.x = node_embeddings
            graph = clean_graph(graph)
            graph.example_index = i

            graph_path = os.path.join(graph_dir, f"graph_{i}.pt")
            torch.save(graph, graph_path)
            index_file.append(graph_path)

        torch.save(index_file, index_path)
        print(f"Saved {len(index_file)} random-edge graphs to {graph_dir}")

    graph_dataset = ShardedGraphDataset(index_path)
    dataset = safe_pickle_load(dataset_file_path)
    tables = safe_pickle_load(tables_file_path)

    loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True)
    return loader, graph_dataset, dataset, tables
