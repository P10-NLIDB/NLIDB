import os
import pickle
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from utils.safe_file import safe_pickle_load, safe_pickle_save

from .process_dataset import generate_preprocessed_relational_data


def build_pyg_graph(entry, db, mode='train', use_syntax=False, label=False):
    """
    Build a PyG graph from a single question+schema entry.
    
    :param entry: dict from preprocessing pipeline (includes schema_linking)
    :param db: dict from schema processing (includes relations)
    :param mode: key for schema_linking (e.g., 'train', 'dev', 'test')
    :param use_syntax: include dependency-based syntax edges (optional)
    :return: torch_geometric.data.Data
    """
    # ---- NODE FEATURES ----
    q_toks = entry['processed_question_toks']
    t_names = db['processed_table_names']
    c_names = db['processed_column_names']
    
    q_num = len(q_toks)
    t_num = len(t_names)
    c_num = len(c_names)

    # Node type one-hot: [question_token, table, column]
    question_nodes = torch.tensor([[1, 0, 0]] * q_num, dtype=torch.float)
    table_nodes = torch.tensor([[0, 1, 0]] * t_num, dtype=torch.float)
    column_nodes = torch.tensor([[0, 0, 1]] * c_num, dtype=torch.float)
    x = torch.cat([question_nodes, table_nodes, column_nodes], dim=0)  # shape [q + t + c, 3]

    # ---- EDGE INDEX ----
    edge_index = []

    # 1. Schema internal edges (tables + columns)
    schema_rel = db['relations']  # (t + c) x (t + c)
    schema_offset = q_num
    for i in range(len(schema_rel)):
        for j in range(len(schema_rel[0])):
            rel = schema_rel[i][j]
            if rel != 'none' and rel != "" and not rel.endswith("-generic"):
                edge_index.append([schema_offset + i, schema_offset + j])

    # 2. Question ↔ Schema links
    q_schema, schema_q = entry['schema_linking']  # q_num x (t + c), (t + c) x q_num
    for i in range(q_num):
        for j in range(t_num + c_num):
            rel = q_schema[i][j]
            if not rel.endswith("-nomatch") and not rel.endswith("-generic"):
                edge_index.append([i, q_num + j])
    for j in range(t_num + c_num):
        for i in range(q_num):
            rel = schema_q[j][i]
            if not rel.endswith("-nomatch") and not rel.endswith("-generic"):
                edge_index.append([q_num + j, i])

    # 3. Syntax tree edges (dependency resolution thingy)
    if use_syntax:
        tree_mat = entry['tree_relations']
        for i in range(len(tree_mat)):
            for j in range(len(tree_mat[i])):
                if tree_mat[i][j] != "None-Syntax":
                    edge_index.append([i, j])

    # Make undirected (for each edge, make it the other way aswell)
    edge_index += [[j, i] for i, j in edge_index]
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]

    # ---- LABEL ---- 1 if is_ambigous is found else set to 0
    y = torch.tensor([1] if label else [0], dtype=torch.float)
    # TODO: surely Martin will fix the dataset so every entry has this attribute:
    #y = torch.tensor([entry.get("is_ambiguous", 0)], dtype=torch.float)

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
    
        true_or_negative_label = True if dataset_name =="ambiQT" else False
        graph_dataset = [build_fn(entry, tables[entry["db_id"]], mode=mode, label=true_or_negative_label ) for entry in dataset]

        safe_pickle_save(graph_dataset, graph_file_path)
        print(f"Saved {len(graph_dataset)} graphs to {graph_file_path}")

    loader = DataLoader(graph_dataset, batch_size=batch_size, shuffle=True)
    return loader, graph_dataset, dataset, tables