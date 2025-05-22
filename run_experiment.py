import argparse
import hydra
import torch
from itertools import product
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from interactive_graph.ask_gpt import evaluate_llm_on_graph_dataset, get_client
from interactive_graph.build_pyg_graph import load_or_build_graphs, create_graph_from_schema
from interactive_graph.opensearch_preprocessing import run_opensearch_preprocessing_pipeline
from models.gnn_classifier import GNNClassifier, evaluate, get_prediction_results, train
from torch.utils.data import random_split
from utils.safe_file import safe_pickle_load, safe_pickle_save
from OpenSearchSQL.src.runner.run_manager import RunManager

def get_eval_question_entries(graph_dataset, original_dataset):
    """
    Extracts (db_id, question_id) pairs and original entries from graph_dataset using example_index.

    Args:
        graph_dataset (List[Data]): List of graph data objects with .example_index.
        original_dataset (List[Dict]): List of original dataset entries.

    Returns:
        Tuple[Set[Tuple[str, str]], List[Dict]]:
            - Set of (db_id, question_id) for evaluation.
            - List of corresponding original entries.
    """
    eval_ids = set()
    eval_entries = []

    for graph in graph_dataset:
        if not hasattr(graph, "example_index"):
            raise AttributeError("Each graph must have an 'example_index' linking it to the dataset.")
        entry = original_dataset[graph.example_index]
        db_id = entry["db_id"]
        question_id = entry["question_id"]
        eval_ids.add((db_id, question_id))
        eval_entries.append(entry)

    return eval_ids, eval_entries

def run_experiments(cfg: DictConfig, do_train: bool, out: str, test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### Run preprocessing for OpenSearch ###
    if cfg.preprocessing.opensearch:
        run_opensearch_preprocessing_pipeline()
    
    ### Load full graph dataset for train/eval ###
    _, graph_dataset, full_dataset, _ = load_or_build_graphs(
        data_base_dir=cfg.data.base_dir,
        dataset_name=cfg.data.ambiqt_dataset_name,
        mode=cfg.experiment.mode,
        batch_size=cfg.experiment.batch_size,
        overwrite=cfg.experiment.overwrite,
        used_coref=cfg.preprocessing.used_coref,
        use_dependency=cfg.preprocessing.use_dependency,
    )


    if test:
        # === Run in test mode ===
        loader = DataLoader(graph_dataset, batch_size=cfg.experiment.batch_size, shuffle=False)
        model = safe_pickle_load(out)
        acc = evaluate(model, loader)
        print(f"Test Accuracy: {acc:.4f}")
        return
    
    ### Prepare dataset splits ###
    total_size = len(graph_dataset)
    indices = list(range(total_size))
    torch.manual_seed(42)
    indices = torch.randperm(total_size).tolist()
    split_idx = int(0.8 * total_size)
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]

    train_graphs = [graph_dataset[i] for i in train_indices]
    eval_graphs = [graph_dataset[i] for i in eval_indices]

    train_entries = [full_dataset[i] for i in train_indices]
    eval_entries = [full_dataset[i] for i in eval_indices]

    train_loader = DataLoader(train_graphs, batch_size=cfg.experiment.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_graphs, batch_size=cfg.experiment.batch_size, shuffle=False)

    ### Train and evaluate GNN ###
    model = GNNClassifier(in_dim=768, hidden_dim=256, num_layers=2).to(device)
    train(model, train_loader, val_loader=eval_loader, epochs=200, lr=1e-2)
    model_acc, model_precision, model_recall, model_f1 = evaluate(model, eval_loader)
    safe_pickle_save(model, out)

    ### LLM ###
    client, deployment = get_client()
    evaluate_llm_on_graph_dataset(client, deployment, eval_entries)

    ### OpenSearch ###
    args = argparse.Namespace(
        data_mode="dev",
        db_root_path="Bird",
        pipeline_nodes="generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+candidate_generate+align_correct+vote+evaluation",
        pipeline_setup='{ "generate_db_schema": { "engine": "gpt-4.1", "bert_model": "all-MiniLM-L6-v2", "device":"cpu" }, "extract_col_value": { "engine": "gpt-4.1", "temperature":0.0 }, "extract_query_noun": { "engine": "gpt-4.1", "temperature":0.0 }, "column_retrieve_and_other_info": { "engine": "gpt-4.1", "bert_model": "all-MiniLM-L6-v2", "device":"cpu", "temperature":0.3, "top_k":10 }, "candidate_generate": { "engine": "gpt-4.1", "temperature":0.7, "n":21, "return_question":"True", "single":"False" }, "align_correct": { "engine": "gpt-4.1", "n":21, "bert_model": "all-MiniLM-L6-v2", "device":"cpu", "align_methods":"style_align+function_align+agent_align" } }',
        use_checkpoint=False,
        checkpoint_nodes=None,
        checkpoint_dir=None,
        log_level="warning",
    )
    print("=== Running OpenSearch without ambiguity detection ===")
    run_manager = RunManager(args)
    run_manager.initialize_tasks(0, len(eval_entries), eval_entries)
    run_manager.run_tasks()
    run_manager.generate_sql_files()

    print("=== Running OpenSearch with ambiguity detection ===")
    predicted_ambiguous_entries, predicted_unambiguous_entries = get_prediction_results(model, eval_loader, eval_entries)
    print(f"GNN predicted {len(predicted_ambiguous_entries)} ambiguous questions. Running OpenSearch with {len(predicted_unambiguous_entries)} / {len(eval_entries)} questions")
    run_manager = RunManager(args)
    run_manager.initialize_tasks(0, len(predicted_unambiguous_entries), predicted_unambiguous_entries)
    run_manager.run_tasks()
    run_manager.generate_sql_files()



@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Starting experiment: {cfg.experiment.name}")
    do_train, out = True, "./models/gnn_trained/model_out.pkl"
    run_experiments(cfg, True, out, False)


if __name__ == "__main__":
    main()
