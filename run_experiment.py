import argparse
from datetime import datetime
import json
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


def run_experiments(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## This should be run once on the environment!!
    if cfg.preprocessing.opensearch:
        run_opensearch_preprocessing_pipeline()
    
    _, ambiqt_graph_dataset, ambiqt_full_dataset, _ = load_or_build_graphs(
        data_base_dir=cfg.data.base_dir,
        dataset_name=cfg.data.ambiqt_dataset_name,
        mode=cfg.experiment.mode,
        batch_size=cfg.experiment.batch_size,
        overwrite=cfg.experiment.overwrite,
        used_coref=cfg.preprocessing.used_coref,
        use_dependency=cfg.preprocessing.use_dependency,
    )
    _, bird_graph_dataset, bird_full_dataset, _ = load_or_build_graphs(
        data_base_dir=cfg.data.base_dir,
        dataset_name=cfg.data.bird_dataset_name,
        mode=cfg.experiment.mode,
        batch_size=cfg.experiment.batch_size,
        overwrite=cfg.experiment.overwrite,
        used_coref=cfg.preprocessing.used_coref,
        use_dependency=cfg.preprocessing.use_dependency,
    )


    ambiqt_graph_eval_loader = DataLoader(ambiqt_graph_dataset, batch_size=cfg.experiment.batch_size, shuffle=False)
    bird_graph_eval_loader = DataLoader(bird_graph_dataset, batch_size=cfg.experiment.batch_size, shuffle=False)

    results = {}

    ## Trained on ambiqt
    print("Results for experiment 1 and 2:")
    print("ambiqt, bird")
    model = safe_pickle_load("models/model_out.pkl")
    acc, prec, rec, f1 = evaluate(model, ambiqt_graph_eval_loader)
    results['Trained on ambiqt - Evaluated on ambiqt'] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    acc, prec, rec, f1 = evaluate(model, bird_graph_eval_loader)
    results['Trained on ambiqt - Evaluated on bird'] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    ## LLM
    client, deployment = get_client()
    acc, prec, rec, f1 = evaluate_llm_on_graph_dataset(client, deployment, ambiqt_full_dataset)
    results['LLM - Evaluated on ambiqt'] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    acc, prec, rec, f1 = evaluate_llm_on_graph_dataset(client, deployment, bird_full_dataset)
    results['LLM - Evaluated on bird'] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    
    print("Results for experiment 3:")
    print("This model is also trained on ambrosia")
    print("ambiqt, bird")

    ## Trained on cross
    cross_trained_model = safe_pickle_load("models/model_out_with_ambrosia.pkl")
    acc, prec, rec, f1 = evaluate(cross_trained_model, ambiqt_graph_eval_loader)
    results['Cross trained - Evaluated on ambiqt'] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    acc, prec, rec, f1 = evaluate(cross_trained_model, bird_graph_eval_loader)
    results['Cross trained - Evaluated on bird'] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    ### OpenSearch ###
    args = argparse.Namespace(
        data_mode="dev",
        db_root_path="data/original_dataset/Bird",
        pipeline_nodes="generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+candidate_generate+align_correct+vote+evaluation",
        pipeline_setup='{ "generate_db_schema": { "engine": "gpt-4.1", "bert_model": "all-MiniLM-L6-v2", "device":"cpu" }, "extract_col_value": { "engine": "gpt-4.1", "temperature":0.0 }, "extract_query_noun": { "engine": "gpt-4.1", "temperature":0.0 }, "column_retrieve_and_other_info": { "engine": "gpt-4.1", "bert_model": "all-MiniLM-L6-v2", "device":"cpu", "temperature":0.3, "top_k":10 }, "candidate_generate": { "engine": "gpt-4.1", "temperature":0.7, "n":21, "return_question":"True", "single":"False" }, "align_correct": { "engine": "gpt-4.1", "n":21, "bert_model": "all-MiniLM-L6-v2", "device":"cpu", "align_methods":"style_align+function_align+agent_align" } }',
        use_checkpoint=False,
        checkpoint_nodes=None,
        checkpoint_dir=None,
        log_level="warning",
    )
    args.run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    print("=== Running OpenSearch without ambiguity detection on bird ===")
    run_manager = RunManager(args)
    run_manager.initialize_tasks(0, len(bird_full_dataset), bird_full_dataset)
    run_manager.run_tasks()
    summary_without_ambi = run_manager.statistics_manager.collect_statistics_summary()
    run_manager.generate_sql_files()

    print("=== Running OpenSearch with ambiguity detection on bird ===")
    predicted_ambiguous_entries, predicted_unambiguous_entries = get_prediction_results(model, bird_graph_dataset, bird_full_dataset)
    print(f"GNN predicted {len(predicted_ambiguous_entries)} ambiguous questions. Running OpenSearch with {len(predicted_unambiguous_entries)} / {len(eval_entries)} questions")
    run_manager = RunManager(args)
    run_manager.initialize_tasks(0, len(predicted_unambiguous_entries), predicted_unambiguous_entries)
    run_manager.run_tasks()
    summary_with_ambi = run_manager.statistics_manager.collect_statistics_summary()
    run_manager.generate_sql_files()

    results['OpenSearch - no prediction'] = summary_without_ambi['overall_accuracy']
    results['OpenSearch - with prediction'] = summary_with_ambi['overall_accuracy']
    
    print(results)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Starting experiment: {cfg.experiment.name}")
    do_train, out = True, "./models/gnn_trained/model_out.pkl"
    run_experiments(cfg)


if __name__ == "__main__":
    main()
