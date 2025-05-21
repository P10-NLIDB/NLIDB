import argparse
import hydra
import torch
from itertools import product
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from interactive_graph.ask_gpt import evaluate_llm_on_graph_dataset, get_client
from interactive_graph.build_pyg_graph import load_or_build_graphs, create_graph_from_schema
from models.gnn_classifier import GNNClassifier, evaluate, train
from torch.utils.data import random_split
from utils.safe_file import safe_pickle_load, safe_pickle_save
from OpenSearchSQL.src.runner.run_manager import RunManager

def run_experiments(cfg: DictConfig, do_train: bool, out: str, test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets once
    loader_ambiQT_val, graph_dataset_ambiQT_val, _, _ = load_or_build_graphs(
        data_base_dir=cfg.data.base_dir,
        dataset_name=cfg.data.ambiqt_dataset_name,
        mode=cfg.experiment.mode,
        batch_size=cfg.experiment.batch_size,
        overwrite=cfg.experiment.overwrite,
        used_coref=cfg.preprocessing.used_coref,
        use_dependency=cfg.preprocessing.use_dependency,
        test=not (cfg.test.do_run_test),
    )

    loader_ambiQT, graph_dataset_ambiQT, _, _ = load_or_build_graphs(
        data_base_dir=cfg.data.base_dir,
        dataset_name=cfg.data.ambiqt_dataset_name,
        mode=cfg.experiment.mode,
        batch_size=cfg.experiment.batch_size,
        overwrite=cfg.experiment.overwrite,
        used_coref=cfg.preprocessing.used_coref,
        use_dependency=cfg.preprocessing.use_dependency,
        test=cfg.test.do_run_test,
    )
    val_dataset = graph_dataset_ambiQT_val
    combined_dataset = graph_dataset_ambiQT
    if test:
        loader = DataLoader(
            combined_dataset, batch_size=cfg.experiment.batch_size, shuffle=False
        )
        model = None
        model = safe_pickle_load(out)
        acc = evaluate(model, loader)
    else:
        hidden_dim = 256
        num_layers = 2
        lr = 1e-2
        epochs = 200
        train_size = int(0.8 * len(combined_dataset))  # 80% for training
        eval_size = len(combined_dataset) - train_size  # 20% for evaluation
        # Split the dataset
        train_dataset, eval_dataset = random_split(
            combined_dataset, [train_size, eval_size])
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.experiment.batch_size, shuffle=True)
        eval_loader = DataLoader(
            val_dataset, batch_size=cfg.experiment.batch_size, shuffle=False)

        model = GNNClassifier(in_dim=768, hidden_dim=hidden_dim,
                              num_layers=num_layers).to(device)

        train(model, train_loader, val_loader=eval_loader, epochs=epochs, lr=lr)
        model_acc, model_precision, model_recall, model_f1 = evaluate(model, eval_loader)
        safe_pickle_save(model, out)

        ### LLM ###

        client, deployment = get_client()
        evaluate_llm_on_graph_dataset(client, deployment, graph_dataset_ambiQT_val, val_dataset)

        ### OpenSearch ###
        ## TODO: Make sure to also get the dataset and get the same splits as from the graphs
        args = argparse.Namespace(
            data_mode="dev",
            db_root_path="Bird",
            pipeline_nodes="generate_db_schema+extract_col_value+extract_query_noun+column_retrieve_and_other_info+candidate_generate+align_correct+vote+evaluation",
            pipeline_setup='{ "generate_db_schema": { "engine": "gpt-4.1", "bert_model": "all-MiniLM-L6-v2", "device":"cpu" }, "extract_col_value": { "engine": "gpt-4.1", "temperature":0.0 }, "extract_query_noun": { "engine": "gpt-4.1", "temperature":0.0 }, "column_retrieve_and_other_info": { "engine": "gpt-4.1", "bert_model": "all-MiniLM-L6-v2", "device":"cpu", "temperature":0.3, "top_k":10 }, "candidate_generate": { "engine": "gpt-4.1", "temperature":0.7, "n":21, "return_question":"True", "single":"False" }, "align_correct": { "engine": "gpt-4.1", "n":21, "bert_model": "all-MiniLM-L6-v2", "device":"cpu", "align_methods":"style_align+function_align+agent_align" } }',
            use_checkpoint=False,
            checkpoint_nodes=None,
            checkpoint_dir=None,
            log_level="warning",
            start=0,
            end=1
        )
        run_manager = RunManager(args)
        run_manager.initialize_tasks(args.start,args.end,dataset)
        run_manager.run_tasks()
        run_manager.generate_sql_files()       

        

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Starting experiment: {cfg.experiment.name}")
    do_train, out = True, "./models/gnn_trained/model_out.pkl"
    run_experiments(cfg, True, out, False)


if __name__ == "__main__":
    main()
