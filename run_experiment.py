import hydra
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from interactive_graph.build_pyg_graph import build_pyg_graph, load_or_build_graphs
from models.gnn_classifier import GNNClassifier, evaluate, train

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Starting experiment: {cfg.experiment.name}")

    loader_spider, graph_dataset_spider, _, _ = load_or_build_graphs(
        data_base_dir=cfg.data.base_dir,
        dataset_name=cfg.data.spider_dataset_name,
        mode=cfg.experiment.mode,
        batch_size=cfg.experiment.batch_size,
        overwrite=cfg.experiment.overwrite,
        used_coref=cfg.preprocessing.used_coref,
        use_dependency=cfg.preprocessing.use_dependency,
        build_fn=build_pyg_graph
    )
    print(f"Loaded {len(graph_dataset_spider)} spider graphs. (non ambiguous)")

    loader_ambiQT, graph_dataset_ambiQT, _, _ = load_or_build_graphs(
        data_base_dir=cfg.data.base_dir,
        dataset_name=cfg.data.ambiqt_dataset_name,
        mode=cfg.experiment.mode,
        batch_size=cfg.experiment.batch_size,
        overwrite=cfg.experiment.overwrite,
        used_coref=cfg.preprocessing.used_coref,
        use_dependency=cfg.preprocessing.use_dependency,
        build_fn=build_pyg_graph
    )
    print(f"Loaded {len(graph_dataset_ambiQT)} ambiQT graphs. (ambiguous)")

    combined_dataset = graph_dataset_spider + graph_dataset_ambiQT
    combined_dataloader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

    print("Starting training...")
    model = GNNClassifier(in_dim=3, hidden_dim=64)
    train(model, combined_dataloader, epochs=10)
    evaluate(model, combined_dataloader) # Change to eval dataset

if __name__ == "__main__":
    main()
