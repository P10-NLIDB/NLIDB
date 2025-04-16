import hydra
from omegaconf import DictConfig

from interactive_graph.build_pyg_graph import build_pyg_graph, load_or_build_graphs
from models.gnn_classifier import GNNClassifier, evaluate, train

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Starting experiment: {cfg.experiment.name}")

    loader, graph_dataset, dataset, tables = load_or_build_graphs(
        data_base_dir=cfg.data.base_dir,
        dataset_name=cfg.data.dataset_name,
        mode=cfg.experiment.mode,
        batch_size=cfg.experiment.batch_size,
        overwrite=cfg.experiment.overwrite,
        used_coref=cfg.preprocessing.used_coref,
        use_dependency=cfg.preprocessing.use_dependency,
        build_fn=build_pyg_graph
    )

    print(f"Loaded {len(graph_dataset)} graphs.")
    for batch in loader:
        print("🔎 x:", batch.x.shape, "| edges:", batch.edge_index.shape, "| y:", batch.y.shape)
        break

    print("Starting training...")
    model = GNNClassifier(in_dim=3, hidden_dim=64)
    train(model, loader, epochs=10)
    evaluate(model, loader)

if __name__ == "__main__":
    main()
