import hydra
import torch
from itertools import product
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from interactive_graph.build_pyg_graph import load_or_build_graphs, build_pyg_graph
from models.gnn_classifier import GNNClassifier, evaluate, train

def run_sweep(cfg: DictConfig):
    # Hyperparameter grid
    hidden_dims = [32, 64, 128]
    num_layers_list = [2, 3]
    lrs = [1e-3, 1e-4]
    epoch_list = [10, 30, 50, 100]

    best_acc = 0.0
    best_config = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets once
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

    loader_ambiQT, graph_dataset_ambiQT, _, _ = load_or_build_graphs(
        data_base_dir=cfg.data.base_dir,
        dataset_name=cfg.data.ambiqt_dataset_name,
        mode=cfg.experiment.mode,
        batch_size=cfg.experiment.batch_size,
        overwrite=cfg.experiment.overwrite,
        used_coref=cfg.preprocessing.used_coref,
        use_dependency=cfg.preprocessing.use_dependency,
    )

    combined_dataset = graph_dataset_spider + graph_dataset_ambiQT

    for hidden_dim, num_layers, lr, epochs in product(hidden_dims, num_layers_list, lrs, epoch_list):
        print(f"\n Trying config: hidden_dim={hidden_dim}, layers={num_layers}, lr={lr}, epochs={epochs}")
        loader = DataLoader(combined_dataset, batch_size=cfg.experiment.batch_size, shuffle=True)
        
        model = GNNClassifier(in_dim=768, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

        train(model, loader, epochs=epochs, lr=lr)
        acc = evaluate(model, loader)

        if acc > best_acc:
            best_acc = acc
            best_config = {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "lr": lr,
                "epochs": epochs
            }

    print("\nBest config:")
    print(best_config)
    print(f"Best Accuracy: {best_acc:.4f}")


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Starting experiment sweep: {cfg.experiment.name}")
    run_sweep(cfg)

if __name__ == "__main__":
    main()