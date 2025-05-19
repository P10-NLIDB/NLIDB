import hydra
import torch
from itertools import product
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from interactive_graph.build_pyg_graph import load_or_build_graphs, create_graph_from_schema
from models.gnn_classifier import GNNClassifier, evaluate, train
from torch.utils.data import random_split
import pickle


def run_sweep(cfg: DictConfig, do_train: bool, out: str, test=False):
    # Hyperparameter grid
    hidden_dims = [32, 64, 128]
    num_layers_list = [2, 3]
    lrs = [1e-3, 1e-4]
    epoch_list = [10, 30, 50, 100]

    best_acc = 0.0
    best_config = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets once
   # loader_spider, graph_dataset_spider, _, _ = load_or_build_graphs(
   #     data_base_dir=cfg.data.base_dir,
   #     dataset_name=cfg.data.spider_dataset_name,
   #     mode=cfg.experiment.mode,
   #     batch_size=cfg.experiment.batch_size,
   #     overwrite=cfg.experiment.overwrite,
   #     used_coref=cfg.preprocessing.used_coref,
   #     use_dependency=cfg.preprocessing.use_dependency,
   # )
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
    if not do_train and not test:
        for hidden_dim, num_layers, lr, epochs in product(hidden_dims, num_layers_list, lrs, epoch_list):
            print(
                f"\n Trying config: hidden_dim={hidden_dim}, layers={num_layers}, lr={lr}, epochs={epochs}")

            # Define the sizes for training and evaluation
            train_size = int(0.8 * len(combined_dataset))  # 80% for training
            eval_size = len(combined_dataset) - \
                train_size  # 20% for evaluation
            # Split the dataset
            train_dataset, eval_dataset = random_split(
                combined_dataset, [train_size, eval_size])
            # Create DataLoaders
            train_loader = DataLoader(
                train_dataset, batch_size=cfg.experiment.batch_size, shuffle=True)
            eval_loader = DataLoader(
                eval_dataset, batch_size=cfg.experiment.batch_size, shuffle=False)

            model = GNNClassifier(in_dim=768, hidden_dim=hidden_dim,
                                  num_layers=num_layers).to(device)

            train(model, train_loader, epochs=epochs, lr=lr)
            acc = evaluate(model, eval_loader)

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
    elif test:
        loader = DataLoader(
            combined_dataset, batch_size=cfg.experiment.batch_size, shuffle=False
        )
        model = None
        with open(out, "rb") as fp:
            model = pickle.load(fp)
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
        acc = evaluate(model, eval_loader)
        with open(out, "wb") as fp:
            pickle.dump(model, fp)


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Starting experiment sweep: {cfg.experiment.name}")
    do_train, out = True, "./models/gnn_trained/model_out.pkl"
    run_sweep(cfg, True, out, False)
    # run_sweep(cfg, False, out, True)


if __name__ == "__main__":
    main()
