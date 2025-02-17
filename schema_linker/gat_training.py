import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from build_graph import build_graph, create_edges, extract_ground_truth_links, extract_schema, preprocess_NL_question

class GATSchemaLinker(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(GATSchemaLinker, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x


# TODO: Training loop is wrong but surely i will get there
def train_gat(model, dataset, schema_dict, epochs=50, lr=0.001):
    print("Training GAT model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # First, build the entire graph
    for NL_question, SQL_query, db_name in dataset[:100]:
        if db_name not in schema_dict:
            continue 

        schema = schema_dict[db_name]
        NL_tokens = preprocess_NL_question(NL_question)
        schema_nodes = extract_schema(schema)
        edges = create_edges(NL_tokens, schema_nodes)

        graph = build_graph(NL_tokens, schema_nodes, edges)
        ground_truth_schema = extract_ground_truth_links(SQL_query, schema)

        labels = torch.zeros(len(schema_nodes))
        for i, schema_node in enumerate(schema_nodes):
            if schema_node in ground_truth_schema:
                labels[i] = 1

        # Training loop
        total_loss = 0
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            output = model(graph)
            schema_embeddings = output[len(NL_tokens):]
            scores = torch.sigmoid(torch.matmul(schema_embeddings, output[:len(NL_tokens)].mean(dim=0)))

            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    print("Training complete!")
    return model
