import torch
from build_graph import build_graph, create_edges, extract_schema, load_spider_data, load_spider_schema, preprocess_NL_question
from gat_training import GATSchemaLinker, train_gat


def get_relevant_schema(NL_question, database_schema, model, top_k=3):
    print(f"Running inference for: {NL_question}")
    NL_tokens = preprocess_NL_question(NL_question)
    schema_nodes = extract_schema(database_schema)
    edges = create_edges(NL_tokens, schema_nodes)

    data = build_graph(NL_tokens, schema_nodes, edges)
    model.eval()

    with torch.no_grad():
        embeddings = model(data)

    schema_embeddings = embeddings[len(NL_tokens):]
    query_embedding = embeddings[:len(NL_tokens)].mean(dim=0)
    scores = F.cosine_similarity(query_embedding, schema_embeddings)
    top_indices = scores.topk(top_k).indices.tolist()
    relevant_schema = [schema_nodes[i] for i in top_indices]

    print(f"Relevant Schema Elements: {relevant_schema}")
    return relevant_schema

if __name__ == "__main__":
    spider_db_dir = "./spider_data"

    schema_dict = load_spider_schema(spider_db_dir)
    dataset = load_spider_data(spider_db_dir)

    model = GATSchemaLinker(input_dim=128, hidden_dim=64, num_heads=4)
    trained_model = train_gat(model, dataset, schema_dict, epochs=3)
