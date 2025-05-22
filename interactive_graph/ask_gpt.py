import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_client():
    load_dotenv()

    api_version = os.getenv("api_version")
    endpoint = os.getenv("endpoint")
    subscription_key = os.getenv("subscription_key")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )

    deployment = os.getenv("deployment")
    return client, deployment

def ask_questions_sequentially(client, deployment, questions: list[str]) -> list[int]:
    responses = []
    for question in questions:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. The question that the user asks you "
                        "is a question that an NL2SQL system might know how to answer. "
                        "Your job is to identify if the question is ambiguous or not. "
                        "Answer either yes or no."
                    )
                },
                {"role": "user", "content": question}
            ],
            max_completion_tokens=50,
            temperature=1.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment
        )
        answer = response.choices[0].message.content.strip().lower()
        if answer.startswith("yes"):
            responses.append(1)
        elif answer.startswith("no"):
            responses.append(0)
        else:
            responses.append(-1)  # Unknown/uninterpretable
        print(f"Q: {question}\nA: {answer}\n")
    return responses

def evaluate_llm_metrix(predictions: list[int], ground_truths: list[int]) -> None:
    # Filter out invalid predictions
    valid = [(p, gt) for p, gt in zip(predictions, ground_truths) if p in [0, 1]]
    if not valid:
        print("No valid predictions to evaluate.")
        return

    y_pred, y_true = zip(*valid)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"LLM | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    return accuracy, precision, recall, f1

def evaluate_llm_on_graph_dataset(client, deployment, eval_entries):
    """
    Evaluates an LLM on a list of dataset entries.

    Args:
        client: Azure/OpenAI client.
        deployment: Deployment name or ID for the LLM.
        eval_entries: List of dataset entries (dicts) to evaluate.
    """
    questions = []
    ground_truths = []

    for entry in eval_entries:
        question = " ".join(entry["processed_question_toks"])
        label = int(entry.get("is_ambiguous", 0.0))
        questions.append(question)
        ground_truths.append(label)

    print(f"\nEvaluating LLM on {len(questions)} examples...")
    predictions = ask_questions_sequentially(client, deployment, questions)
    evaluate_llm_metrix(predictions, ground_truths)


if __name__ == "__main__":
    client, deployment = get_client()
    
    questions = [
        "Is Sigurd goated?",
        "Which products had the highest sales last year?",
        "What is the name?",
        "Which departments sold the most units in Q3?",
    ]

    ground_truths = [1, 0, 1, 0]

    predictions = ask_questions_sequentially(client, deployment, questions)
    evaluate_llm_metrix(predictions, ground_truths)
