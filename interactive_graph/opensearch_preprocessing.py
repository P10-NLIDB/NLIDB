import os
import logging

from OpenSearchSQL.src.database_process.data_preprocess import bird_pre_process
from OpenSearchSQL.src.database_process.prepare_train_queries import prepare_train_queries
from OpenSearchSQL.src.database_process.generate_question import generate_questions_and_estimates
from OpenSearchSQL.src.database_process.make_emb import make_emb_all


def run_opensearch_preprocessing_pipeline():
    db_root_directory = "data/original_dataset/Bird"
    dev_json = "dev/dev.json"
    train_json = "train/train.json"
    dev_table = "dev/dev_tables.json"
    train_table = "train/train_tables.json"
    dev_database = "dev/dev_databases"
    fewshot_llm = "gpt-4.1"
    DAIL_SQL = "data/original_dataset/Bird/bird_dev.json"
    bert_model = "all-MiniLM-L6-v2"

    logging.info(f"Start data_preprocess, the output will be in {db_root_directory}/data_preprocess")
    bird_pre_process(
        bird_dir=db_root_directory,
        with_evidence=False,
        dev_json=dev_json,
        train_json=train_json,
        dev_table=dev_table,
        train_table=train_table
    )

    logging.info(f"Start prepare_train_queries, output will be in {db_root_directory}/llm_train_parse.json")
    llm_train_json = os.path.join(db_root_directory, 'llm_train_parse.json')
    prepare_train_queries(
        db_root_directory,
        llm_train_json,
        fewshot_llm,
        start=0,
        end=200
    )

    logging.info(f"Start generate_questions, output will be in {db_root_directory}/fewshot")
    generate_questions_and_estimates(
        db_root_directory,
        DAIL_SQL
    )

    logging.info(f"Start make_emb, output will be in {db_root_directory}/emb")
    make_emb_all(
        db_root_directory,
        dev_database,
        bert_model
    )

if __name__ == "__main__":
    run_opensearch_preprocessing_pipeline()