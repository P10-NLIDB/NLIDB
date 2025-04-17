import sqlite3
import os, json, pickle, time
import numpy as np
import stanza
from tqdm import tqdm
from itertools import combinations, product
from nltk.corpus import stopwords
from utils.safe_file import safe_json_load, safe_pickle_save, safe_pickle_load

from .tree import Tree
from .constants import MAX_RELATIVE_DIST

nlp_tokenize = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse', tokenize_pretokenized = False, use_gpu=True)#, use_gpu=False)
stopwords = stopwords.words("english")


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def quote_normalization(dataset_name, data_idx, question):
    """ Normalize all usage of quotation marks into a separate \" """
    new_question, quotation_marks, change_marks = [], ["'", '"', '`', '‘', '’', '“', '”', "‘‘", "’’"], ['``', "''"]
    idx = 0
    while idx < len(question):
        tok=question[idx]
        if dataset_name == "sparc":
            tok = " ".join(tok.split("\ufeff")).strip()
        for mark in change_marks:
            tok = tok.replace(mark, "\"")
        if len(tok) > 2 and tok[0] in quotation_marks and tok[-1] in quotation_marks:
            new_question += [tok[0], tok[1:-1], tok[-1]]
        elif len(tok) > 2 and tok[0] in quotation_marks:
            new_question += [tok[0], tok[1:]]
        elif len(tok) > 2 and tok[-1] in quotation_marks:
            new_question += [tok[:-1], tok[-1]]
        elif tok in quotation_marks:
            new_question.append(tok)
        elif len(tok) == 2 and tok[0] in quotation_marks:
            # special case: the length of entity value is 1
            if idx + 1 < len(question) and question[idx + 1] in quotation_marks:
                new_question += [tok[0], tok[1]]
            else:
                new_question.append(tok)
        else:
            new_question.append(tok)
        idx += 1
    return new_question


def build_dependency_tree_matrix(doc, tree_mat):
    """ This is needed if we want to have dependency relations"""
    trees = dict()
    root = None
    
    root_list = []
    
    bias = 0
    for sent in doc.sentences: 
        for word in sent.words:
            tree = Tree()
            tree.idx = word.id -1 + bias 
            trees[tree.idx] = tree
        bias += len(sent.words)
    bias = 0
    for idx, sent in enumerate(doc.sentences): 
        # trees = trees_list[idx]
        for word in sent.words:
            head_id = word.head - 1 + bias
            word_id = word.id - 1 + bias
            if word.head - 1 == -1:
                root = trees[word_id]
                root_list.append([root, bias+len(sent.words)])
                continue
            # tree_mat[head_id, word.id-1] = word.deprel
            trees[head_id].add_child(trees[word_id])
            tree_mat[head_id, word_id] = "Forward-Syntax"
            tree_mat[word_id, head_id] = "Backward-Syntax"
        bias += len(sent.words)
    return root_list, tree_mat.tolist()


def normalize_and_build_schema_relations(db: dict):
    """ Tokenize, lemmatize, lowercase table and column names for database. Then build schema relations"""
    table_toks, table_names = [], []
    for tab in db['table_names']:
        doc = nlp_tokenize(tab)
        tab = [w.lemma.lower() for s in doc.sentences for w in s.words]
        table_toks.append(tab)
        table_names.append(" ".join(tab))
    db['processed_table_toks'], db['processed_table_names'] = table_toks, table_names
    column_toks, column_names = [], []
    for _, c in db['column_names']:
        doc = nlp_tokenize(c)
        c = [w.lemma.lower() for s in doc.sentences for w in s.words]
        column_toks.append(c)
        column_names.append(" ".join(c))
    db['processed_column_toks'], db['processed_column_names'] = column_toks, column_names
    column2table = list(map(lambda x: x[0], db['column_names'])) # from column id to table id
    table2columns = [[] for _ in range(len(table_names))] # from table id to column ids list
    for col_id, col in enumerate(db['column_names']):
        if col_id == 0: continue
        table2columns[col[0]].append(col_id)
    db['column2table'], db['table2columns'] = column2table, table2columns

    t_num, c_num, dtype = len(db['table_names']), len(db['column_names']), '<U100'

    # relations in tables, tab_num * tab_num
    tab_mat = np.array([['table-table-generic'] * t_num for _ in range(t_num)], dtype=dtype)
    table_fks = set(map(lambda pair: (column2table[pair[0]], column2table[pair[1]]), db['foreign_keys']))
    for (tab1, tab2) in table_fks:
        if (tab2, tab1) in table_fks:
            tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'table-table-fkb', 'table-table-fkb'
        else:
            tab_mat[tab1, tab2], tab_mat[tab2, tab1] = 'table-table-fk', 'table-table-fkr'
    tab_mat[list(range(t_num)), list(range(t_num))] = 'table-table-identity'

    # relations in columns, c_num * c_num
    col_mat = np.array([['column-column-generic'] * c_num for _ in range(c_num)], dtype=dtype)
    for i in range(t_num):
        col_ids = [idx for idx, t in enumerate(column2table) if t == i]
        col1, col2 = list(zip(*list(product(col_ids, col_ids))))
        col_mat[col1, col2] = 'column-column-sametable'
    col_mat[list(range(c_num)), list(range(c_num))] = 'column-column-identity'
    if len(db['foreign_keys']) > 0:
        col1, col2 = list(zip(*db['foreign_keys']))
        col_mat[col1, col2], col_mat[col2, col1] = 'column-column-fk', 'column-column-fkr'
    col_mat[0, list(range(c_num))] = '*-column-generic'
    col_mat[list(range(c_num)), 0] = 'column-*-generic'
    col_mat[0, 0] = '*-*-identity'

    # relations between tables and columns, t_num*c_num and c_num*t_num
    tab_col_mat = np.array([['table-column-generic'] * c_num for _ in range(t_num)], dtype=dtype)
    col_tab_mat = np.array([['column-table-generic'] * t_num for _ in range(c_num)], dtype=dtype)
    cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), range(1, c_num))))) # ignore *
    col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'column-table-has', 'table-column-has'
    if len(db['primary_keys']) > 0:
        cols, tabs = list(zip(*list(map(lambda x: (x, column2table[x]), db['primary_keys']))))
        col_tab_mat[cols, tabs], tab_col_mat[tabs, cols] = 'column-table-pk', 'table-column-pk'
    col_tab_mat[0, list(range(t_num))] = '*-table-generic'
    tab_col_mat[list(range(t_num)), 0] = 'table-*-generic'

    relations = np.concatenate([
        np.concatenate([tab_mat, tab_col_mat], axis=1),
        np.concatenate([col_tab_mat, col_mat], axis=1)
    ], axis=0)
    db['relations'] = relations.tolist()
    return db
    

def preprocess_natural_language_question( entry: dict, dataset_name: str, data_idx: int):
    """ Tokenize, lemmatize, lowercase question"""
    question = " ".join(quote_normalization(dataset_name, data_idx, entry["question_toks"]))
    entry["processed_text_list"] = [question]
    question = question.strip()

    doc = nlp_tokenize(question)
    raw_toks = [w.text.lower() for s in doc.sentences for w in s.words]
    toks = [w.lemma.lower() for s in doc.sentences for w in s.words]
    entry[f'raw_question_toks'] = raw_toks
    entry[f'ori_toks'] = [w.text for s in doc.sentences for w in s.words]
    entry[f'processed_question_toks'] = toks
    # print(question, [w.text for s in doc.sentences for w in s.words])
    # TOOD: Pretty sure this is just used for coref, but for now we will keep it but just use 0 instead of turn
    entry["final_preprocessed_text_list"].append([0, [w.text for s in doc.sentences for w in s.words], len([w.text for s in doc.sentences for w in s.words])])
    # relations in questions, q_num * q_num
    q_num, dtype = len(toks), '<U100'
    if q_num <= MAX_RELATIVE_DIST + 1:
        dist_vec = ['question-question-dist' + str(i) if i != 0 else 'question-question-identity'
            for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)]
        starting = MAX_RELATIVE_DIST
    else:
        dist_vec = ['question-question-generic'] * (q_num - MAX_RELATIVE_DIST - 1) + \
            ['question-question-dist' + str(i) if i != 0 else 'question-question-identity' \
                for i in range(- MAX_RELATIVE_DIST, MAX_RELATIVE_DIST + 1, 1)] + \
                ['question-question-generic'] * (q_num - MAX_RELATIVE_DIST - 1)
        starting = q_num - 1
    q_mat = np.array([dist_vec[starting - i: starting - i + q_num] for i in range(q_num)], dtype=dtype)
    entry[f'relations'] = q_mat.tolist()

    tree_mat = np.array([["None-Syntax"] * q_num for _ in range(q_num)], dtype=dtype)
    root_list, tree_mat = build_dependency_tree_matrix(doc, tree_mat)
    entry[f'tree_relations'] = tree_mat
    return entry


def schema_linking(entry: dict, db: dict):
        """ Perform schema linking: both question and database need to be preprocessed """
        # Todo: Change the db_dir to actual db location once dataset is in. Kind of curesd that it is hard coded here
        db_dir = "data/original_dataset/spider/database"
        db_content = True
        raw_question_toks, question_toks = entry[f'raw_question_toks'], entry[f'processed_question_toks']
        table_toks, column_toks = db['processed_table_toks'], db['processed_column_toks']
        table_names, column_names = db['processed_table_names'], db['processed_column_names']
        q_num, t_num, c_num, dtype = len(question_toks), len(table_toks), len(column_toks), '<U100'

        # relations between questions and tables, q_num*t_num and t_num*q_num
        table_matched_pairs = {'partial': [], 'exact': []}
        q_tab_mat = np.array([['question-table-nomatch'] * t_num for _ in range(q_num)], dtype=dtype)
        tab_q_mat = np.array([['table-question-nomatch'] * q_num for _ in range(t_num)], dtype=dtype)
        max_len = max([len(t) for t in table_toks])
        index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
        index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
        for i, j in index_pairs:
            phrase = ' '.join(question_toks[i: j])
            if phrase in stopwords: continue
            for idx, name in enumerate(table_names):
                if phrase == name: # fully match will overwrite partial match due to sort
                    q_tab_mat[range(i, j), idx] = 'question-table-exactmatch'
                    tab_q_mat[idx, range(i, j)] = 'table-question-exactmatch'
                elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                    q_tab_mat[range(i, j), idx] = 'question-table-partialmatch'
                    tab_q_mat[idx, range(i, j)] = 'table-question-partialmatch'

        # relations between questions and columns
        column_matched_pairs = {'partial': [], 'exact': [], 'value': []}
        q_col_mat = np.array([['question-column-nomatch'] * c_num for _ in range(q_num)], dtype=dtype)
        col_q_mat = np.array([['column-question-nomatch'] * q_num for _ in range(c_num)], dtype=dtype)
        max_len = max([len(c) for c in column_toks])
        index_pairs = list(filter(lambda x: x[1] - x[0] <= max_len, combinations(range(q_num + 1), 2)))
        index_pairs = sorted(index_pairs, key=lambda x: x[1] - x[0])
        for i, j in index_pairs:
            phrase = ' '.join(question_toks[i: j])
            if phrase in stopwords: continue
            for idx, name in enumerate(column_names):
                if phrase == name: # fully match will overwrite partial match due to sort
                    q_col_mat[range(i, j), idx] = 'question-column-exactmatch'
                    col_q_mat[idx, range(i, j)] = 'column-question-exactmatch'
                elif (j - i == 1 and phrase in name.split()) or (j - i > 1 and phrase in name):
                    q_col_mat[range(i, j), idx] = 'question-column-partialmatch'
                    col_q_mat[idx, range(i, j)] = 'column-question-partialmatch'
        if db_content:
            db_file = os.path.join(db_dir, db['db_id'], db['db_id'] + '.sqlite')
            if not os.path.exists(db_file):
                raise ValueError('[ERROR]: database file %s not found ...' % (db_file))
            conn = sqlite3.connect(db_file)
            conn.text_factory = lambda b: b.decode(errors='ignore')
            conn.execute('pragma foreign_keys=ON')
            for i, (tab_id, col_name) in enumerate(db['column_names_original']):
                if i == 0 or 'id' in column_toks[i]: # ignore * and special token 'id'
                    continue
                tab_name = db['table_names_original'][tab_id]
                try:
                    cursor = conn.execute("SELECT DISTINCT \"%s\" FROM \"%s\";" % (col_name, tab_name))
                    cell_values = cursor.fetchall()
                    cell_values = [str(each[0]) for each in cell_values]
                    cell_values = [[str(float(each))] if is_number(each) else each.lower().split() for each in cell_values]
                except Exception as e:
                    print(e)
                for j, word in enumerate(raw_question_toks):
                    word = str(float(word)) if is_number(word) else word
                    for c in cell_values:
                        if word in c and 'nomatch' in q_col_mat[j, i] and word not in stopwords:
                            q_col_mat[j, i] = 'question-column-valuematch'
                            col_q_mat[i, j] = 'column-question-valuematch'
                            break
            conn.close()

        # two symmetric schema linking matrix: q_num x (t_num + c_num), (t_num + c_num) x q_num
        q_col_mat[:, 0] = 'question-*-generic'
        col_q_mat[0] = '*-question-generic'
        q_schema = np.concatenate([q_tab_mat, q_col_mat], axis=1)
        schema_q = np.concatenate([tab_q_mat, col_q_mat], axis=0)
        entry[f'schema_linking'] = (q_schema.tolist(), schema_q.tolist())
        return entry


def run_preprocessing_pipeline_on_entry(entry: dict, db: dict, dataset_name: str, data_idx: int):
    entry["final_preprocessed_text_list"] = []
    entry = preprocess_natural_language_question(entry, dataset_name, data_idx)
    entry = schema_linking(entry, db)
    return entry


def process_all_databases(tables_list, output_path=None):
    """ Processes all databases in the dataset """
    tables = {}
    for idx, each in tqdm(enumerate(tables_list)):
        tables[each['db_id']] = normalize_and_build_schema_relations(each)
    print('In total, process %d databases.' % (len(tables)))
    if output_path is not None:
        safe_pickle_save(tables, output_path)
    return tables


def process_dataset_entries(dataset, tables, dataset_name, mode, output_path_base=None, used_coref=False):
    processed_dataset = []
    # if dataset_name == "cosql":
    #     wfile = open(f"./dataset_files/original_dataset/cosql_dataset/sql_state_tracking/{mode}_text_list.txt", "w")
    if used_coref and not os.path.exists(os.path.join(output_path_base, f"{mode}_coref.json")):
        wfile = open(os.path.join(output_path_base, f"{mode}_text_list.txt"), "w")
    for idx, entry in tqdm(enumerate(dataset)):
        # if idx > 100:
        #     continue
        if dataset_name in ["spider", "ambiQT"]:
            entry = run_preprocessing_pipeline_on_entry(entry, tables[entry['db_id']], dataset_name, idx)
        elif dataset_name in ["cosql", "sparc"]:
            entry = run_preprocessing_pipeline_on_entry(entry, tables[entry['database_id']], dataset_name, idx)
        else:
            raise NotImplementedError
        if used_coref and not os.path.exists(os.path.join(output_path_base, f"{mode}_coref.json")):
            wfile.write(str(entry['final_preprocessed_text_list'])+"\n")
        processed_dataset.append(entry)
    safe_pickle_save(processed_dataset, os.path.join(output_path_base, f"{mode}.pkl"))
    return processed_dataset


def get_dataset_file_paths(data_base_dir, dataset_name, mode):
    """ Adapter to get all the dir information in one place"""

    db_dir = os.path.join(data_base_dir, "original_dataset", dataset_name, "database")
    table_data_path=os.path.join(data_base_dir, "original_dataset", dataset_name, "tables.json")
    table_out_path=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "tables.pkl")
    if mode == "train":
        if dataset_name == "spider":
            dataset_path = os.path.join(data_base_dir, "original_dataset", dataset_name, "train_spider.json")
        elif dataset_name == "cosql":
            db_dir = os.path.join(data_base_dir, "original_dataset", "cosql_dataset", "database")
            dataset_path = os.path.join(data_base_dir, "original_dataset", "cosql_dataset/sql_state_tracking/", "cosql_train.json")
            table_data_path = os.path.join(data_base_dir, "original_dataset", "cosql_dataset", "tables.json")
            
        elif dataset_name == "sparc":
            dataset_path = os.path.join(data_base_dir, "original_dataset", dataset_name, "train.json")
        elif dataset_name == "ambiQT":
            dataset_path = os.path.join(data_base_dir, "original_dataset", dataset_name, "ambiqt_data.json") # TODO: change to train later
        else:
            raise NotImplementedError
        # dataset_output_path_base=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "train.pkl")
    elif mode == "dev": 
        if dataset_name in ["spider", "sparc"] :
            dataset_path=os.path.join(data_base_dir, "original_dataset", dataset_name, "dev.json")
        elif dataset_name == "cosql":
            db_dir = os.path.join(data_base_dir, "original_dataset", "cosql_dataset", "database")
            dataset_path=os.path.join(data_base_dir, "original_dataset", "cosql_dataset/sql_state_tracking/", "cosql_dev.json")
            table_data_path=os.path.join(data_base_dir, "original_dataset", "cosql_dataset", "tables.json")
            
        else:
            raise NotImplementedError
        # dataset_output_path=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name, "dev.pkl")
    else:
        raise NotImplementedError
    dataset_output_path_base=os.path.join(data_base_dir, "preprocessed_dataset", dataset_name)
    if not os.path.exists(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name)):
        os.makedirs(os.path.join(data_base_dir, "preprocessed_dataset", dataset_name))
    return db_dir, table_data_path, table_out_path, dataset_path, dataset_output_path_base


def generate_preprocessed_relational_data(data_base_dir, dataset_name, mode, used_coref = False, use_dependency=False):
    """End to end handling of the preprocessing and relation generation"""
    db_dir, table_data_path, table_out_path, dataset_path, dataset_output_path_base = get_dataset_file_paths(data_base_dir, dataset_name, mode)
    
    print(f"Dataset name: {dataset_name}")
    print(f"Mode: {mode}")

    # Load or preprocessed tables
    if os.path.exists(table_out_path):
        print("Loading preprocessed tables from disk...")
        tables = safe_pickle_load(table_out_path)
    else:
        print("Preprocessing database schemas...")
        tables_list = safe_json_load(table_data_path)
        start_time = time.time()
        tables = process_all_databases(tables_list)
        print('Databases preprocessing costs %.4fs .' % (time.time() - start_time))
        safe_pickle_save(tables, table_out_path)

    # Load or preprocess dataset
    if os.path.exists(os.path.join(dataset_output_path_base, f"{mode}.pkl")):
        print("Loading preprocessed dataset from disk...")
        dataset = safe_pickle_load(os.path.join(dataset_output_path_base, f"{mode}.pkl"))
    else:
        print("Preprocessing dataset entries...")
        raw_dataset = safe_json_load(dataset_path)
        start_time = time.time()
        dataset = process_dataset_entries(raw_dataset, tables, dataset_name, mode, dataset_output_path_base, used_coref)
        print('Dataset preprocessing costs %.4fs .' % (time.time() - start_time))

    return dataset, tables
