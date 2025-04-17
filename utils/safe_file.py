import json
import pickle
import portalocker


def safe_pickle_load(file_path):
    with open(file_path, "rb") as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        data = pickle.load(f)
        portalocker.unlock(f)
    return data

def safe_pickle_save(obj, file_path):
    with open(file_path, "wb") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        pickle.dump(obj, f)
        portalocker.unlock(f)

def safe_json_load(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        portalocker.lock(f, portalocker.LOCK_SH)
        data = json.load(f)
        portalocker.unlock(f)
    return data

def safe_json_save(obj, file_path, indent=2):
    with open(file_path, "w", encoding="utf-8") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        json.dump(obj, f, indent=indent)
        portalocker.unlock(f)