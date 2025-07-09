import pandas as pd

def read_privacy_info_known_info_from_csv(file_path: str):
    df = pd.read_csv(file_path)
    privacy_info_list = df["privacy_info"].to_list()
    known_info_list = df["known_info"].to_list()

    # convert string to dict
    known_info_list = [eval(known_info) for known_info in known_info_list]
    privacy_info_list = [eval(privacy_info) for privacy_info in privacy_info_list]

    target_list = df["target"].to_list()
    return privacy_info_list, known_info_list, target_list