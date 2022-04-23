from .client import SemRushClient


def load_from_string(string: str,
                     default_database: str = "uk") -> SemRushClient:
    api_key = string.split(":")[1].strip()
    return SemRushClient(api_key=api_key,
                         default_database=default_database)


def load_from_file(file_path: str,
                   default_database: str = "uk"):
    with open(file_path, "r") as f:
        string = f.read()
    return load_from_string(string=string,
                            default_database=default_database)
