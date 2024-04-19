import os
import glob

def get_project_root() -> str:
    """
    Get the absolute path of the project root

    Returns
    -------

    """
    return os.path.join(os.path.dirname(__file__), ".")

def get_latest_run_id(log_path: str = "", log_name: str = "") -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, f"{glob.escape(log_name)}_[0-9]*")):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id
