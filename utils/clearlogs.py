import os, shutil

def clear_logs(log_path):
    log_files = os.listdir(log_path)

    for folder_name in log_files:
        folder = os.path.join(log_path, folder_name)
        shutil.rmtree(folder)
        print(f"Deleted: {folder}") 