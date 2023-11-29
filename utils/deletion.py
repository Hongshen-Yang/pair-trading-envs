import os

def clear_previous_res(prefix):
    folder_path = f"result/gridsearch/"
    files = os.listdir(folder_path)

    for file in files:
        if file.startswith(prefix):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")