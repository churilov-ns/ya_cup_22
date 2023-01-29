import os
import shutil


__all__ = [
    'clear_dir',
]


def clear_dir(work_dir: str):
    for filename in os.listdir(work_dir):
        file_path = os.path.join(work_dir, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
