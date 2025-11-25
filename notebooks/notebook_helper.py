import sys
import os

def setup_repo_path():
    repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    print(f"Repo root added to sys.path: {repo_root}")