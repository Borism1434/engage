# scripts/script_helper.py
import sys
import os

def add_repo_root_to_syspath():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    print(f"Repo root added to sys.path: {repo_root}")