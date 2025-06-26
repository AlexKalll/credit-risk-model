import os
from pathlib import Path

def create_project_structure(root_dir):
    """
    Creates the project structure for the credit risk modeling project.
    
    Args:
        root_dir (str): Path to the root directory where structure should be created
    """
    # Define the directory structure
    dirs = [
        ".github/workflows",
        "data/raw",
        "data/processed",
        "notebooks",
        "src/api",
        "tests",
    ]
    
    # Define files to create
    files = [
        "notebooks/01_data_exploration.ipynb",
        "notebooks/02_feature_engineering.ipynb",
        "notebooks/03_model_experimentation.ipynb",
        "notebooks/04_results_analysis.ipynb",
        "notebooks/README.md",
        "src/__init__.py",
        "src/data_processing.py",
        "src/train.py",
        "src/predict.py",
        "src/api/__init__.py",
        "src/api/main.py",
        "src/api/pydantic_models.py",
        "tests/__init__.py",
        "tests/test_data_processing.py",
        "tests/test_models.py",
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        ".gitignore",
        "README.md",
        ".github/workflows/ci.yml"
    ]
    
    # Create root directory if it doesn't exist
    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)
    print(f"\nCreating project structure at: {root_path.resolve()}\n")
    
    # Create all directories
    for dir_path in dirs:
        full_path = root_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {full_path}")
    
    # Create all files
    for file_path in files:
        full_path = root_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add basic content to key files
        if file_path.endswith("README.md"):
            with open(full_path, "w") as f:
                f.write("# Notebooks Directory\n\n")
                f.write("This directory contains exploratory notebooks for the credit risk modeling project.\n")
        elif file_path.endswith(".py"):
            with open(full_path, "w") as f:
                if file_path.endswith("__init__.py"):
                    f.write("# Package initialization\n")
                else:
                    f.write(f"# {full_path.name}\n")
                    f.write('"""\n')
                    f.write(f"Module for {full_path.stem.replace('_', ' ')}\n")
                    f.write('"""\n\n')
                    if "test_" in file_path:
                        f.write("import pytest\n\n")
        elif file_path.endswith("requirements.txt"):
            with open(full_path, "w") as f:
                f.write("# Project dependencies\n")
                f.write("pandas\nnumpy\nscikit-learn\nmlflow\nfastapi\nuvicorn\n")
                f.write("pytest\npython-dotenv\njupyter\nmatplotlib\nseaborn\n")
        elif file_path.endswith(".ipynb"):
            with open(full_path, "w") as f:
                f.write("{}")  # Empty JSON for Jupyter notebook
        else:
            full_path.touch()
        
        print(f"Created file: {full_path}")
    
    print("\nProject structure created successfully!")

if __name__ == "__main__":
    # Run from current directory
    create_project_structure(".")