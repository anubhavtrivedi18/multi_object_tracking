from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Folder structure
folders = [
    "data/raw",
    "data/processed",
    "data/videos",
    "data/tracking",
    "data/sequences",
    "models",
    "src",
    "static",
    "tests",
]

# Create folders
for folder in folders:
    folder_path = PROJECT_ROOT / folder
    folder_path.mkdir(parents=True, exist_ok=True)

# Create README.md
readme_path = PROJECT_ROOT / "README.md"
readme_path.touch(exist_ok=True)

print("Project structure initialized successfully.")
