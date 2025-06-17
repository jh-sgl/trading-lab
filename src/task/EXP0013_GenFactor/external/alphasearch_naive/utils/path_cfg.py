from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent

data_dir = project_root / 'data'
result_dir = project_root / 'results'

margin_rate_path = project_root.parent / 'Live4Common' / 'csv' / 'margin_rate.csv'