from pathlib import Path
import shutil
import fnmatch

def freeze_code(source, dest, formats=['*.py', '*.yaml']):
    source_path = Path(source)
    dest_path = Path(dest)

    for file_path in source_path.rglob('*'):
        if file_path.is_file():
            for format in formats:
                if fnmatch.fnmatch(file_path.name, format):
                    dest_file_path = dest_path / file_path.relative_to(source_path)
                    dest_file_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_file_path)
                    break