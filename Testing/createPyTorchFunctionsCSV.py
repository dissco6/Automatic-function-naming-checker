import os
import ast
import csv

MAX_FUNCTIONS = 5000

def extract_functions_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            source = f.read()
            tree = ast.parse(source)
        except SyntaxError:
            return []

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            try:
                func_code = ast.get_source_segment(source, node)
            except Exception:
                func_code = "<source not available>"
            functions.append((node.name, func_code))
    return functions

def find_python_files(base_path):
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith('.py'):
                yield os.path.join(root, file)

def extract_and_save_functions(base_dir, output_csv='functions.csv'):
    count = 0
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['function_name', 'function_code'])

        for file_path in find_python_files(base_dir):
            functions = extract_functions_from_file(file_path)
            for name, code in functions:
                writer.writerow([name, code])
                count += 1
                if count >= MAX_FUNCTIONS:
                    return

if __name__ == "__main__":
    extract_and_save_functions('./pytorch', 'open_source_func.csv')
