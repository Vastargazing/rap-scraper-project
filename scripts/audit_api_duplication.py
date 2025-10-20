#!/usr/bin/env python3
"""
🔍 FastAPI Duplication Audit Script
Анализирует все три API файла и показывает дублирование
"""

import os
import re
from pathlib import Path
from collections import defaultdict

# Пути к файлам
API_FILES = {
    "api.py": "/home/va/Documents/Github/rap_scraper/rap-scraper-project/api.py",
    "ml_api_service.py": "/home/va/Documents/Github/rap_scraper/rap-scraper-project/src/models/ml_api_service.py",
    "ml_api_service_v2.py": "/home/va/Documents/Github/rap_scraper/rap-scraper-project/src/api/ml_api_service_v2.py",
}


def extract_endpoints(file_path):
    """Извлеки все эндпоинты из FastAPI файла"""
    endpoints = []
    
    if not os.path.exists(file_path):
        return endpoints
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Найти все @app.METHOD("/path") паттерны
    pattern = r'@app\.(get|post|put|delete|patch)\("([^"]+)"'
    matches = re.findall(pattern, content)
    
    for method, path in matches:
        endpoints.append((method.upper(), path))
    
    return endpoints


def extract_imports(file_path):
    """Извлеки все импорты"""
    imports = []
    
    if not os.path.exists(file_path):
        return imports
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('from ') or line.startswith('import '):
            imports.append(line)
    
    return imports


def extract_functions(file_path):
    """Извлеки все функции"""
    functions = []
    
    if not os.path.exists(file_path):
        return functions
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Найти все async def и def
    pattern = r'(?:async\s+)?def\s+(\w+)\s*\('
    matches = re.findall(pattern, content)
    
    return list(set(matches))


def get_file_size(file_path):
    """Получи размер файла в строках"""
    if not os.path.exists(file_path):
        return 0
    
    with open(file_path, 'r') as f:
        return len(f.readlines())


def main():
    print("\n" + "="*80)
    print("🔍 FastAPI DUPLICATION AUDIT REPORT")
    print("="*80 + "\n")
    
    # 1. Размеры файлов
    print("📊 FILE SIZES")
    print("-"*80)
    total_lines = 0
    for name, path in API_FILES.items():
        lines = get_file_size(path)
        total_lines += lines
        print(f"  {name:30} {lines:5} lines")
    print(f"  {'TOTAL':30} {total_lines:5} lines")
    print()
    
    # 2. Эндпоинты
    print("🎯 ENDPOINTS ANALYSIS")
    print("-"*80)
    
    all_endpoints = {}
    for name, path in API_FILES.items():
        endpoints = extract_endpoints(path)
        all_endpoints[name] = endpoints
        
        print(f"\n  {name}:")
        for method, endpoint in endpoints:
            print(f"    {method:6} {endpoint}")
    
    # Найти дублирование
    print("\n\n⚠️ DUPLICATED ENDPOINTS:")
    print("-"*80)
    
    endpoint_map = defaultdict(list)
    for file_name, endpoints in all_endpoints.items():
        for method, path in endpoints:
            endpoint_map[f"{method} {path}"].append(file_name)
    
    duplicates = {k: v for k, v in endpoint_map.items() if len(v) > 1}
    
    if duplicates:
        for endpoint, files in sorted(duplicates.items()):
            print(f"\n  {endpoint}")
            for file in files:
                print(f"    ✓ {file}")
    else:
        print("\n  ✅ Нет дублирования эндпоинтов!")
    
    # 3. Импорты
    print("\n\n📦 DEPENDENCIES ANALYSIS")
    print("-"*80)
    
    all_imports = {}
    for name, path in API_FILES.items():
        imports = extract_imports(path)
        all_imports[name] = imports
        
        print(f"\n  {name}: {len(imports)} импортов")
        
        # Показать ключевые импорты
        key_imports = [imp for imp in imports if 'from src' in imp or 'from models' in imp]
        for imp in key_imports[:5]:
            print(f"    • {imp[:70]}")
        if len(key_imports) > 5:
            print(f"    ... и еще {len(key_imports)-5}")
    
    # 4. Функции
    print("\n\n🔧 FUNCTIONS ANALYSIS")
    print("-"*80)
    
    all_functions = {}
    for name, path in API_FILES.items():
        functions = extract_functions(path)
        all_functions[name] = functions
        
        print(f"\n  {name}: {len(functions)} функций")
        
        # Показать async функции (endpoints)
        async_funcs = [f for f in functions if f.startswith('async_') or any(
            x in f for x in ['root', 'health', 'analyze', 'batch', 'generate', 'predict']
        )]
        for func in async_funcs[:5]:
            print(f"    • {func}")
        if len(async_funcs) > 5:
            print(f"    ... и еще {len(async_funcs)-5}")
    
    # Найти дублирование функций
    print("\n\n⚠️ DUPLICATED FUNCTIONS:")
    print("-"*80)
    
    func_map = defaultdict(list)
    for file_name, functions in all_functions.items():
        for func in functions:
            func_map[func].append(file_name)
    
    dup_funcs = {k: v for k, v in func_map.items() if len(v) > 1}
    
    if dup_funcs:
        for func, files in sorted(dup_funcs.items()):
            print(f"\n  {func}()")
            for file in files:
                print(f"    ✓ {file}")
    else:
        print("\n  ✅ Нет дублирования функций!")
    
    # 5. Итоговая статистика
    print("\n\n📈 SUMMARY")
    print("-"*80)
    print(f"\n  Total Files: 3")
    print(f"  Total Lines: {total_lines}")
    print(f"  Duplicated Endpoints: {len(duplicates)}")
    print(f"  Duplicated Functions: {len(dup_funcs)}")
    print(f"\n  ⚠️ RECOMMENDATION: Consolidate into single src/api/main.py")
    print(f"     - Choose ml_api_service_v2.py as base (type-safe config)")
    print(f"     - Add routes from ml_api_service.py (ML models)")
    print(f"     - Add routes from api.py (web interface, batch)")
    print(f"     - Organize as modular route files")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
