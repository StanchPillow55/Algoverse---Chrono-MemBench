#!/usr/bin/env python3
"""
Find orphaned Python and Jupyter notebook files that are not referenced 
by tests, documentation, or main application code.
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import Set, List, Dict


def find_python_imports(file_path: Path) -> Set[str]:
    """Extract all import statements from a Python file."""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to find imports
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
                        # Also add submodule imports
                        for alias in node.names:
                            imports.add(f"{node.module}.{alias.name}")
        except SyntaxError:
            # Fall back to regex parsing for files with syntax issues
            pass
        
        # Also use regex to catch dynamic imports and string references
        import_patterns = [
            r'import\s+([a-zA-Z_][a-zA-Z0-9_\.]*)',
            r'from\s+([a-zA-Z_][a-zA-Z0-9_\.]*)\s+import',
            r'__import__\([\'"]([^\'\"]+)[\'"]',
            r'importlib\.import_module\([\'"]([^\'\"]+)[\'"]'
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content)
            imports.update(matches)
            
    except Exception as e:
        print(f"Warning: Could not parse {file_path}: {e}")
    
    return imports


def find_string_references(file_path: Path, target_modules: Set[str]) -> Set[str]:
    """Find string references to target modules in file content."""
    references = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for module in target_modules:
            # Look for module references in strings, comments, etc.
            module_parts = module.split('.')
            for part in module_parts:
                if len(part) > 3 and part in content:  # Avoid short false positives
                    references.add(module)
                    break
                    
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
    
    return references


def get_module_name_from_path(file_path: Path, project_root: Path) -> str:
    """Convert file path to Python module name."""
    rel_path = file_path.relative_to(project_root)
    
    # Remove .py extension
    if rel_path.suffix == '.py':
        rel_path = rel_path.with_suffix('')
    
    # Convert to module name
    parts = list(rel_path.parts)
    
    # Remove __init__ from module names
    if parts[-1] == '__init__':
        parts = parts[:-1]
    
    # Handle special directories
    module_name = '.'.join(parts)
    
    # Alternative module names
    alternatives = set()
    alternatives.add(module_name)
    
    # Add variations
    if 'src' in parts:
        src_idx = parts.index('src')
        if src_idx < len(parts) - 1:
            alt_parts = parts[src_idx + 1:]
            alternatives.add('.'.join(alt_parts))
    
    return alternatives


def find_all_python_files(project_root: Path) -> List[Path]:
    """Find all Python files in the project."""
    python_files = []
    
    for pattern in ['**/*.py', '**/*.ipynb']:
        python_files.extend(project_root.glob(pattern))
    
    # Filter out common directories to skip
    skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.venv', 'venv'}
    
    filtered_files = []
    for file_path in python_files:
        if not any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            filtered_files.append(file_path)
    
    return filtered_files


def analyze_project_imports(project_root: Path) -> Dict[str, Set[str]]:
    """Analyze all imports across the project."""
    
    all_files = find_all_python_files(project_root)
    all_imports = set()
    file_imports = {}
    
    print(f"🔍 Analyzing {len(all_files)} Python/Jupyter files...")
    
    for file_path in all_files:
        if file_path.suffix == '.py':
            imports = find_python_imports(file_path)
            all_imports.update(imports)
            file_imports[file_path] = imports
        elif file_path.suffix == '.ipynb':
            # For notebooks, we'll do a simpler text search
            file_imports[file_path] = set()
    
    return file_imports, all_imports


def find_orphaned_files(project_root: Path) -> List[Path]:
    """Find Python files that appear to be orphaned."""
    
    file_imports, all_imports = analyze_project_imports(project_root)
    all_python_files = [f for f in find_all_python_files(project_root) if f.suffix == '.py']
    
    # Build module name mapping
    module_to_file = {}
    file_to_modules = {}
    
    for file_path in all_python_files:
        module_names = get_module_name_from_path(file_path, project_root)
        file_to_modules[file_path] = module_names
        for module_name in module_names:
            module_to_file[module_name] = file_path
    
    # Find referenced modules
    referenced_modules = set()
    
    for file_path, imports in file_imports.items():
        for imp in imports:
            referenced_modules.add(imp)
            # Also add parent modules
            parts = imp.split('.')
            for i in range(1, len(parts)):
                referenced_modules.add('.'.join(parts[:i]))
    
    # Find files that are not referenced
    orphaned_files = []
    
    for file_path in all_python_files:
        module_names = file_to_modules[file_path]
        
        # Check if any of this file's module names are referenced
        is_referenced = False
        
        for module_name in module_names:
            if any(module_name in ref or ref.startswith(module_name + '.') 
                   for ref in referenced_modules):
                is_referenced = True
                break
        
        # Special cases - always keep these
        special_files = {
            'setup.py', 'conftest.py', '__init__.py', 'test_*.py', 
            '*_test.py', 'manage.py', 'wsgi.py', 'asgi.py'
        }
        
        file_name = file_path.name
        is_special = any(
            file_name == special or 
            (special.startswith('*') and file_name.endswith(special[1:])) or
            (special.endswith('*') and file_name.startswith(special[:-1]))
            for special in special_files
        )
        
        # Keep files in certain directories
        keep_dirs = {'tests', 'scripts', 'docs', 'notebooks'}
        is_in_keep_dir = any(keep_dir in file_path.parts for keep_dir in keep_dirs)
        
        # Keep main entry points
        is_main_entry = any(part in file_path.parts for part in ['cli.py', 'main.py', '__main__.py'])
        
        if not is_referenced and not is_special and not is_in_keep_dir and not is_main_entry:
            orphaned_files.append(file_path)
    
    return orphaned_files


def main():
    """Main function to find and report orphaned files."""
    
    project_root = Path(__file__).parent.parent
    print(f"🏠 Project root: {project_root}")
    
    orphaned_files = find_orphaned_files(project_root)
    
    if orphaned_files:
        print(f"\n🗑️  Found {len(orphaned_files)} potentially orphaned files:")
        
        for file_path in sorted(orphaned_files):
            rel_path = file_path.relative_to(project_root)
            print(f"   • {rel_path}")
        
        print(f"\n📝 Recommendations:")
        print(f"   1. Review each file to confirm it's not needed")
        print(f"   2. Move truly orphaned files to legacy/")
        print(f"   3. Delete files that are definitely unused")
        print(f"   4. Update imports if files are actually used")
        
    else:
        print("\n✅ No orphaned files found!")
    
    # Also find notebooks
    notebooks = list(project_root.glob('**/*.ipynb'))
    if notebooks:
        print(f"\n📓 Found {len(notebooks)} Jupyter notebooks:")
        for nb in sorted(notebooks):
            rel_path = nb.relative_to(project_root)
            print(f"   • {rel_path}")


if __name__ == '__main__':
    main()
