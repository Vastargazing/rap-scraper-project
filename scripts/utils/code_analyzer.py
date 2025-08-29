#!/usr/bin/env python3
"""
Code Analysis Tool for Refactoring
Analyzes project structure, identifies large files, duplicated functions, and dependencies
"""

import os
import ast
import re
from pathlib import Path
from collections import defaultdict, Counter
import json
from datetime import datetime


class CodeAnalyzer:
    """Analyzes code structure for refactoring insights"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "file_sizes": [],
            "function_names": [],
            "import_dependencies": defaultdict(set),
            "large_files": [],
            "potential_duplicates": [],
            "refactoring_opportunities": []
        }
    
    def analyze_file_sizes(self):
        """Analyze file sizes to identify large files"""
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if any(part.startswith('.') for part in file_path.parts):
                continue  # Skip hidden directories
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                
                relative_path = str(file_path.relative_to(self.project_root))
                self.results["file_sizes"].append({
                    "file": relative_path,
                    "lines": lines,
                    "category": self._categorize_file(relative_path)
                })
                
                # Flag large files (>500 lines)
                if lines > 500:
                    self.results["large_files"].append({
                        "file": relative_path,
                        "lines": lines,
                        "refactor_priority": "high" if lines > 1000 else "medium"
                    })
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    def analyze_function_names(self):
        """Find function names and potential duplicates"""
        python_files = list(self.project_root.rglob("*.py"))
        function_counter = Counter()
        
        for file_path in python_files:
            if any(part.startswith('.') for part in file_path.parts):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple regex to find function definitions
                functions = re.findall(r'def (\w+)\s*\(', content)
                relative_path = str(file_path.relative_to(self.project_root))
                
                for func_name in functions:
                    function_counter[func_name] += 1
                    self.results["function_names"].append({
                        "function": func_name,
                        "file": relative_path
                    })
                    
            except Exception as e:
                print(f"Error analyzing functions in {file_path}: {e}")
        
        # Find potential duplicates (same function name in multiple files)
        for func_name, count in function_counter.items():
            if count > 1:
                files_with_func = [
                    item["file"] for item in self.results["function_names"] 
                    if item["function"] == func_name
                ]
                self.results["potential_duplicates"].append({
                    "function": func_name,
                    "count": count,
                    "files": files_with_func
                })
    
    def analyze_imports(self):
        """Analyze import dependencies"""
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if any(part.startswith('.') for part in file_path.parts):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                relative_path = str(file_path.relative_to(self.project_root))
                
                # Find imports
                import_pattern = r'^(?:from\s+(\S+)\s+import|import\s+(\S+))'
                imports = re.findall(import_pattern, content, re.MULTILINE)
                
                for from_module, import_module in imports:
                    module = from_module or import_module
                    if module:
                        self.results["import_dependencies"][relative_path].add(module)
                        
            except Exception as e:
                print(f"Error analyzing imports in {file_path}: {e}")
    
    def identify_refactoring_opportunities(self):
        """Identify specific refactoring opportunities"""
        opportunities = []
        
        # Large files
        for large_file in self.results["large_files"]:
            opportunities.append({
                "type": "large_file",
                "description": f"Large file: {large_file['file']} ({large_file['lines']} lines)",
                "recommendation": "Consider splitting into smaller modules",
                "priority": large_file["refactor_priority"],
                "file": large_file["file"]
            })
        
        # Duplicate function names
        for duplicate in self.results["potential_duplicates"]:
            if duplicate["count"] > 2:  # More than 2 occurrences
                opportunities.append({
                    "type": "function_duplication",
                    "description": f"Function '{duplicate['function']}' appears in {duplicate['count']} files",
                    "recommendation": "Consider creating a shared utility function",
                    "priority": "medium",
                    "files": duplicate["files"]
                })
        
        # Files with many imports (high coupling)
        for file_path, imports in self.results["import_dependencies"].items():
            if len(imports) > 15:
                opportunities.append({
                    "type": "high_coupling",
                    "description": f"File {file_path} has {len(imports)} imports",
                    "recommendation": "Consider reducing dependencies or splitting file",
                    "priority": "low",
                    "file": file_path
                })
        
        self.results["refactoring_opportunities"] = opportunities
    
    def _categorize_file(self, file_path: str) -> str:
        """Categorize file by purpose"""
        if 'analyzer' in file_path:
            return 'analyzer'
        elif 'scraper' in file_path:
            return 'scraper'
        elif 'enhancer' in file_path:
            return 'enhancer'
        elif 'cli' in file_path or file_path.endswith('_cli.py'):
            return 'cli'
        elif 'test' in file_path:
            return 'test'
        elif 'util' in file_path:
            return 'utility'
        else:
            return 'other'
    
    def generate_report(self) -> dict:
        """Generate comprehensive analysis report"""
        self.analyze_file_sizes()
        self.analyze_function_names()
        self.analyze_imports()
        self.identify_refactoring_opportunities()
        
        # Sort results
        self.results["file_sizes"].sort(key=lambda x: x["lines"], reverse=True)
        self.results["large_files"].sort(key=lambda x: x["lines"], reverse=True)
        self.results["potential_duplicates"].sort(key=lambda x: x["count"], reverse=True)
        
        # Summary statistics
        total_files = len(self.results["file_sizes"])
        total_lines = sum(item["lines"] for item in self.results["file_sizes"])
        avg_file_size = total_lines / total_files if total_files > 0 else 0
        
        self.results["summary"] = {
            "total_files": total_files,
            "total_lines": total_lines,
            "average_file_size": round(avg_file_size, 1),
            "large_files_count": len(self.results["large_files"]),
            "potential_duplicates_count": len(self.results["potential_duplicates"]),
            "refactoring_opportunities_count": len(self.results["refactoring_opportunities"])
        }
        
        return self.results
    
    def save_report(self, filename: str = "refactoring_analysis.json"):
        """Save analysis report to file"""
        # Convert defaultdict to regular dict for JSON serialization
        results_copy = dict(self.results)
        results_copy["import_dependencies"] = {
            k: list(v) for k, v in self.results["import_dependencies"].items()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis report saved to: {filename}")
        return filename
    
    def print_summary(self):
        """Print analysis summary"""
        summary = self.results.get("summary", {})
        
        print("\n" + "="*60)
        print("🔍 CODE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"📁 Total Python files: {summary.get('total_files', 0)}")
        print(f"📝 Total lines of code: {summary.get('total_lines', 0):,}")
        print(f"📊 Average file size: {summary.get('average_file_size', 0)} lines")
        
        print(f"\n🚨 REFACTORING OPPORTUNITIES:")
        print(f"   Large files (>500 lines): {summary.get('large_files_count', 0)}")
        print(f"   Potential duplicates: {summary.get('potential_duplicates_count', 0)}")
        print(f"   Total opportunities: {summary.get('refactoring_opportunities_count', 0)}")
        
        # Top 5 largest files
        print(f"\n📈 TOP 5 LARGEST FILES:")
        for i, file_info in enumerate(self.results["file_sizes"][:5]):
            print(f"   {i+1}. {file_info['file']}: {file_info['lines']} lines ({file_info['category']})")
        
        # High priority refactoring opportunities
        high_priority = [
            opp for opp in self.results["refactoring_opportunities"] 
            if opp.get("priority") == "high"
        ]
        
        if high_priority:
            print(f"\n🔥 HIGH PRIORITY REFACTORING:")
            for opp in high_priority[:3]:
                print(f"   • {opp['description']}")
                print(f"     → {opp['recommendation']}")


def main():
    """Main function"""
    analyzer = CodeAnalyzer()
    
    print("🚀 Starting code analysis for refactoring...")
    report = analyzer.generate_report()
    
    analyzer.print_summary()
    
    # Save detailed report
    report_file = analyzer.save_report("results/refactoring_analysis.json")
    
    print(f"\n✅ Analysis complete! Check {report_file} for details.")
    
    return report


if __name__ == "__main__":
    main()
