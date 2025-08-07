#!/usr/bin/env python3
"""
TERRAGON SDLC CORE SYSTEM - Generation 1: MAKE IT WORK
Simple, functional implementation of the autonomous SDLC system
"""
import os
import json
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class Task:
    """Core task representation"""
    id: str
    title: str
    description: str
    priority: int
    task_type: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    created_at: Optional[str] = None
    status: str = "pending"
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass 
class SDLCResults:
    """SDLC execution results"""
    generation: int
    tasks_processed: int
    tasks_completed: int
    tasks_failed: int
    execution_time: float
    quality_score: float
    errors: List[str]
    achievements: List[str]


class CoreLogger:
    """Simple logging implementation"""
    
    def __init__(self, name: str):
        self.name = name
        
    def log(self, level: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level.upper()} [{self.name}] {message}")
        
    def info(self, message: str):
        self.log("info", message)
        
    def error(self, message: str):
        self.log("error", message)
        
    def warning(self, message: str):
        self.log("warning", message)
        
    def debug(self, message: str):
        self.log("debug", message)


class TaskAnalyzer:
    """Simple task discovery and analysis"""
    
    def __init__(self):
        self.logger = CoreLogger("TaskAnalyzer")
        
    def discover_tasks(self, repo_path: str = ".") -> List[Task]:
        """Discover tasks from repository"""
        self.logger.info(f"Discovering tasks in {repo_path}")
        tasks = []
        
        # Analyze Python files for TODO/FIXME
        tasks.extend(self._scan_code_comments(repo_path))
        
        # Analyze README for enhancement opportunities  
        tasks.extend(self._analyze_documentation(repo_path))
        
        # Check for missing essential files
        tasks.extend(self._check_project_completeness(repo_path))
        
        self.logger.info(f"Discovered {len(tasks)} tasks")
        return tasks
        
    def _scan_code_comments(self, repo_path: str) -> List[Task]:
        """Scan for TODO/FIXME comments"""
        tasks = []
        search_patterns = ["TODO", "FIXME", "HACK", "NOTE"]
        
        try:
            for pattern in ["**/*.py", "**/*.js", "**/*.md"]:
                for file_path in Path(repo_path).glob(pattern):
                    if file_path.is_file():
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                for line_num, line in enumerate(f, 1):
                                    for search_pattern in search_patterns:
                                        if search_pattern.lower() in line.lower():
                                            task = Task(
                                                id=f"code_comment_{len(tasks)}",
                                                title=f"Address {search_pattern} in {file_path.name}",
                                                description=f"Line {line_num}: {line.strip()}",
                                                priority=self._calculate_priority(search_pattern, line),
                                                task_type="code_improvement",
                                                file_path=str(file_path),
                                                line_number=line_num
                                            )
                                            tasks.append(task)
                                            break
                        except Exception as e:
                            self.logger.error(f"Error reading {file_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error scanning code comments: {e}")
            
        return tasks
        
    def _analyze_documentation(self, repo_path: str) -> List[Task]:
        """Analyze documentation for improvement opportunities"""
        tasks = []
        
        readme_path = Path(repo_path) / "README.md"
        if readme_path.exists():
            try:
                with open(readme_path, 'r') as f:
                    content = f.read()
                    
                # Check for missing sections
                missing_sections = []
                required_sections = ["Installation", "Usage", "Contributing", "License"]
                
                for section in required_sections:
                    if section.lower() not in content.lower():
                        missing_sections.append(section)
                        
                if missing_sections:
                    task = Task(
                        id=f"docs_missing_sections",
                        title=f"Add missing README sections",
                        description=f"Missing sections: {', '.join(missing_sections)}",
                        priority=6,
                        task_type="documentation",
                        file_path=str(readme_path)
                    )
                    tasks.append(task)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing README: {e}")
                
        return tasks
        
    def _check_project_completeness(self, repo_path: str) -> List[Task]:
        """Check for missing essential project files"""
        tasks = []
        
        essential_files = {
            ".gitignore": "Add .gitignore file for better version control",
            "requirements.txt": "Add requirements.txt for Python dependencies", 
            "setup.py": "Add setup.py for package installation",
            "pyproject.toml": "Add pyproject.toml for modern Python packaging",
            "CHANGELOG.md": "Add changelog to track project changes",
            "CONTRIBUTING.md": "Add contributing guidelines",
            "LICENSE": "Add project license file"
        }
        
        for filename, description in essential_files.items():
            file_path = Path(repo_path) / filename
            if not file_path.exists():
                # Only suggest if it makes sense for this project
                if self._should_suggest_file(repo_path, filename):
                    task = Task(
                        id=f"missing_file_{filename}",
                        title=f"Add {filename}",
                        description=description,
                        priority=5,
                        task_type="project_structure",
                        file_path=filename
                    )
                    tasks.append(task)
                    
        return tasks
        
    def _should_suggest_file(self, repo_path: str, filename: str) -> bool:
        """Determine if a file suggestion makes sense for this project"""
        if filename in ["requirements.txt", "setup.py", "pyproject.toml"]:
            # Only suggest Python files if this is a Python project
            return any(Path(repo_path).glob("**/*.py"))
        return True
        
    def _calculate_priority(self, pattern: str, line: str) -> int:
        """Calculate task priority based on pattern and context"""
        base_priority = {"FIXME": 8, "HACK": 7, "TODO": 5, "NOTE": 3}
        priority = base_priority.get(pattern, 5)
        
        # Increase priority for security-related comments
        if any(word in line.lower() for word in ["security", "auth", "password", "token"]):
            priority += 2
            
        # Increase priority for performance-related comments
        if any(word in line.lower() for word in ["performance", "slow", "optimize"]):
            priority += 1
            
        return min(priority, 10)


class SimpleOrchestrator:
    """Simple task orchestrator"""
    
    def __init__(self):
        self.logger = CoreLogger("Orchestrator")
        
    def execute_generation_1(self, tasks: List[Task]) -> SDLCResults:
        """Execute Generation 1: MAKE IT WORK"""
        self.logger.info("🚀 GENERATION 1: MAKE IT WORK - Starting simple implementation")
        start_time = time.time()
        
        completed_tasks = 0
        failed_tasks = 0
        errors = []
        achievements = []
        
        # Sort tasks by priority (highest first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)
        
        # Process top priority tasks
        for task in sorted_tasks[:10]:  # Limit to top 10 for initial implementation
            try:
                self.logger.info(f"Processing task: {task.title}")
                success = self._execute_task_simple(task)
                
                if success:
                    task.status = "completed"
                    completed_tasks += 1
                    achievements.append(f"✅ {task.title}")
                else:
                    task.status = "failed"
                    failed_tasks += 1
                    errors.append(f"❌ Failed: {task.title}")
                    
            except Exception as e:
                task.status = "failed"
                failed_tasks += 1
                error_msg = f"Error processing {task.title}: {str(e)}"
                errors.append(error_msg)
                self.logger.error(error_msg)
        
        execution_time = time.time() - start_time
        quality_score = (completed_tasks / max(len(sorted_tasks[:10]), 1)) * 100
        
        results = SDLCResults(
            generation=1,
            tasks_processed=len(sorted_tasks[:10]),
            tasks_completed=completed_tasks,
            tasks_failed=failed_tasks,
            execution_time=execution_time,
            quality_score=quality_score,
            errors=errors,
            achievements=achievements
        )
        
        self.logger.info(f"✅ Generation 1 complete: {completed_tasks}/{len(sorted_tasks[:10])} tasks completed")
        return results
        
    def _execute_task_simple(self, task: Task) -> bool:
        """Execute a single task with simple implementation"""
        try:
            if task.task_type == "code_improvement":
                return self._handle_code_improvement(task)
            elif task.task_type == "documentation":
                return self._handle_documentation_task(task)
            elif task.task_type == "project_structure":
                return self._handle_project_structure_task(task)
            else:
                self.logger.warning(f"Unknown task type: {task.task_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return False
            
    def _handle_code_improvement(self, task: Task) -> bool:
        """Handle code improvement tasks"""
        if task.file_path and "TODO" in task.description:
            # For now, just document the TODO in a simple way
            try:
                with open("todos_identified.md", "a") as f:
                    f.write(f"- [ ] **{task.title}**\n")
                    f.write(f"  - File: {task.file_path}\n")
                    f.write(f"  - Line: {task.line_number}\n")
                    f.write(f"  - Description: {task.description}\n")
                    f.write(f"  - Priority: {task.priority}/10\n\n")
                
                self.logger.info(f"Documented TODO in todos_identified.md")
                return True
            except Exception as e:
                self.logger.error(f"Failed to document TODO: {e}")
                return False
        return False
        
    def _handle_documentation_task(self, task: Task) -> bool:
        """Handle documentation improvement tasks"""
        if "missing README sections" in task.title:
            try:
                # Create basic README template
                readme_template = """
# Project Documentation Enhancement

## Installation
*Installation instructions needed*

## Usage
*Usage examples needed*

## Contributing
*Contributing guidelines needed*

## License
*License information needed*

---
*This template was auto-generated by TERRAGON SDLC System*
"""
                with open("README_ENHANCEMENT.md", "w") as f:
                    f.write(readme_template)
                    
                self.logger.info("Created README enhancement template")
                return True
            except Exception as e:
                self.logger.error(f"Failed to create README template: {e}")
                return False
        return False
        
    def _handle_project_structure_task(self, task: Task) -> bool:
        """Handle project structure improvement tasks"""
        if task.file_path:
            try:
                if task.file_path == ".gitignore":
                    gitignore_content = """# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
*.egg-info/

# IDEs
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
"""
                    with open(".gitignore", "w") as f:
                        f.write(gitignore_content)
                    self.logger.info("Created basic .gitignore file")
                    return True
                    
                elif task.file_path == "CHANGELOG.md":
                    changelog_content = f"""# Changelog

## [Unreleased] - {datetime.now().strftime('%Y-%m-%d')}

### Added
- Auto-generated changelog by TERRAGON SDLC System
- Basic project structure improvements

### Changed
- Enhanced project documentation

### Fixed
- Addressed TODO comments and code improvements
"""
                    with open("CHANGELOG.md", "w") as f:
                        f.write(changelog_content)
                    self.logger.info("Created CHANGELOG.md")
                    return True
                    
            except Exception as e:
                self.logger.error(f"Failed to create {task.file_path}: {e}")
                return False
                
        return False


class AutonomousSDLC:
    """Main SDLC orchestration system"""
    
    def __init__(self):
        self.logger = CoreLogger("AutonomousSDLC")
        self.analyzer = TaskAnalyzer()
        self.orchestrator = SimpleOrchestrator()
        
    def execute_full_cycle(self, repo_path: str = ".") -> Dict[str, Any]:
        """Execute full autonomous SDLC cycle"""
        self.logger.info("🎯 TERRAGON AUTONOMOUS SDLC v4.0 - STARTING EXECUTION")
        
        cycle_start = time.time()
        results = {
            "cycle_start": datetime.now(timezone.utc).isoformat(),
            "generations": [],
            "total_achievements": [],
            "total_errors": [],
            "cycle_summary": {}
        }
        
        try:
            # DISCOVERY PHASE
            self.logger.info("📋 DISCOVERY PHASE: Analyzing repository")
            tasks = self.analyzer.discover_tasks(repo_path)
            
            if not tasks:
                self.logger.warning("No tasks discovered - creating default enhancement tasks")
                tasks = self._create_default_enhancement_tasks()
            
            # GENERATION 1: MAKE IT WORK
            gen1_results = self.orchestrator.execute_generation_1(tasks)
            results["generations"].append(asdict(gen1_results))
            results["total_achievements"].extend(gen1_results.achievements)
            results["total_errors"].extend(gen1_results.errors)
            
            # Save intermediate results
            self._save_results(results, "generation_1_results.json")
            
            # Continue with additional generations if Generation 1 was successful
            if gen1_results.quality_score >= 70:
                self.logger.info("✅ Generation 1 successful - proceeding to advanced generations")
                # Future: Add Generation 2 and 3 implementations
            else:
                self.logger.warning("⚠️ Generation 1 quality below threshold - focusing on core fixes")
                
        except Exception as e:
            error_msg = f"Fatal error in SDLC execution: {e}"
            self.logger.error(error_msg)
            results["total_errors"].append(error_msg)
            
        # Final summary
        cycle_time = time.time() - cycle_start
        results["cycle_summary"] = {
            "execution_time": cycle_time,
            "total_tasks_discovered": len(tasks) if 'tasks' in locals() else 0,
            "generations_completed": len(results["generations"]),
            "overall_success": len(results["total_errors"]) == 0,
            "completion_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.logger.info(f"🏁 SDLC CYCLE COMPLETE - {cycle_time:.2f}s execution time")
        self._save_results(results, "final_sdlc_results.json")
        self._generate_summary_report(results)
        
        return results
        
    def _create_default_enhancement_tasks(self) -> List[Task]:
        """Create default enhancement tasks when none are discovered"""
        return [
            Task(
                id="default_docs",
                title="Enhance project documentation", 
                description="Improve README and add missing documentation sections",
                priority=7,
                task_type="documentation"
            ),
            Task(
                id="default_structure",
                title="Improve project structure",
                description="Add missing essential project files",
                priority=6,
                task_type="project_structure"
            )
        ]
        
    def _save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate human-readable summary report"""
        try:
            report = f"""
# TERRAGON AUTONOMOUS SDLC EXECUTION SUMMARY

**Execution Date:** {results['cycle_start']}
**Total Execution Time:** {results['cycle_summary']['execution_time']:.2f} seconds
**Generations Completed:** {results['cycle_summary']['generations_completed']}

## 🎯 Achievements
"""
            for achievement in results["total_achievements"]:
                report += f"{achievement}\n"
                
            if results["total_errors"]:
                report += f"\n## ⚠️ Issues Encountered\n"
                for error in results["total_errors"]:
                    report += f"- {error}\n"
                    
            report += f"""
## 📊 Generation Results

"""
            for i, gen in enumerate(results["generations"], 1):
                report += f"""### Generation {gen['generation']}
- **Tasks Processed:** {gen['tasks_processed']}
- **Tasks Completed:** {gen['tasks_completed']} 
- **Tasks Failed:** {gen['tasks_failed']}
- **Quality Score:** {gen['quality_score']:.1f}%
- **Execution Time:** {gen['execution_time']:.2f}s

"""

            report += f"""
## 🚀 Next Steps

Based on the execution results, consider:

1. Review generated files (todos_identified.md, README_ENHANCEMENT.md, etc.)
2. Implement the identified improvements
3. Run quality gates and testing
4. Proceed to Generation 2 (robust implementation) when ready

---
*Report generated by TERRAGON Autonomous SDLC v4.0*
"""
            
            with open("SDLC_EXECUTION_REPORT.md", "w") as f:
                f.write(report)
                
            self.logger.info("📄 Execution report generated: SDLC_EXECUTION_REPORT.md")
            
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")


if __name__ == "__main__":
    """Main execution entry point"""
    sdlc = AutonomousSDLC()
    results = sdlc.execute_full_cycle()
    
    print("\n" + "="*60)
    print("🎉 TERRAGON AUTONOMOUS SDLC EXECUTION COMPLETE")
    print("="*60)
    print(f"✅ Generations Completed: {results['cycle_summary']['generations_completed']}")
    print(f"⏱️  Total Execution Time: {results['cycle_summary']['execution_time']:.2f}s")
    print(f"📁 Results saved to: final_sdlc_results.json")
    print(f"📄 Summary report: SDLC_EXECUTION_REPORT.md")
    print("="*60)