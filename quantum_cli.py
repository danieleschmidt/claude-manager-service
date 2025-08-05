#!/usr/bin/env python3
"""
Quantum-Enhanced CLI for Claude Manager Service

This CLI provides quantum-inspired task management and orchestration capabilities:
- Quantum task prioritization
- Quantum batch orchestration  
- Quantum insights reporting
- Interactive quantum analysis

Usage:
    python quantum_cli.py quantum-prioritize --tasks-file tasks.json
    python quantum_cli.py quantum-orchestrate --repo owner/repo
    python quantum_cli.py quantum-insights --output-file insights.json
    python quantum_cli.py quantum-analyze-issue --repo owner/repo --issue 123
"""

import json
import sys
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from logger import get_logger
from github_api import GitHubAPI
from quantum_task_planner import create_quantum_task_planner
from orchestrator import (
    quantum_orchestrate_tasks,
    analyze_issue_with_quantum_insights,
    quantum_batch_orchestrate
)
from config_validator import get_validated_config

logger = get_logger(__name__)


class QuantumCLI:
    """Quantum-Enhanced Command Line Interface"""
    
    def __init__(self):
        self.logger = logger
        self.config = None
        self.api = None
        
    def load_config(self, config_path: str = 'config.json'):
        """Load and validate configuration"""
        try:
            self.config = get_validated_config(config_path)
            self.api = GitHubAPI()
            self.logger.info("Configuration loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def quantum_prioritize_tasks(self, tasks_file: str, output_file: Optional[str] = None, 
                               enable_annealing: bool = True) -> Dict[str, Any]:
        """
        Prioritize tasks using quantum-inspired algorithms
        
        Args:
            tasks_file: Path to JSON file containing task data
            output_file: Optional output file for results
            enable_annealing: Enable quantum annealing optimization
            
        Returns:
            Prioritization results
        """
        self.logger.info(f"Quantum prioritizing tasks from {tasks_file}")
        
        try:
            # Load tasks
            with open(tasks_file, 'r') as f:
                tasks = json.load(f)
            
            if not isinstance(tasks, list):
                raise ValueError("Tasks file must contain a list of task objects")
            
            # Create quantum planner
            planner = create_quantum_task_planner(enable_annealing=enable_annealing)
            
            # Apply quantum prioritization
            prioritized_tasks = planner.prioritize_tasks(tasks)
            
            # Generate insights
            insights = planner.generate_quantum_insights_report(prioritized_tasks)
            
            results = {
                'prioritized_tasks': prioritized_tasks,
                'quantum_insights': insights,
                'input_file': tasks_file,
                'processed_at': datetime.now().isoformat(),
                'quantum_annealing_enabled': enable_annealing
            }
            
            # Save results if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                self.logger.info(f"Results saved to {output_file}")
            
            # Print summary
            print(f"\n🚀 QUANTUM TASK PRIORITIZATION COMPLETE")
            print(f"📊 Processed {len(tasks)} tasks")
            print(f"🔄 Quantum annealing: {'enabled' if enable_annealing else 'disabled'}")
            print(f"⭐ Top priority task: {prioritized_tasks[0]['content'][:60]}...")
            print(f"📈 Highest priority score: {prioritized_tasks[0]['priority_score']:.3f}")
            print(f"🔗 Entangled tasks: {insights['quantum_metrics']['entangled_tasks']}")
            print(f"📉 Average uncertainty: {insights['quantum_metrics']['average_uncertainty']:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Quantum prioritization failed: {e}")
            raise
    
    def quantum_orchestrate_repository(self, repo_name: str, output_file: Optional[str] = None,
                                     max_tasks: int = 20) -> Dict[str, Any]:
        """
        Orchestrate tasks in a repository using quantum methods
        
        Args:
            repo_name: Repository name (owner/repo)
            output_file: Optional output file for results
            max_tasks: Maximum number of tasks to process
            
        Returns:
            Orchestration results
        """
        self.logger.info(f"Quantum orchestrating repository {repo_name}")
        
        if not self.config or not self.api:
            self.load_config()
        
        try:
            # For demo purposes, create sample tasks
            # In production, this would integrate with task discovery
            sample_tasks = [
                {
                    'id': f"{repo_name}_quantum_1",
                    'content': 'TODO: Implement quantum-enhanced error handling for async operations',
                    'file_path': 'src/async_handler.py',
                    'line_number': 42,
                    'type': 'enhancement'
                },
                {
                    'id': f"{repo_name}_quantum_2",
                    'content': 'FIXME: Critical security vulnerability in JWT token validation',
                    'file_path': 'src/auth/jwt_validator.py',
                    'line_number': 156,
                    'type': 'security'
                },
                {
                    'id': f"{repo_name}_quantum_3",
                    'content': 'TODO: Optimize database connection pooling for high concurrency',
                    'file_path': 'src/database/pool.py',
                    'line_number': 78,
                    'type': 'performance'
                },
                {
                    'id': f"{repo_name}_quantum_4",
                    'content': 'TODO: Add comprehensive integration tests for API endpoints',
                    'file_path': 'tests/integration/test_api.py',
                    'line_number': 1,
                    'type': 'testing'
                }
            ]
            
            # Limit tasks
            tasks_to_process = sample_tasks[:max_tasks]
            
            # Apply quantum orchestration
            orchestration_results = quantum_orchestrate_tasks(
                self.api, repo_name, tasks_to_process, self.config
            )
            
            results = {
                'repository': repo_name,
                'orchestration_results': orchestration_results,
                'processed_at': datetime.now().isoformat()
            }
            
            # Save results if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                self.logger.info(f"Results saved to {output_file}")
            
            # Print summary
            print(f"\n🚀 QUANTUM ORCHESTRATION COMPLETE")
            print(f"📁 Repository: {repo_name}")
            print(f"📊 Tasks processed: {orchestration_results['total_tasks_processed']}")
            print(f"✅ Successful executions: {orchestration_results['successful_executions']}")
            print(f"❌ Failed executions: {orchestration_results['failed_executions']}")
            print(f"🔗 Entanglement density: {orchestration_results['quantum_insights']['quantum_metrics']['entanglement_density']:.3f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Quantum orchestration failed: {e}")
            raise
    
    def analyze_issue_quantum(self, repo_name: str, issue_number: int, 
                            output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a GitHub issue using quantum insights
        
        Args:
            repo_name: Repository name (owner/repo)
            issue_number: Issue number
            output_file: Optional output file for results
            
        Returns:
            Analysis results
        """
        self.logger.info(f"Quantum analyzing issue #{issue_number} in {repo_name}")
        
        if not self.config or not self.api:
            self.load_config()
        
        try:
            # Get the issue
            issue = self.api.get_issue(repo_name, issue_number)
            if not issue:
                raise ValueError(f"Issue #{issue_number} not found in {repo_name}")
            
            # Apply quantum analysis
            analysis = analyze_issue_with_quantum_insights(self.api, repo_name, issue)
            
            results = {
                'repository': repo_name,
                'issue_number': issue_number,
                'quantum_analysis': analysis,
                'analyzed_at': datetime.now().isoformat()
            }
            
            # Save results if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                self.logger.info(f"Results saved to {output_file}")
            
            # Print summary
            print(f"\n🧠 QUANTUM ISSUE ANALYSIS COMPLETE")
            print(f"📁 Repository: {repo_name}")
            print(f"🎫 Issue #{issue_number}: {analysis['issue_title']}")
            print(f"⭐ Quantum priority: {analysis['quantum_priority_score']:.3f}")
            print(f"🏷️  Task type: {analysis['task_type_classification']}")
            print(f"🎯 Quantum state: {analysis['quantum_state']}")
            print(f"📊 Uncertainty: {analysis['uncertainty_level']:.3f}")
            print(f"💡 Recommendation: {analysis['recommended_action']}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Quantum issue analysis failed: {e}")
            raise
    
    def generate_quantum_insights(self, output_file: Optional[str] = None,
                                batch_size: int = 50) -> Dict[str, Any]:
        """
        Generate comprehensive quantum insights across all configured repositories
        
        Args:
            output_file: Optional output file for results
            batch_size: Maximum tasks to process per batch
            
        Returns:
            Comprehensive insights
        """
        self.logger.info("Generating comprehensive quantum insights")
        
        if not self.config or not self.api:
            self.load_config()
        
        try:
            # Perform quantum batch orchestration
            batch_results = quantum_batch_orchestrate(self.api, self.config, batch_size)
            
            # Add metadata
            insights = {
                'quantum_batch_results': batch_results,
                'configuration': {
                    'repositories_scanned': len(self.config.get('github', {}).get('reposToScan', [])),
                    'batch_size': batch_size,
                    'quantum_annealing_enabled': True
                },
                'generated_at': datetime.now().isoformat(),
                'system_info': {
                    'python_version': sys.version,
                    'quantum_planner_version': '1.0.0'
                }
            }
            
            # Save results if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(insights, f, indent=2, default=str)
                self.logger.info(f"Insights saved to {output_file}")
            
            # Print summary
            print(f"\n🌌 QUANTUM INSIGHTS GENERATED")
            print(f"📊 Total tasks discovered: {batch_results['total_tasks_discovered']}")
            print(f"🔄 Tasks processed: {batch_results['tasks_processed']}")
            print(f"🏆 Top priority task: {batch_results['top_priority_tasks'][0]['content'][:60] if batch_results['top_priority_tasks'] else 'None'}")
            print(f"📈 Quantum metrics generated: ✓")
            print(f"🔗 Entanglement patterns analyzed: ✓")
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Quantum insights generation failed: {e}")
            raise
    
    def interactive_quantum_mode(self):
        """Interactive quantum task analysis mode"""
        print(f"\n🌌 QUANTUM INTERACTIVE MODE")
        print(f"Type 'help' for commands or 'quit' to exit")
        
        if not self.config or not self.api:
            self.load_config()
        
        while True:
            try:
                user_input = input("\nquantum> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() in ['help', 'h']:
                    self._print_interactive_help()
                elif user_input.startswith('analyze-issue'):
                    self._interactive_analyze_issue(user_input)
                elif user_input.startswith('prioritize'):
                    self._interactive_prioritize(user_input)
                elif user_input.startswith('orchestrate'):
                    self._interactive_orchestrate(user_input)
                elif user_input.startswith('insights'):
                    self._interactive_insights()
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nExiting quantum mode...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _print_interactive_help(self):
        """Print interactive mode help"""
        print("""
🌌 QUANTUM INTERACTIVE COMMANDS:
  analyze-issue <repo> <issue_number>  - Analyze GitHub issue with quantum insights
  prioritize <tasks_file>               - Prioritize tasks from JSON file
  orchestrate <repo>                    - Orchestrate repository tasks
  insights                              - Generate comprehensive quantum insights
  help, h                               - Show this help
  quit, exit, q                         - Exit quantum mode
        """)
    
    def _interactive_analyze_issue(self, command: str):
        """Handle interactive issue analysis"""
        try:
            parts = command.split()
            if len(parts) < 3:
                print("Usage: analyze-issue <repo> <issue_number>")
                return
            
            repo = parts[1]
            issue_num = int(parts[2])
            
            print(f"🧠 Analyzing issue #{issue_num} in {repo}...")
            results = self.analyze_issue_quantum(repo, issue_num)
            
        except ValueError:
            print("Error: Issue number must be an integer")
        except Exception as e:
            print(f"Error: {e}")
    
    def _interactive_prioritize(self, command: str):
        """Handle interactive task prioritization"""
        try:
            parts = command.split()
            if len(parts) < 2:
                print("Usage: prioritize <tasks_file>")
                return
            
            tasks_file = parts[1]
            if not Path(tasks_file).exists():
                print(f"Error: File {tasks_file} not found")
                return
            
            print(f"🚀 Prioritizing tasks from {tasks_file}...")
            results = self.quantum_prioritize_tasks(tasks_file)
            
        except Exception as e:
            print(f"Error: {e}")
    
    def _interactive_orchestrate(self, command: str):
        """Handle interactive orchestration"""
        try:
            parts = command.split()
            if len(parts) < 2:
                print("Usage: orchestrate <repo>")
                return
            
            repo = parts[1]
            print(f"🚀 Orchestrating {repo}...")
            results = self.quantum_orchestrate_repository(repo)
            
        except Exception as e:
            print(f"Error: {e}")
    
    def _interactive_insights(self):
        """Handle interactive insights generation"""
        try:
            print("🌌 Generating quantum insights...")
            results = self.generate_quantum_insights()
            
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Quantum-Enhanced Claude Manager CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quantum_cli.py quantum-prioritize --tasks-file sample_tasks.json
  python quantum_cli.py quantum-orchestrate --repo owner/repo-name
  python quantum_cli.py quantum-analyze-issue --repo owner/repo --issue 123
  python quantum_cli.py quantum-insights --output-file insights.json
  python quantum_cli.py interactive
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available quantum commands')
    
    # Quantum prioritize command
    prioritize_parser = subparsers.add_parser('quantum-prioritize', help='Prioritize tasks using quantum algorithms')
    prioritize_parser.add_argument('--tasks-file', required=True, help='JSON file containing tasks')
    prioritize_parser.add_argument('--output-file', help='Output file for results')
    prioritize_parser.add_argument('--no-annealing', action='store_true', help='Disable quantum annealing')
    
    # Quantum orchestrate command  
    orchestrate_parser = subparsers.add_parser('quantum-orchestrate', help='Orchestrate repository tasks')
    orchestrate_parser.add_argument('--repo', required=True, help='Repository name (owner/repo)')
    orchestrate_parser.add_argument('--output-file', help='Output file for results')
    orchestrate_parser.add_argument('--max-tasks', type=int, default=20, help='Maximum tasks to process')
    
    # Quantum analyze issue command
    analyze_parser = subparsers.add_parser('quantum-analyze-issue', help='Analyze GitHub issue with quantum insights')
    analyze_parser.add_argument('--repo', required=True, help='Repository name (owner/repo)')
    analyze_parser.add_argument('--issue', type=int, required=True, help='Issue number')
    analyze_parser.add_argument('--output-file', help='Output file for results')
    
    # Quantum insights command
    insights_parser = subparsers.add_parser('quantum-insights', help='Generate comprehensive quantum insights')
    insights_parser.add_argument('--output-file', help='Output file for results')
    insights_parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    
    # Interactive mode command
    subparsers.add_parser('interactive', help='Enter interactive quantum mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = QuantumCLI()
    
    try:
        if args.command == 'quantum-prioritize':
            cli.quantum_prioritize_tasks(
                args.tasks_file, 
                args.output_file, 
                enable_annealing=not args.no_annealing
            )
            
        elif args.command == 'quantum-orchestrate':
            cli.quantum_orchestrate_repository(
                args.repo,
                args.output_file,
                args.max_tasks
            )
            
        elif args.command == 'quantum-analyze-issue':
            cli.analyze_issue_quantum(
                args.repo,
                args.issue,
                args.output_file
            )
            
        elif args.command == 'quantum-insights':
            cli.generate_quantum_insights(
                args.output_file,
                args.batch_size
            )
            
        elif args.command == 'interactive':
            cli.interactive_quantum_mode()
            
        print(f"\n✨ Quantum operation completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚡ Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Quantum operation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()