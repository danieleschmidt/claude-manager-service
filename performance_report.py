#!/usr/bin/env python3
"""
Performance Report CLI Tool for Claude Manager Service

This script provides a command-line interface to view performance metrics,
generate reports, and monitor system health.

Usage:
    python performance_report.py [command] [options]

Commands:
    report [hours]     - Generate performance report for last N hours (default: 24)
    summary           - Show quick performance summary
    function <name>   - Show detailed metrics for specific function
    api-calls         - Show API call statistics
    save             - Save current metrics to file
    alerts           - Show recent performance alerts

Examples:
    python performance_report.py report 12
    python performance_report.py function github_create_issue
    python performance_report.py api-calls
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from performance_monitor import get_monitor


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way"""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def format_memory(mb: float) -> str:
    """Format memory usage in a human-readable way"""
    if abs(mb) < 1:
        return f"{mb * 1024:.1f}KB"
    elif abs(mb) < 1024:
        return f"{mb:.1f}MB"
    else:
        return f"{mb / 1024:.2f}GB"


def print_performance_report(hours: int = 24):
    """Print comprehensive performance report"""
    monitor = get_monitor()
    report = monitor.get_performance_report(hours)
    
    if "message" in report:
        print(f"📊 {report['message']}")
        return
    
    print(f"🚀 Performance Report - Last {hours} Hours")
    print(f"Generated at: {report['generated_at']}")
    print("=" * 60)
    
    # Overall statistics
    stats = report['overall_stats']
    print(f"\n📈 Overall Statistics:")
    print(f"  Total Operations: {stats['total_operations']:,}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Total Duration: {format_duration(stats['total_duration'])}")
    print(f"  Average Duration: {format_duration(stats['average_duration'])}")
    print(f"  Median Duration: {format_duration(stats['median_duration'])}")
    print(f"  95th Percentile: {format_duration(stats['p95_duration'])}")
    print(f"  99th Percentile: {format_duration(stats['p99_duration'])}")
    
    # API call summary
    if report['api_call_summary']:
        print(f"\n🌐 API Call Summary:")
        for api_name, api_stats in report['api_call_summary'].items():
            print(f"  {api_name}:")
            print(f"    Calls: {api_stats['total_calls']:,} | Success: {api_stats['success_rate']:.1%}")
    
    # Function breakdown (top 10)
    print(f"\n⚡ Function Performance (Top 10 by call count):")
    sorted_functions = sorted(
        report['function_breakdown'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )[:10]
    
    for func_name, func_stats in sorted_functions:
        print(f"  {func_name}:")
        print(f"    Calls: {func_stats['count']:,} | Success: {func_stats['success_rate']:.1%} | Avg: {format_duration(func_stats['avg_duration'])}")
    
    # Memory statistics
    if report['memory_stats']:
        memory = report['memory_stats']
        print(f"\n💾 Memory Statistics:")
        print(f"  Operations with memory tracking: {memory['total_operations_with_memory_tracking']}")
        print(f"  Average memory delta: {format_memory(memory['avg_memory_delta'])}")
        print(f"  Max memory delta: {format_memory(memory['max_memory_delta'])}")
        print(f"  Min memory delta: {format_memory(memory['min_memory_delta'])}")
    
    # Slowest operations
    print(f"\n🐌 Slowest Operations:")
    for i, op in enumerate(report['slowest_operations'][:5], 1):
        status = "✅" if op['success'] else "❌"
        print(f"  {i}. {op['function']} - {format_duration(op['duration'])} {status}")


def print_function_metrics(function_name: str):
    """Print detailed metrics for a specific function"""
    monitor = get_monitor()
    metrics = monitor.get_function_metrics(function_name)
    
    if not metrics:
        print(f"❌ No metrics found for function: {function_name}")
        return
    
    print(f"🔍 Function Metrics: {metrics.function_name}")
    if metrics.module_name != "mixed":
        print(f"Module: {metrics.module_name}")
    print("=" * 50)
    
    print(f"\n📊 Call Statistics:")
    print(f"  Total Calls: {metrics.total_calls:,}")
    print(f"  Successful: {metrics.successful_calls:,}")
    print(f"  Failed: {metrics.failed_calls:,}")
    print(f"  Success Rate: {metrics.success_rate:.1%}")
    
    print(f"\n⏱️  Duration Statistics:")
    print(f"  Total Duration: {format_duration(metrics.total_duration)}")
    print(f"  Average: {format_duration(metrics.average_duration)}")
    print(f"  Median: {format_duration(metrics.median_duration)}")
    print(f"  Min: {format_duration(metrics.min_duration)}")
    print(f"  Max: {format_duration(metrics.max_duration)}")
    print(f"  95th Percentile: {format_duration(metrics.p95_duration)}")
    print(f"  99th Percentile: {format_duration(metrics.p99_duration)}")
    
    print(f"\n📅 Timing:")
    print(f"  First Called: {metrics.first_called}")
    print(f"  Last Called: {metrics.last_called}")
    
    if metrics.avg_memory_usage is not None:
        print(f"\n💾 Memory:")
        print(f"  Average Memory Delta: {format_memory(metrics.avg_memory_usage)}")
    
    if metrics.error_types:
        print(f"\n❌ Error Types:")
        for error_type, count in metrics.error_types.items():
            print(f"  {error_type}: {count}")


def print_api_summary():
    """Print API call summary"""
    monitor = get_monitor()
    api_summary = monitor.get_api_call_summary()
    
    if not api_summary:
        print("📡 No API calls recorded")
        return
    
    print("🌐 API Call Summary")
    print("=" * 40)
    
    for api_name, stats in api_summary.items():
        print(f"\n{api_name}:")
        print(f"  Total Calls: {stats['total_calls']:,}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Error Rate: {stats['error_rate']:.1%}")


def print_quick_summary():
    """Print quick performance summary"""
    monitor = get_monitor()
    
    total_ops = len(monitor.operations)
    if total_ops == 0:
        print("📊 No operations recorded yet")
        return
    
    recent_ops = list(monitor.operations)[-100:]  # Last 100 operations
    successful = sum(1 for op in recent_ops if op.success)
    success_rate = successful / len(recent_ops) if recent_ops else 0
    
    durations = [op.duration for op in recent_ops]
    avg_duration = sum(durations) / len(durations) if durations else 0
    
    print("⚡ Quick Performance Summary")
    print("=" * 35)
    print(f"Total Operations: {total_ops:,}")
    print(f"Recent Success Rate: {success_rate:.1%} (last 100 ops)")
    print(f"Average Duration: {format_duration(avg_duration)} (last 100 ops)")
    print(f"Function Types: {len(monitor.function_stats)}")
    print(f"API Endpoints: {len(monitor.api_call_stats)}")


def save_metrics():
    """Save current metrics to file"""
    monitor = get_monitor()
    filepath = monitor.save_metrics()
    
    if filepath:
        print(f"💾 Metrics saved to: {filepath}")
    else:
        print("❌ Failed to save metrics")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Claude Manager Service Performance Report Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate performance report')
    report_parser.add_argument('hours', type=int, nargs='?', default=24,
                              help='Number of hours to include in report (default: 24)')
    
    # Function command
    func_parser = subparsers.add_parser('function', help='Show function metrics')
    func_parser.add_argument('name', help='Function name to analyze')
    
    # Other commands
    subparsers.add_parser('summary', help='Show quick summary')
    subparsers.add_parser('api-calls', help='Show API call statistics')
    subparsers.add_parser('save', help='Save metrics to file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'report':
            print_performance_report(args.hours)
        elif args.command == 'function':
            print_function_metrics(args.name)
        elif args.command == 'api-calls':
            print_api_summary()
        elif args.command == 'summary':
            print_quick_summary()
        elif args.command == 'save':
            save_metrics()
        else:
            parser.print_help()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())