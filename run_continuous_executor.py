#!/usr/bin/env python3
"""
CLI Runner for Continuous Backlog Executor

Usage:
    python run_continuous_executor.py [--config CONFIG_PATH] [--dry-run] [--max-cycles N]
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from continuous_backlog_executor import ContinuousBacklogExecutor
from logger import get_logger


async def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Continuous Backlog Execution Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config
    python run_continuous_executor.py
    
    # Run with custom config
    python run_continuous_executor.py --config /path/to/config.json
    
    # Dry run mode (discover and plan only)
    python run_continuous_executor.py --dry-run
    
    # Limit to specific number of cycles
    python run_continuous_executor.py --max-cycles 5
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )
    
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Discover and plan tasks but do not execute them'
    )
    
    parser.add_argument(
        '--max-cycles', '-m',
        type=int,
        default=None,
        help='Maximum number of execution cycles to run'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--discover-only',
        action='store_true',
        help='Only run task discovery, do not process items'
    )
    
    parser.add_argument(
        '--status-report',
        action='store_true',
        help='Generate status report and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = get_logger(__name__)
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Create executor
        executor = ContinuousBacklogExecutor(str(config_path))
        
        # Handle different modes
        if args.status_report:
            await generate_status_report(executor)
        elif args.discover_only:
            await run_discovery_only(executor)
        elif args.dry_run:
            await run_dry_run(executor)
        else:
            await run_continuous_execution(executor, args.max_cycles)
    
    except KeyboardInterrupt:
        logger.info("🛑 Execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)


async def run_continuous_execution(executor: ContinuousBacklogExecutor, max_cycles: int = None):
    """Run full continuous execution"""
    logger = get_logger(__name__)
    
    logger.info("🚀 Starting Continuous Backlog Execution")
    
    if max_cycles:
        logger.info(f"🔢 Limited to {max_cycles} cycles")
        
        cycle_count = 0
        while cycle_count < max_cycles:
            cycle_count += 1
            logger.info(f"📋 Cycle {cycle_count}/{max_cycles}")
            
            # Run one cycle
            await executor._sync_and_refresh()
            
            actionable_items = executor._get_actionable_items()
            if not actionable_items:
                logger.info("✅ No actionable items remaining")
                break
            
            # Process a few items this cycle
            items_to_process = min(3, len(actionable_items))
            for i in range(items_to_process):
                item = actionable_items[i]
                result = await executor._process_backlog_item(item)
                logger.info(f"📝 Processed {item.title}: {result}")
            
            # Generate cycle report
            from continuous_backlog_executor import ExecutionMetrics
            from datetime import datetime
            
            metrics = ExecutionMetrics(cycle_start=datetime.now())
            metrics.items_processed = items_to_process
            await executor._generate_status_report(metrics)
            
            if executor._should_terminate():
                break
    else:
        # Run unlimited cycles
        await executor.run_continuous_execution()
    
    logger.info("🏁 Continuous execution completed")


async def run_dry_run(executor: ContinuousBacklogExecutor):
    """Run discovery and planning without execution"""
    logger = get_logger(__name__)
    
    logger.info("🔍 Running dry-run mode (discovery and planning only)")
    
    # Run discovery
    await executor._sync_and_refresh()
    
    # Show what would be done
    logger.info(f"📊 Discovered {len(executor.backlog)} backlog items")
    
    actionable_items = executor._get_actionable_items()
    logger.info(f"⚡ {len(actionable_items)} items are actionable")
    
    if actionable_items:
        logger.info("🎯 Top 5 items that would be processed:")
        for i, item in enumerate(actionable_items[:5], 1):
            logger.info(f"  {i}. {item.title} (WSJF: {item.wsjf_score:.2f}, Impact: {item.impact}, Effort: {item.effort})")
    
    # Show status distribution
    from collections import Counter
    status_counts = Counter(item.status.value for item in executor.backlog)
    logger.info(f"📈 Status distribution: {dict(status_counts)}")
    
    # Show type distribution
    type_counts = Counter(item.task_type.value for item in executor.backlog)
    logger.info(f"🏷️  Type distribution: {dict(type_counts)}")
    
    logger.info("✅ Dry-run completed")


async def run_discovery_only(executor: ContinuousBacklogExecutor):
    """Run only task discovery"""
    logger = get_logger(__name__)
    
    logger.info("🔍 Running discovery-only mode")
    
    # Load existing backlog
    await executor._load_backlog()
    initial_count = len(executor.backlog)
    
    # Discover new tasks
    await executor._discover_new_tasks()
    
    # Normalize and score
    executor._normalize_backlog_items()
    executor._score_and_rank_backlog()
    
    # Save updated backlog
    await executor._save_backlog()
    
    new_count = len(executor.backlog)
    discovered_count = new_count - initial_count
    
    logger.info(f"📊 Discovery completed:")
    logger.info(f"  - Existing items: {initial_count}")
    logger.info(f"  - New items discovered: {discovered_count}")
    logger.info(f"  - Total items: {new_count}")
    
    if discovered_count > 0:
        logger.info("🆕 New items discovered:")
        new_items = executor.backlog[initial_count:]
        for item in new_items[:10]:  # Show first 10
            logger.info(f"  - {item.title} ({item.task_type.value}, WSJF: {item.wsjf_score:.2f})")
        
        if len(new_items) > 10:
            logger.info(f"  ... and {len(new_items) - 10} more")


async def generate_status_report(executor: ContinuousBacklogExecutor):
    """Generate and display status report"""
    logger = get_logger(__name__)
    
    logger.info("📊 Generating status report")
    
    # Load current backlog
    await executor._load_backlog()
    
    if not executor.backlog:
        logger.info("📭 No backlog items found")
        return
    
    # Generate comprehensive report
    from collections import Counter
    from datetime import datetime
    
    total_items = len(executor.backlog)
    status_counts = Counter(item.status.value for item in executor.backlog)
    type_counts = Counter(item.task_type.value for item in executor.backlog)
    
    # Calculate WSJF statistics
    scores = [item.wsjf_score for item in executor.backlog if item.wsjf_score > 0]
    avg_wsjf = sum(scores) / len(scores) if scores else 0
    
    logger.info(f"📈 Backlog Status Report")
    logger.info(f"  Total items: {total_items}")
    logger.info(f"  Average WSJF score: {avg_wsjf:.2f}")
    logger.info("")
    
    logger.info("📊 Status Distribution:")
    for status, count in status_counts.most_common():
        percentage = (count / total_items) * 100
        logger.info(f"  {status}: {count} ({percentage:.1f}%)")
    logger.info("")
    
    logger.info("🏷️  Type Distribution:")
    for task_type, count in type_counts.most_common():
        percentage = (count / total_items) * 100
        logger.info(f"  {task_type}: {count} ({percentage:.1f}%)")
    logger.info("")
    
    # Show top priority items
    actionable_items = executor._get_actionable_items()
    if actionable_items:
        logger.info("🎯 Top Priority Actionable Items:")
        for i, item in enumerate(actionable_items[:5], 1):
            logger.info(f"  {i}. {item.title}")
            logger.info(f"     Type: {item.task_type.value}, WSJF: {item.wsjf_score:.2f}")
            logger.info(f"     Impact: {item.impact}, Effort: {item.effort}")
    else:
        logger.info("⚡ No actionable items available")
    
    # Show blocked items
    blocked_items = [item for item in executor.backlog if item.status.value == "BLOCKED"]
    if blocked_items:
        logger.info("")
        logger.info("🚫 Blocked Items:")
        for item in blocked_items[:5]:
            logger.info(f"  - {item.title}: {item.blocked_reason}")
    
    logger.info("✅ Status report completed")


if __name__ == "__main__":
    asyncio.run(main())