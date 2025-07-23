# Continuous Backlog Execution Engine

The Continuous Backlog Execution Engine is an autonomous senior coding assistant that continuously processes backlog items using TDD discipline and WSJF (Weighted Shortest Job First) prioritization.

## 🎯 Mission

**Continuously ingest, score, order, and drive *every actionable item* in the backlog to completion**, pausing only for human input where risk or ambiguity demands it.

## 🏗️ Architecture

### Core Components

- **ContinuousBacklogExecutor**: Main orchestration engine
- **BacklogItem**: Structured representation of work items
- **WSJF Scoring**: Impact-based prioritization with aging multiplier
- **TDD Micro-Cycle**: Red-Green-Refactor discipline for each item
- **Discovery Engine**: Automated task discovery from multiple sources

### Task Status Flow

```
NEW → REFINED → READY → DOING → PR → MERGED/DONE
                          ↓
                      BLOCKED
```

## 🚀 Quick Start

### Basic Usage

```bash
# Run continuous execution with default config
python run_continuous_executor.py

# Run with custom configuration
python run_continuous_executor.py --config /path/to/config.json

# Dry run (plan without executing)
python run_continuous_executor.py --dry-run

# Discovery only
python run_continuous_executor.py --discover-only

# Status report
python run_continuous_executor.py --status-report
```

### Configuration

The executor uses the existing `config.json` format:

```json
{
  "github": {
    "username": "your-username",
    "managerRepo": "your-username/manager-repo",
    "reposToScan": ["repo1", "repo2"]
  },
  "analyzer": {
    "scanForTodos": true,
    "scanOpenIssues": true
  }
}
```

## 📊 WSJF Scoring System

Each backlog item is scored using:

**WSJF = (Impact + Time Criticality + Risk Reduction) × Aging Multiplier / Effort**

### Scoring Scale (1-13)

- **Impact**: Business/user value (1=Low, 13=Critical)
- **Effort**: Implementation complexity (1=Very Small, 13=Very Large)
- **Aging Multiplier**: 1.0 + (days_old × 0.1), capped at 2.0

### Automatic Impact Estimation

The system automatically estimates impact based on content keywords:

- **High Impact (8)**: security, vulnerability, critical, crash, performance
- **Medium Impact (5)**: bug, error, fix, improvement
- **Low Impact (3)**: documentation, cleanup, minor improvements

## 🔄 Execution Loop

### Macro Cycle (Continuous)

1. **Sync & Refresh**
   - Load existing backlog
   - Discover new tasks from multiple sources
   - Normalize and score all items
   - Rank by WSJF score

2. **Select & Process**
   - Walk prioritized list
   - Skip blocked items
   - Process actionable items using TDD

3. **Reassess & Continue**
   - Update metrics
   - Re-score remaining items
   - Loop until backlog empty or all blocked

### Micro Cycle (Per Item)

1. **Restate Acceptance Criteria**
2. **Write Failing Test** (Red)
3. **Implement Minimal Code** (Green)
4. **Refactor & Clean** (Blue)
5. **Security & Compliance Checks**
6. **Documentation Updates**
7. **CI Pipeline Verification**

## 🔍 Task Discovery Sources

### Currently Implemented

- **TODO/FIXME Comments**: Scans code for action items
- **Task Classification**: Automatically categorizes by type
- **Security Detection**: Identifies security-related items

### Planned Sources

- **Failing Tests**: Convert test failures to backlog items
- **PR Feedback**: Extract action items from review comments
- **Security Scans**: Import vulnerability reports
- **Dependency Alerts**: Track outdated/vulnerable dependencies

## 🛡️ Safety & Quality Guardrails

### Human Escalation Triggers

- Public interface changes
- Security-critical modules (auth, crypto, secrets)
- Large refactors (effort ≥ 8)
- Data migrations
- Performance-critical paths

### Security Checklist (Every Task)

- Input sanitization
- Authentication & access control
- Secrets handling
- Error handling
- Logging hygiene

### Test Requirements

- Unit tests for new functionality
- Integration tests for system boundaries
- All existing tests must pass
- Security test coverage where applicable

## 📈 Metrics & Monitoring

### Execution Metrics

```json
{
  "timestamp": "2025-07-23T10:00:00Z",
  "completed_items": ["item1", "item2"],
  "coverage_delta": 5.5,
  "cycle_time_avg": 120.5,
  "items_processed": 10,
  "items_completed": 7,
  "items_blocked": 2,
  "backlog_size_by_status": {
    "READY": 15,
    "DOING": 2,
    "BLOCKED": 3,
    "DONE": 45
  }
}
```

### Status Reports

Generated after each cycle and stored in `/DOCS/status/`:

- Backlog size and distribution
- WSJF score distribution
- Completion rates
- Notable risks and blocks
- Performance trends

## 🎛️ Configuration Options

### Execution Parameters

```python
slice_size_threshold = 5     # Split items with effort > 5
max_cycle_time = 3600       # 1 hour max per cycle
aging_cap = 2.0             # Maximum aging multiplier
```

### File Locations

- **Backlog**: `/DOCS/backlog.yml`
- **Status Reports**: `/DOCS/status/`
- **Tech Debt**: `/DOCS/tech_debt.md`

## 🔧 Development & Testing

### Running Tests

```bash
# Unit tests
python -m pytest tests/unit/test_continuous_backlog_executor.py

# Integration tests
python -m pytest tests/integration/test_continuous_executor_integration.py

# All tests
python -m pytest tests/
```

### Development Mode

```bash
# Enable debug logging
python run_continuous_executor.py --log-level DEBUG

# Limit cycles for testing
python run_continuous_executor.py --max-cycles 3
```

## 📚 Advanced Usage

### Custom Task Creation

You can manually add items to the backlog:

```python
from src.continuous_backlog_executor import BacklogItem, TaskType, TaskStatus
from datetime import datetime

item = BacklogItem(
    id="custom_1",
    title="Custom Task",
    description="Manually created task",
    task_type=TaskType.FEATURE,
    impact=8,
    effort=3,
    status=TaskStatus.READY,
    wsjf_score=0.0,  # Will be calculated
    created_at=datetime.now(),
    updated_at=datetime.now(),
    links=["https://github.com/repo/issues/123"],
    acceptance_criteria=[
        "Implement feature X",
        "Add unit tests",
        "Update documentation"
    ],
    security_notes="Standard security review",
    test_notes="Requires integration tests"
)
```

### Integration with Existing Workflows

The executor integrates with existing GitHub workflows:

- Uses same configuration format
- Leverages existing GitHub API wrapper
- Compatible with existing task analyzer
- Works with current orchestrator system

## 🚨 Troubleshooting

### Common Issues

1. **No Actionable Items**
   - Check that items have acceptance criteria
   - Verify items are marked as READY
   - Ensure items aren't blocked

2. **All Items Blocked**
   - Review blocked reasons in status report
   - Consider human intervention for high-risk items
   - Check for missing dependencies

3. **Low Discovery Rate**
   - Verify repository access permissions
   - Check GitHub API rate limits
   - Review TODO/FIXME patterns in code

### Logging

Enable detailed logging for debugging:

```bash
export LOG_LEVEL=DEBUG
python run_continuous_executor.py --log-level DEBUG
```

## 🔄 Continuous Improvement

The system includes meta-improvement capabilities:

- **Metrics Analysis**: Track execution efficiency
- **Process Refinement**: Adjust scoring weights based on outcomes
- **Discovery Enhancement**: Improve task detection accuracy
- **Quality Measurement**: Monitor code health impact

### End-of-Pass Review

After each full backlog pass:

1. **Measure Impact**: Code health, reliability, delivery velocity
2. **Identify Regressions**: Performance, quality, or process issues
3. **Create Meta-Tasks**: Process improvements as backlog items
4. **Tune Parameters**: Scoring weights, slice sizes, cycle times

---

## 📞 Support

For issues or questions:

- Review logs in `/DOCS/status/` for execution history
- Check blocked items for common patterns
- Use `--dry-run` mode to test changes safely
- Examine backlog file at `/DOCS/backlog.yml` for state inspection

The Continuous Backlog Executor represents a shift from manual task management to automated, disciplined, continuous delivery of valuable work.