#!/bin/bash

# One-off Task Creation Script
# Usage: ./scripts/one_off_task.sh "Task Title" "Task Description" "target-repo" "executor-type"

set -e

TASK_TITLE="$1"
TASK_DESCRIPTION="$2"
TARGET_REPO="$3"
EXECUTOR_TYPE="$4"  # "terragon" or "claude-flow"

if [ -z "$TASK_TITLE" ] || [ -z "$TASK_DESCRIPTION" ] || [ -z "$TARGET_REPO" ]; then
    echo "Usage: $0 \"Task Title\" \"Task Description\" \"target-repo\" [executor-type]"
    echo "Example: $0 \"Fix login bug\" \"The login form is not validating properly\" \"myorg/myproject\" \"terragon\""
    exit 1
fi

# Default to terragon if no executor specified
if [ -z "$EXECUTOR_TYPE" ]; then
    EXECUTOR_TYPE="terragon"
fi

# Validate executor type
if [ "$EXECUTOR_TYPE" != "terragon" ] && [ "$EXECUTOR_TYPE" != "claude-flow" ]; then
    echo "Error: executor-type must be 'terragon' or 'claude-flow'"
    exit 1
fi

echo "Creating manual task..."
echo "Title: $TASK_TITLE"
echo "Target Repository: $TARGET_REPO"
echo "Executor: $EXECUTOR_TYPE"
echo ""

# Create the issue body
ISSUE_BODY="**Manual Task Creation**

**Target Repository:** $TARGET_REPO
**Requested Executor:** $EXECUTOR_TYPE

**Description:**
$TASK_DESCRIPTION

---
*This task was manually created using the one-off task script.*"

# Determine labels based on executor type
if [ "$EXECUTOR_TYPE" = "terragon" ]; then
    LABELS="manual-task,terragon-task"
else
    LABELS="manual-task,claude-flow-task"
fi

# Create the issue using GitHub CLI (requires gh to be installed and authenticated)
if command -v gh &> /dev/null; then
    echo "Creating issue using GitHub CLI..."
    gh issue create \
        --title "$TASK_TITLE" \
        --body "$ISSUE_BODY" \
        --label "$LABELS"
    
    echo "Issue created successfully!"
    echo "Add the 'approved-for-dev' label to trigger execution."
else
    echo "GitHub CLI not found. Please install 'gh' to use this script."
    echo "Alternatively, create the issue manually with:"
    echo "Title: $TASK_TITLE"
    echo "Labels: $LABELS"
    echo "Body:"
    echo "$ISSUE_BODY"
fi