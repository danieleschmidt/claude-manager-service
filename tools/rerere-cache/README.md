# Rerere Cache Audit

This directory contains Git rerere cache entries for auditing automatic merge conflict resolutions.

## Purpose

The rerere (reuse recorded resolution) cache stores how merge conflicts were previously resolved, enabling automatic resolution of similar conflicts in the future.

## Audit Process

1. After each merge conflict resolution, the solution is stored in Git's rerere cache
2. CI workflows can run `git rerere diff` to show what automatic resolutions would be applied
3. Cache entries are uploaded as artifacts for human review
4. Any suspicious automatic resolutions can be flagged for manual review

## Safety Features

- Only applies to file types specified in `.gitattributes`
- Binary files are locked to prevent corruption
- Documentation files use line-union merge for safe concatenation
- Lock files use "theirs" strategy to prefer incoming changes

## Monitoring

The auto-rebase GitHub Action will:
- Log all automatic resolutions
- Fail fast if genuine logic conflicts are detected
- Provide clear error messages for manual intervention needed