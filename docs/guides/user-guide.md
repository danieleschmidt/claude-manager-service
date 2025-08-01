# Claude Manager Service - User Guide

## Overview

This guide provides comprehensive instructions for using the Claude Manager Service, an autonomous SDLC management system that helps development teams automate task discovery, prioritization, and execution.

## Getting Started

### Prerequisites

- GitHub account with repository access
- Python 3.9+ installed
- Docker (optional, for containerized deployment)
- Node.js 16+ (for dashboard features)

### Quick Start

1. **Clone and Setup**
   ```bash
   git clone https://github.com/danieleschmidt/claude-manager-service.git
   cd claude-manager-service
   pip install -r requirements.txt
   ```

2. **Configure GitHub Integration**
   ```bash
   cp config.json.example config.json
   # Edit config.json with your GitHub settings
   ```

3. **Set Environment Variables**
   ```bash
   export GITHUB_TOKEN="your_github_token"
   export TERRAGON_TOKEN="your_terragon_token"  # Optional
   ```

4. **Run the Service**
   ```bash
   python src/task_analyzer.py  # For task discovery
   python start_dashboard.py   # For web dashboard
   ```

## Core Features

### 1. Autonomous Task Discovery

The system automatically scans your repositories for:

- **TODO Comments**: Identifies actionable TODO and FIXME comments
- **Stale Issues**: Finds inactive issues that need attention
- **Code Quality Issues**: Detects potential refactoring opportunities
- **Security Vulnerabilities**: Identifies security-related tasks

#### Configuration
Edit `config.json` to configure scanning:
```json
{
  "analyzer": {
    "scanForTodos": true,
    "scanOpenIssues": true,
    "scanSecurityIssues": true
  }
}
```

### 2. Task Prioritization

Tasks are automatically prioritized based on:
- Business impact assessment
- Technical complexity analysis
- Resource availability
- Dependencies and prerequisites

### 3. AI Agent Integration

#### Terragon Tasks
Label issues with `terragon-task` to trigger Terragon AI:
- Automatic code generation
- Bug fixing and optimization
- Test creation and enhancement

#### Claude Flow Tasks
Label issues with `claude-flow-task` to trigger Claude Flow:
- Complex refactoring operations
- Architecture improvements
- Documentation generation

### 4. Dashboard and Monitoring

Access the web dashboard at `http://localhost:8080` for:
- Real-time task status
- Performance metrics
- System health monitoring
- Team productivity insights

## Workflow Management

### Creating Manual Tasks

Use the one-off task script:
```bash
./scripts/one_off_task.sh "Fix login bug" "The login form validation is broken" "myorg/myrepo" "terragon"
```

### Approving Tasks

1. Navigate to the manager repository issues
2. Review the automatically created task proposal
3. Add the `approved-for-dev` label to trigger execution
4. Optionally add `terragon-task` or `claude-flow-task` labels

### Monitoring Task Progress

- **GitHub Issues**: Track progress through issue comments and status updates
- **Web Dashboard**: Real-time visualization of task execution
- **Performance Metrics**: Monitor system performance and efficiency

## Configuration Options

### Repository Settings

Configure which repositories to scan in `config.json`:
```json
{
  "github": {
    "username": "your-username",
    "managerRepo": "your-username/claude-manager-service",
    "reposToScan": [
      "your-username/project-1",
      "your-username/project-2"
    ]
  }
}
```

### Performance Tuning

Adjust performance settings through environment variables:
```bash
# Performance monitoring
export PERF_ALERT_DURATION=15.0
export PERF_MAX_OPERATIONS=20000

# Rate limiting
export RATE_LIMIT_REQUESTS=1000
export RATE_LIMIT_WINDOW=3600

# Security settings
export SECURITY_MAX_CONTENT_LENGTH=75000
```

### Feature Flags

Enable/disable features:
```bash
export FEATURE_MONITORING_ENABLED=true
export FEATURE_RATE_LIMITING_ENABLED=true
export FEATURE_ENHANCED_SECURITY=true
```

## Best Practices

### Task Organization

1. **Use Descriptive Labels**: Apply clear, consistent labels to help with task categorization
2. **Regular Review**: Periodically review and prioritize the task backlog
3. **Team Coordination**: Ensure team members understand the automated workflow

### Quality Assurance

1. **Test Coverage**: Maintain high test coverage for reliable automation
2. **Code Reviews**: Review AI-generated code before merging
3. **Performance Monitoring**: Keep an eye on system performance metrics

### Security Considerations

1. **Token Management**: Securely store GitHub and AI service tokens
2. **Access Control**: Limit repository access to necessary permissions only
3. **Regular Updates**: Keep dependencies and security patches up to date

## Troubleshooting

### Common Issues

**Issue**: "GitHub API rate limit exceeded"
**Solution**: 
- Implement API request caching
- Use multiple GitHub tokens
- Reduce scanning frequency

**Issue**: "Terragon/Claude Flow not responding"
**Solution**:
- Check AI service authentication
- Verify network connectivity
- Review service quotas and limits

**Issue**: "Dashboard not loading"
**Solution**:
- Check if port 8080 is available
- Verify Python dependencies are installed
- Check application logs for errors

### Performance Issues

**Slow Task Discovery**:
- Reduce repository scan scope
- Implement parallel processing
- Optimize database queries

**High Memory Usage**:
- Adjust batch processing sizes
- Implement result caching
- Monitor for memory leaks

### Getting Help

1. **Documentation**: Check the comprehensive docs in `/docs`
2. **GitHub Issues**: Report bugs and feature requests
3. **Community**: Join our Discord/Slack for support
4. **Logs**: Check application logs in `/logs` directory

## Advanced Usage

### Custom Analyzers

Create custom task analyzers by extending the base analyzer class:
```python
from src.task_analyzer import BaseAnalyzer

class CustomAnalyzer(BaseAnalyzer):
    def analyze(self, repo):
        # Your custom analysis logic
        pass
```

### Webhook Integration

Set up webhooks for real-time task updates:
```python
from flask import Flask, request
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    # Process GitHub webhook events
    pass
```

### API Integration

Use the REST API for programmatic access:
```bash
# Get task status
curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:8080/api/tasks

# Create manual task
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"title": "Fix bug", "description": "Bug description"}' \
     http://localhost:8080/api/tasks
```

## Migration Guide

### From Manual Processes

1. **Audit Current Workflow**: Document existing task management processes
2. **Gradual Rollout**: Start with one repository and expand gradually
3. **Team Training**: Ensure team members understand the new workflow
4. **Monitor and Adjust**: Track adoption and adjust configuration as needed

### Version Updates

1. **Backup Configuration**: Save current settings before updating
2. **Review Changelog**: Check for breaking changes
3. **Test in Staging**: Verify functionality in non-production environment
4. **Monitor After Update**: Watch for issues after deployment

## Support and Resources

- **Documentation**: `/docs` directory
- **GitHub Issues**: Report bugs and feature requests
- **Security Issues**: Report to security@example.com
- **Community Support**: Join our development community
- **Commercial Support**: Contact enterprise support team

---

For additional help, see the [Developer Guide](developer-guide.md) and [API Documentation](../API.md).