# Claude Manager Service - Web Dashboard

A comprehensive web interface for monitoring and managing the Claude Manager Service, providing real-time insights into system performance, backlog status, and task execution.

## Features

### 📊 Dashboard Overview
- **System Health Monitoring**: Real-time status indicators for service health
- **Backlog Progress**: Visual representation of completed vs pending tasks
- **Performance Metrics**: Memory usage, uptime, and operation statistics
- **Recent Tasks**: Historical view of task execution with priority and status

### 🎯 Key Capabilities
- **Real-time Updates**: Auto-refresh every 30 seconds
- **Responsive Design**: Mobile-friendly Bootstrap UI
- **Interactive Charts**: Chart.js powered visualizations
- **Configuration Monitoring**: Current system settings and status
- **API Endpoints**: RESTful API for programmatic access

## Quick Start

### Prerequisites
- Python 3.8+
- Flask 2.3.0+
- All Claude Manager Service dependencies

### Installation
```bash
# Install additional dependencies
pip install Flask>=2.3.0 Flask-CORS>=4.0.0

# Or install from requirements.txt
pip install -r requirements.txt
```

### Running the Dashboard
```bash
# From the project root directory
python start_dashboard.py
```

The dashboard will be available at: `http://localhost:5000`

### Configuration
Set environment variables to customize the dashboard:

```bash
export DASHBOARD_HOST=0.0.0.0      # Default: 0.0.0.0
export DASHBOARD_PORT=5000         # Default: 5000  
export FLASK_DEBUG=False           # Default: False
```

## API Endpoints

### Health Check
```
GET /api/health
```
Returns system health status and component information.

### Backlog Status
```
GET /api/backlog
```
Returns current backlog completion statistics and progress.

### Performance Metrics
```
GET /api/performance
```
Returns real-time performance metrics including memory usage and uptime.

### Recent Tasks
```
GET /api/tasks?limit=10
```
Returns recent task execution history with optional limit parameter.

### System Configuration
```
GET /api/config
```
Returns sanitized system configuration (sensitive data redacted).

## Architecture

### Frontend
- **Framework**: Vanilla JavaScript with Bootstrap 5
- **Charts**: Chart.js for data visualization
- **Styling**: Custom CSS with responsive design
- **Real-time**: Auto-refresh with AJAX calls

### Backend
- **Framework**: Flask with Flask-CORS
- **API**: RESTful endpoints returning JSON
- **Integration**: Direct integration with Claude Manager Service modules
- **Security**: Sensitive data sanitization and validation

### File Structure
```
web/
├── app.py                 # Main Flask application
├── templates/
│   └── dashboard.html     # Main dashboard template
├── static/
│   ├── css/
│   │   └── dashboard.css  # Custom styles
│   └── js/
│       └── dashboard.js   # Frontend JavaScript
└── README.md             # This file
```

## Development

### Adding New Features
1. **Backend**: Add new API endpoints in `app.py`
2. **Frontend**: Update JavaScript in `dashboard.js`
3. **Styling**: Modify CSS in `dashboard.css`
4. **Templates**: Update HTML in `dashboard.html`

### API Integration
The dashboard integrates with existing Claude Manager Service modules:
- `performance_monitor`: For system metrics
- `config_validator`: For configuration data
- `task_prioritization`: For task analysis
- `logger`: For consistent logging

### Security Considerations
- Sensitive configuration data is automatically sanitized
- No authentication implemented (intended for internal use)
- CORS enabled for development (configure for production)

## Monitoring Features

### System Health
- Service status indicator
- Component health checks
- Real-time status updates

### Backlog Management
- Completion percentage tracking
- Visual progress indicators
- High-priority task monitoring

### Performance Tracking
- Memory usage monitoring
- Operation count tracking
- Uptime statistics
- Peak performance metrics

### Task History
- Recent task execution log
- Priority-based filtering
- Status tracking
- Repository association

## Troubleshooting

### Common Issues

**Dashboard won't start**
- Ensure all dependencies are installed
- Check that the src/ directory is accessible
- Verify Python path configuration

**API endpoints return errors**
- Check that Claude Manager Service modules are importable
- Verify configuration files exist
- Check log output for specific error messages

**Charts not displaying**
- Ensure internet connectivity for CDN resources
- Check browser console for JavaScript errors
- Verify API endpoints are returning valid data

### Logging
The dashboard uses the Claude Manager Service logging system. Check logs for detailed error information and debugging.

## Future Enhancements

### Planned Features
- WebSocket integration for real-time updates
- User authentication and authorization
- Task execution controls and triggers
- Advanced filtering and search
- Export functionality for reports
- Mobile app integration

### Scalability
- Database integration for historical data
- Caching layer for improved performance
- Load balancing for multiple instances
- Advanced monitoring and alerting

## Contributing

The web dashboard follows the same development patterns as the main Claude Manager Service:
- Comprehensive error handling
- Structured logging
- Security-first design
- Modular architecture
- Thorough documentation

For more information, see the main project documentation.