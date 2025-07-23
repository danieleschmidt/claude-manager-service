/**
 * Claude Manager Service Dashboard JavaScript
 * 
 * Handles real-time data updates, chart rendering, and user interactions
 * for the web dashboard interface.
 */

class DashboardManager {
    constructor() {
        this.updateInterval = 30000; // 30 seconds
        this.charts = {};
        this.lastUpdate = null;
        
        this.init();
    }
    
    init() {
        console.log('Initializing Claude Manager Dashboard...');
        
        // Initialize charts
        this.initCharts();
        
        // Load initial data
        this.updateAllData();
        
        // Set up auto-refresh
        setInterval(() => this.updateAllData(), this.updateInterval);
        
        // Set up event listeners
        this.setupEventListeners();
    }
    
    initCharts() {
        // Backlog status chart
        this.charts.backlog = new Chart(document.getElementById('backlogChart'), {
            type: 'doughnut',
            data: {
                labels: ['Completed', 'Pending'],
                datasets: [{
                    data: [0, 0],
                    backgroundColor: ['#28a745', '#ffc107'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
        
        // Memory usage chart
        this.charts.memory = new Chart(document.getElementById('memoryChart'), {
            type: 'doughnut',
            data: {
                labels: ['Used', 'Free'],
                datasets: [{
                    data: [0, 100],
                    backgroundColor: ['#17a2b8', '#e9ecef'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    async updateAllData() {
        console.log('Updating dashboard data...');
        
        try {
            // Update all data sections
            await Promise.all([
                this.updateSystemHealth(),
                this.updateBacklogStatus(),
                this.updatePerformanceMetrics(),
                this.updateRecentTasks(),
                this.updateConfiguration()
            ]);
            
            // Update last update timestamp
            this.lastUpdate = new Date();
            document.getElementById('update-time').textContent = 
                this.lastUpdate.toLocaleTimeString();
                
        } catch (error) {
            console.error('Error updating dashboard data:', error);
            this.showAlert('Error updating dashboard data', 'danger');
        }
    }
    
    async updateSystemHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            const statusElement = document.getElementById('system-status');
            if (data.status === 'healthy') {
                statusElement.innerHTML = '<i class="fas fa-check-circle"></i> Healthy';
                statusElement.parentElement.parentElement.parentElement.className = 
                    'card bg-success text-white';
            } else {
                statusElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Warning';
                statusElement.parentElement.parentElement.parentElement.className = 
                    'card bg-warning text-white';
            }
        } catch (error) {
            console.error('Error updating system health:', error);
            document.getElementById('system-status').innerHTML = 
                '<i class="fas fa-times-circle"></i> Error';
        }
    }
    
    async updateBacklogStatus() {
        try {
            const response = await fetch('/api/backlog');
            const data = await response.json();
            
            if (data.error) {
                console.error('Backlog API error:', data.error);
                return;
            }
            
            // Update summary cards
            document.getElementById('completion-rate').textContent = 
                data.completion_rate + '%';
            document.getElementById('completed-count').textContent = 
                data.completed_items;
            document.getElementById('pending-count').textContent = 
                data.pending_items;
            document.getElementById('total-count').textContent = 
                data.total_items;
            
            // Update chart
            this.charts.backlog.data.datasets[0].data = [
                data.completed_items,
                data.pending_items
            ];
            this.charts.backlog.update();
            
        } catch (error) {
            console.error('Error updating backlog status:', error);
        }
    }
    
    async updatePerformanceMetrics() {
        try {
            const response = await fetch('/api/performance');
            const data = await response.json();
            
            if (data.error) {
                console.error('Performance API error:', data.error);
                return;
            }
            
            // Update memory usage card
            const memoryMB = data.memory_usage?.current_mb || 0;
            document.getElementById('memory-usage').textContent = 
                memoryMB.toFixed(1) + ' MB';
            
            // Update performance stats
            document.getElementById('uptime').textContent = 
                (data.system_health?.uptime_hours || 0).toFixed(1) + 'h';
            document.getElementById('total-operations').textContent = 
                data.system_health?.total_operations || 0;
            document.getElementById('peak-memory').textContent = 
                (data.memory_usage?.peak_mb || 0).toFixed(1) + ' MB';
            document.getElementById('last-operation').textContent = 
                data.system_health?.last_operation || 'Never';
            
            // Update memory chart (assuming max 1GB for visualization)
            const maxMemory = 1024; // 1GB in MB
            const usedPercent = (memoryMB / maxMemory) * 100;
            this.charts.memory.data.datasets[0].data = [
                usedPercent,
                100 - usedPercent
            ];
            this.charts.memory.update();
            
        } catch (error) {
            console.error('Error updating performance metrics:', error);
        }
    }
    
    async updateRecentTasks() {
        try {
            const response = await fetch('/api/tasks?limit=10');
            const tasks = await response.json();
            
            const tbody = document.getElementById('tasks-tbody');
            
            if (tasks.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" class="text-center text-muted">
                            No recent tasks found
                        </td>
                    </tr>
                `;
                return;
            }
            
            tbody.innerHTML = tasks.map(task => `
                <tr>
                    <td class="text-truncate-custom" title="${task.title}">
                        ${task.title}
                    </td>
                    <td>
                        <span class="badge type-${task.type}">${task.type}</span>
                    </td>
                    <td>
                        <span class="badge priority-${this.getPriorityLevel(task.priority_score)}">
                            ${task.priority_score}
                        </span>
                    </td>
                    <td>
                        <span class="badge status-${task.status.replace('_', '-')}">
                            ${task.status.replace('_', ' ')}
                        </span>
                    </td>
                    <td class="text-truncate-custom" title="${task.repository}">
                        ${task.repository}
                    </td>
                    <td>
                        <small class="text-muted">
                            ${this.formatDate(task.created_at)}
                        </small>
                    </td>
                </tr>
            `).join('');
            
            // Update active tasks count
            const activeTasks = tasks.filter(t => t.status === 'in_progress').length;
            document.getElementById('active-tasks').textContent = activeTasks;
            
        } catch (error) {
            console.error('Error updating recent tasks:', error);
        }
    }
    
    async updateConfiguration() {
        try {
            const response = await fetch('/api/config');
            const config = await response.json();
            
            if (config.error) {
                console.error('Config API error:', config.error);
                return;
            }
            
            const configContent = document.getElementById('config-content');
            configContent.innerHTML = `
                <div class="config-item">
                    <div class="config-label">Repositories</div>
                    <div class="config-value">${config.repositories?.length || 0} configured</div>
                </div>
                <div class="config-item">
                    <div class="config-label">TODO Scanning</div>
                    <div class="config-value">
                        <span class="status-indicator ${config.analyzer?.scanForTodos ? 'status-healthy' : 'status-warning'}"></span>
                        ${config.analyzer?.scanForTodos ? 'Enabled' : 'Disabled'}
                    </div>
                </div>
                <div class="config-item">
                    <div class="config-label">Issue Analysis</div>
                    <div class="config-value">
                        <span class="status-indicator ${config.analyzer?.scanOpenIssues ? 'status-healthy' : 'status-warning'}"></span>
                        ${config.analyzer?.scanOpenIssues ? 'Enabled' : 'Disabled'}
                    </div>
                </div>
                <div class="config-item">
                    <div class="config-label">Terragon Integration</div>
                    <div class="config-value">
                        <span class="status-indicator ${config.terragon?.enabled ? 'status-healthy' : 'status-warning'}"></span>
                        ${config.terragon?.enabled ? 'Connected' : 'Not configured'}
                    </div>
                </div>
                <div class="config-item">
                    <div class="config-label">Manager Repository</div>
                    <div class="config-value">${config.manager_repo || 'Not configured'}</div>
                </div>
            `;
            
        } catch (error) {
            console.error('Error updating configuration:', error);
        }
    }
    
    setupEventListeners() {
        // Add refresh button functionality if needed
        // Add modal dialogs for detailed views
        // Add real-time updates via WebSocket (future enhancement)
    }
    
    getPriorityLevel(score) {
        if (score >= 7) return 'high';
        if (score >= 4) return 'medium';
        return 'low';
    }
    
    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffMs = now - date;
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        
        if (diffHours < 1) {
            const diffMinutes = Math.floor(diffMs / (1000 * 60));
            return `${diffMinutes}m ago`;
        } else if (diffHours < 24) {
            return `${diffHours}h ago`;
        } else {
            return date.toLocaleDateString();
        }
    }
    
    showAlert(message, type = 'info') {
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show alert-banner" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        // Remove existing alerts
        document.querySelectorAll('.alert-banner').forEach(alert => alert.remove());
        
        // Add new alert
        document.body.insertAdjacentHTML('beforeend', alertHtml);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            document.querySelectorAll('.alert-banner').forEach(alert => {
                if (alert.classList.contains('show')) {
                    alert.classList.remove('show');
                    setTimeout(() => alert.remove(), 150);
                }
            });
        }, 5000);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    new DashboardManager();
});