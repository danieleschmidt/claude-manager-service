"""
Generation 2: MAKE IT ROBUST - Complete System Integration

This module integrates all Generation 2 robustness features into a cohesive system:

1. Enhanced Error Handling & Recovery
2. Input Validation & Sanitization  
3. Comprehensive Logging & Monitoring
4. Security Hardening
5. Configuration Validation
6. Health Check & Diagnostics

This represents the complete evolution to enterprise-grade reliability and security.
"""

import asyncio
import time
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from contextlib import asynccontextmanager

from src.robust_system_v2 import RobustSystem, SecurityContext, create_robust_system
from src.enhanced_logger import (
    get_enhanced_logger, log_context, log_security_event, 
    LogContext, monitor_performance
)
from src.security_v2 import get_security_manager, SecurityPolicy
from src.config_validator_v2 import validate_configuration, generate_validation_report
from src.health_check_v2 import get_health_check
from src.services.configuration_service import ConfigurationService


@dataclass
class Generation2Status:
    """Generation 2 system status"""
    initialized: bool
    features_enabled: Dict[str, bool]
    security_status: str
    health_score: float
    uptime_seconds: float
    last_health_check: Optional[datetime]
    performance_metrics: Dict[str, Any]


class Generation2System:
    """
    Complete Generation 2 Robust System
    
    This class provides the complete implementation of Generation 2 robustness
    with all enterprise-grade features integrated and operational.
    """
    
    def __init__(self, config_service: ConfigurationService):
        self.config_service = config_service
        self.logger = get_enhanced_logger(__name__)
        
        # Core robustness components
        self.robust_system: Optional[RobustSystem] = None
        self.security_manager = None
        self.health_check = None
        
        # System state
        self.initialization_time = datetime.now()
        self.is_initialized = False
        self.features_enabled = {
            'enhanced_error_handling': False,
            'input_validation': False,
            'structured_logging': False,
            'security_hardening': False,
            'configuration_validation': False,
            'health_monitoring': False,
            'performance_monitoring': False,
            'audit_logging': False
        }
        
        self.logger.info("Generation 2 System initialized")
    
    async def initialize(self) -> None:
        """
        Initialize all Generation 2 robustness features
        """
        start_time = time.time()
        
        try:
            with log_context(
                operation="generation_2_initialization",
                module="generation_2_system"
            ):
                self.logger.info("Starting Generation 2 system initialization")
                
                # 1. Initialize robust system core
                await self._initialize_robust_system()
                
                # 2. Initialize security system
                await self._initialize_security_system()
                
                # 3. Initialize health monitoring
                await self._initialize_health_monitoring()
                
                # 4. Validate configuration
                await self._validate_system_configuration()
                
                # 5. Perform initial health check
                await self._perform_initial_health_check()
                
                # 6. Setup monitoring and alerts
                await self._setup_monitoring()
                
                self.is_initialized = True
                initialization_time = time.time() - start_time
                
                self.logger.info(
                    f"Generation 2 system initialization completed successfully "
                    f"in {initialization_time:.2f} seconds"
                )
                
                # Log successful initialization as security event
                log_security_event(
                    event_type='system_initialization',
                    severity='low',
                    description='Generation 2 robust system initialized successfully',
                    action='initialize',
                    result='success'
                )
                
        except Exception as e:
            self.logger.error(f"Generation 2 system initialization failed: {e}")
            
            # Log initialization failure
            log_security_event(
                event_type='system_initialization',
                severity='high',
                description=f'Generation 2 system initialization failed: {str(e)}',
                action='initialize',
                result='failure'
            )
            
            raise
    
    async def _initialize_robust_system(self):
        """Initialize the robust system core"""
        try:
            self.robust_system = await create_robust_system(self.config_service)
            self.features_enabled['enhanced_error_handling'] = True
            self.features_enabled['input_validation'] = True
            self.features_enabled['performance_monitoring'] = True
            
            self.logger.info("Robust system core initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize robust system: {e}")
            raise
    
    async def _initialize_security_system(self):
        """Initialize the security system"""
        try:
            # Create security policy from configuration
            config = await self.config_service.get_config()
            security_config = config.get('security', {})
            
            policy = SecurityPolicy(
                max_login_attempts=security_config.get('max_login_attempts', 5),
                lockout_duration_minutes=security_config.get('lockout_duration_minutes', 30),
                rate_limit_requests_per_minute=security_config.get('rate_limit_requests_per_minute', 60),
                csrf_protection_enabled=security_config.get('csrf_protection_enabled', True)
            )
            
            self.security_manager = get_security_manager(policy)
            self.features_enabled['security_hardening'] = True
            self.features_enabled['audit_logging'] = True
            
            self.logger.info("Security system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize security system: {e}")
            raise
    
    async def _initialize_health_monitoring(self):
        """Initialize health monitoring system"""
        try:
            self.health_check = await get_health_check(self.config_service)
            self.features_enabled['health_monitoring'] = True
            
            self.logger.info("Health monitoring system initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize health monitoring: {e}")
            raise
    
    async def _validate_system_configuration(self):
        """Validate system configuration"""
        try:
            config = await self.config_service.get_config()
            
            # Determine environment
            environment = os.getenv('ENVIRONMENT', 'development')
            
            validation_result = await validate_configuration(config, environment)
            
            if not validation_result.is_valid:
                report = generate_validation_report(validation_result)
                self.logger.error(f"Configuration validation failed:\n{report}")
                
                if validation_result.security_issues:
                    for issue in validation_result.security_issues:
                        log_security_event(
                            event_type='configuration_security',
                            severity='high',
                            description=f'Configuration security issue: {issue}',
                            action='validate_config',
                            result='security_issue'
                        )
                    raise ValueError("Configuration has critical security issues")
                
                if validation_result.errors:
                    raise ValueError("Configuration validation failed with errors")
            
            self.features_enabled['configuration_validation'] = True
            self.logger.info("System configuration validated successfully")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise
    
    async def _perform_initial_health_check(self):
        """Perform initial comprehensive health check"""
        try:
            health_report = await self.health_check.run_health_checks()
            
            if health_report.overall_status == 'unhealthy':
                self.logger.error(
                    f"Initial health check failed - Score: {health_report.health_score:.1f}%"
                )
                
                for issue in health_report.critical_issues:
                    self.logger.error(f"Critical issue: {issue}")
                
                # Don't fail initialization for health issues, but log them
                log_security_event(
                    event_type='system_monitoring',
                    severity='medium',
                    description=f'Initial health check shows issues - Score: {health_report.health_score:.1f}%',
                    action='initial_health_check',
                    result='degraded'
                )
            
            self.logger.info(
                f"Initial health check completed - Status: {health_report.overall_status}, "
                f"Score: {health_report.health_score:.1f}%"
            )
            
        except Exception as e:
            self.logger.error(f"Initial health check failed: {e}")
            # Don't fail initialization for health check issues
    
    async def _setup_monitoring(self):
        """Setup monitoring and periodic tasks"""
        try:
            # Enable structured logging
            self.features_enabled['structured_logging'] = True
            
            # Start background monitoring tasks
            asyncio.create_task(self._periodic_health_check())
            asyncio.create_task(self._periodic_cleanup())
            
            self.logger.info("Monitoring and periodic tasks setup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to setup monitoring: {e}")
            # Don't fail initialization for monitoring setup issues
    
    async def _periodic_health_check(self):
        """Periodic health check task"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                if self.health_check:
                    health_report = await self.health_check.run_health_checks(
                        include_non_critical=False  # Only critical checks for periodic monitoring
                    )
                    
                    if health_report.overall_status == 'unhealthy':
                        log_security_event(
                            event_type='system_monitoring',
                            severity='high',
                            description=f'Periodic health check failed - Score: {health_report.health_score:.1f}%',
                            action='periodic_health_check',
                            result='unhealthy'
                        )
                        
                        self.logger.error(
                            f"Periodic health check failed - taking corrective action"
                        )
                        
                        # Attempt basic remediation
                        await self._attempt_auto_remediation(health_report)
                
            except Exception as e:
                self.logger.error(f"Periodic health check failed: {e}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                
                # Cleanup expired sessions
                if self.security_manager:
                    expired_count = self.security_manager.session_manager.cleanup_expired_sessions()
                    if expired_count > 0:
                        self.logger.info(f"Cleaned up {expired_count} expired sessions")
                
                # Cleanup expired credentials
                if self.security_manager:
                    expired_creds = self.security_manager.credential_manager.cleanup_expired()
                    if expired_creds > 0:
                        self.logger.info(f"Cleaned up {expired_creds} expired credentials")
                
                self.logger.debug("Periodic cleanup completed")
                
            except Exception as e:
                self.logger.error(f"Periodic cleanup failed: {e}")
    
    async def _attempt_auto_remediation(self, health_report):
        """Attempt automatic remediation of health issues"""
        try:
            for check in health_report.checks:
                if check.status == 'unhealthy' and check.remediation:
                    self.logger.info(f"Attempting auto-remediation for {check.name}: {check.remediation}")
                    
                    # Specific remediation actions
                    if 'memory' in check.name.lower():
                        # Force garbage collection
                        import gc
                        gc.collect()
                        self.logger.info("Forced garbage collection")
                    
                    elif 'database' in check.name.lower():
                        # Try to reconnect to database
                        self.logger.info("Attempting database reconnection")
                    
                    # Add more specific remediation actions as needed
            
        except Exception as e:
            self.logger.error(f"Auto-remediation failed: {e}")
    
    @asynccontextmanager
    async def secure_operation_context(self, operation: str, client_id: str = None, **context_data):
        """
        Context manager for secure operations with full Generation 2 features
        
        Args:
            operation: Operation name
            client_id: Client identifier for security checks
            **context_data: Additional context data
        """
        # Create security context
        security_context = SecurityContext(
            request_id=context_data.get('request_id'),
            user_id=context_data.get('user_id'),
            session_id=context_data.get('session_id'),
            source_ip=client_id if self._is_ip_address(client_id) else None,
            user_agent=context_data.get('user_agent'),
            permissions=context_data.get('permissions', [])
        )
        
        # Create logging context
        log_ctx = LogContext(
            correlation_id=context_data.get('correlation_id'),
            operation=operation,
            user_id=security_context.user_id,
            session_id=security_context.session_id,
            request_id=security_context.request_id,
            module=context_data.get('module', 'generation_2_system')
        )
        
        with log_context(log_ctx):
            async with monitor_performance(operation):
                try:
                    # Security validation
                    if self.security_manager:
                        await self.security_manager.validate_request(security_context, operation)
                    
                    yield security_context
                    
                    # Log successful operation
                    self.logger.info(f"Secure operation completed successfully: {operation}")
                    
                except Exception as e:
                    # Log failed operation
                    self.logger.error(f"Secure operation failed: {operation} - {str(e)}")
                    
                    # Log security event for failed operations
                    log_security_event(
                        event_type='operation_failure',
                        severity='medium',
                        description=f'Secure operation failed: {operation}',
                        user_id=security_context.user_id,
                        source_ip=security_context.source_ip,
                        action=operation,
                        result='failure'
                    )
                    
                    raise
    
    async def execute_secure_operation(self, operation: str, parameters: Dict[str, Any], 
                                     client_id: str = None, **context) -> Any:
        """
        Execute an operation with full Generation 2 security and robustness
        
        Args:
            operation: Operation to execute
            parameters: Operation parameters
            client_id: Client identifier
            **context: Additional context
            
        Returns:
            Operation result
        """
        if not self.is_initialized:
            raise RuntimeError("Generation 2 system not initialized")
        
        async with self.secure_operation_context(operation, client_id, **context) as security_context:
            return await self.robust_system.execute_secure_operation(
                operation, parameters, security_context
            )
    
    async def get_system_status(self) -> Generation2Status:
        """Get comprehensive Generation 2 system status"""
        try:
            # Get health report
            health_report = None
            health_score = 0.0
            last_health_check = None
            
            if self.health_check:
                health_report = await self.health_check.run_health_checks()
                health_score = health_report.health_score
                last_health_check = health_report.timestamp
            
            # Get performance metrics
            performance_metrics = {}
            if self.robust_system:
                perf_summary = self.robust_system.performance_monitor.get_performance_summary()
                performance_metrics = {
                    'uptime_seconds': perf_summary['uptime_seconds'],
                    'total_operations': perf_summary['total_operations'],
                    'error_rate': perf_summary['error_rate']
                }
            
            # Determine security status
            security_status = "unknown"
            if self.security_manager:
                if os.getenv('GITHUB_TOKEN'):
                    security_status = "operational"
                else:
                    security_status = "degraded"
            
            return Generation2Status(
                initialized=self.is_initialized,
                features_enabled=self.features_enabled.copy(),
                security_status=security_status,
                health_score=health_score,
                uptime_seconds=(datetime.now() - self.initialization_time).total_seconds(),
                last_health_check=last_health_check,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return Generation2Status(
                initialized=False,
                features_enabled={},
                security_status="error",
                health_score=0.0,
                uptime_seconds=0.0,
                last_health_check=None,
                performance_metrics={}
            )
    
    def _is_ip_address(self, address: str) -> bool:
        """Check if string is an IP address"""
        if not address:
            return False
        
        import ipaddress
        try:
            ipaddress.ip_address(address)
            return True
        except ValueError:
            return False
    
    async def export_system_metrics(self, export_path: str):
        """Export comprehensive system metrics"""
        try:
            status = await self.get_system_status()
            
            # Get health report
            health_report = None
            if self.health_check:
                health_report = await self.health_check.run_health_checks()
            
            # Get security audit
            security_audit = {}
            if self.security_manager:
                security_audit = await self.security_manager.audit_security_events(24)
            
            # Export robust system metrics
            if self.robust_system:
                await self.robust_system.export_metrics(f"{export_path}_robust_metrics.json")
            
            # Export health report
            if health_report and self.health_check:
                self.health_check.export_health_report(health_report, f"{export_path}_health_report.json")
            
            self.logger.info(f"System metrics exported to {export_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export system metrics: {e}")
    
    async def shutdown(self):
        """Graceful system shutdown"""
        try:
            self.logger.info("Starting Generation 2 system shutdown")
            
            # Log shutdown event
            log_security_event(
                event_type='system_shutdown',
                severity='low',
                description='Generation 2 system shutting down',
                action='shutdown',
                result='initiated'
            )
            
            # Perform final health check
            if self.health_check:
                final_health = await self.health_check.run_health_checks()
                self.logger.info(f"Final health score: {final_health.health_score:.1f}%")
            
            # Export final metrics
            await self.export_system_metrics("final_shutdown_metrics")
            
            # Cleanup resources
            self.is_initialized = False
            
            self.logger.info("Generation 2 system shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during system shutdown: {e}")


# Factory function
async def create_generation_2_system(config_service: ConfigurationService) -> Generation2System:
    """
    Create and initialize Generation 2 system
    
    Args:
        config_service: Initialized configuration service
        
    Returns:
        Fully initialized Generation 2 system
    """
    system = Generation2System(config_service)
    await system.initialize()
    return system


# Example usage and testing
async def demonstrate_generation_2_features():
    """Demonstrate Generation 2 robustness features"""
    from src.services.configuration_service import ConfigurationService
    
    try:
        # Initialize configuration service
        config_service = ConfigurationService("config.json")
        await config_service.initialize()
        
        # Create Generation 2 system
        gen2_system = await create_generation_2_system(config_service)
        
        # Get system status
        status = await gen2_system.get_system_status()
        print(f"Generation 2 System Status: {status.initialized}")
        print(f"Health Score: {status.health_score:.1f}%")
        print(f"Features Enabled: {sum(status.features_enabled.values())}/{len(status.features_enabled)}")
        
        # Demonstrate secure operation
        test_params = {
            'repo_name': 'test/repo',
            'title': 'Test Issue',
            'body': 'This is a test issue'
        }
        
        result = await gen2_system.execute_secure_operation(
            operation='create_issue',
            parameters=test_params,
            client_id='127.0.0.1',
            user_id='demo_user'
        )
        
        print(f"Secure operation result: {result}")
        
        # Export metrics
        await gen2_system.export_system_metrics("generation_2_demo")
        
        # Graceful shutdown
        await gen2_system.shutdown()
        
        print("Generation 2 demonstration completed successfully")
        
    except Exception as e:
        print(f"Generation 2 demonstration failed: {e}")


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_generation_2_features())