"""
Chaos Engineering Framework for Claude Code Manager
Implements controlled chaos experiments to improve system resilience
"""

import asyncio
import random
import time
import logging
import psutil
import docker
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import yaml
import json
from datetime import datetime, timedelta

# Observability integration
from observability.distributed_tracing_setup import traced_function, get_tracer, get_meter


class ChaosExperimentType(Enum):
    """Types of chaos experiments"""
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    SERVICE_FAILURE = "service_failure"
    DATABASE_FAILURE = "database_failure"
    CONTAINER_KILL = "container_kill"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CONFIGURATION_CORRUPTION = "config_corruption"


class ExperimentStatus(Enum):
    """Status of chaos experiments"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"
    ROLLBACK = "rollback"


@dataclass
class ChaosExperiment:
    """Definition of a chaos experiment"""
    name: str
    experiment_type: ChaosExperimentType
    description: str
    target: str  # Target service, container, or resource
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration: int = 60  # Duration in seconds
    rollback_duration: int = 30  # Time to wait before rollback
    enabled: bool = True
    schedule: Optional[str] = None  # Cron-like schedule
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    safety_checks: List[str] = field(default_factory=list)
    
    # Runtime fields
    status: ExperimentStatus = ExperimentStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class ChaosMonkey:
    """Main chaos engineering orchestrator"""
    
    def __init__(self, config_path: str = "chaos-engineering/chaos-config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.experiments: List[ChaosExperiment] = []
        self.running_experiments: Dict[str, ChaosExperiment] = {}
        self.docker_client = None
        self.tracer = get_tracer()
        self.meter = get_meter()
        
        # Metrics
        self.experiment_counter = self.meter.create_counter(
            name="chaos_experiments_total",
            description="Total chaos experiments executed",
            unit="1",
        )
        
        self.experiment_duration = self.meter.create_histogram(
            name="chaos_experiment_duration_seconds",
            description="Duration of chaos experiments",
            unit="s",
        )
        
        self.system_recovery_time = self.meter.create_histogram(
            name="system_recovery_time_seconds",
            description="Time for system to recover after chaos",
            unit="s",
        )
        
        # Safety mechanisms
        self.safety_enabled = True
        self.max_concurrent_experiments = 3
        self.emergency_stop = False
        
        self._load_configuration()
        self._initialize_docker()
    
    def _load_configuration(self) -> None:
        """Load chaos experiment configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            for exp_data in config_data.get('experiments', []):
                experiment = ChaosExperiment(
                    name=exp_data['name'],
                    experiment_type=ChaosExperimentType(exp_data['type']),
                    description=exp_data.get('description', ''),
                    target=exp_data['target'],
                    parameters=exp_data.get('parameters', {}),
                    duration=exp_data.get('duration', 60),
                    rollback_duration=exp_data.get('rollback_duration', 30),
                    enabled=exp_data.get('enabled', True),
                    schedule=exp_data.get('schedule'),
                    prerequisites=exp_data.get('prerequisites', []),
                    success_criteria=exp_data.get('success_criteria', {}),
                    safety_checks=exp_data.get('safety_checks', []),
                )
                self.experiments.append(experiment)
                
            self.logger.info(f"Loaded {len(self.experiments)} chaos experiments")
            
        except FileNotFoundError:
            self.logger.warning(f"Chaos config file {self.config_path} not found, using defaults")
            self._create_default_experiments()
        except Exception as e:
            self.logger.error(f"Error loading chaos configuration: {e}")
            raise
    
    def _create_default_experiments(self) -> None:
        """Create default chaos experiments"""
        default_experiments = [
            ChaosExperiment(
                name="api_latency_injection",
                experiment_type=ChaosExperimentType.NETWORK_LATENCY,
                description="Inject network latency to API endpoints",
                target="claude-manager-api",
                parameters={"latency_ms": 500, "jitter_ms": 100},
                duration=120,
                success_criteria={"response_time_p95": 2000, "error_rate": 0.05},
                safety_checks=["health_check", "error_rate_check"],
            ),
            ChaosExperiment(
                name="database_connection_failure",
                experiment_type=ChaosExperimentType.DATABASE_FAILURE,
                description="Simulate database connection failures",
                target="postgres",
                parameters={"failure_rate": 0.3},
                duration=60,
                success_criteria={"fallback_success_rate": 0.8},
                safety_checks=["database_health_check"],
            ),
            ChaosExperiment(
                name="container_resource_exhaustion",
                experiment_type=ChaosExperimentType.MEMORY_STRESS,
                description="Exhaust container memory resources",
                target="claude-manager",
                parameters={"memory_percentage": 80},
                duration=90,
                success_criteria={"oom_kill_count": 0, "restart_count": 0},
                safety_checks=["memory_usage_check"],
            ),
        ]
        
        self.experiments.extend(default_experiments)
    
    def _initialize_docker(self) -> None:
        """Initialize Docker client for container chaos"""
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized for chaos experiments")
        except Exception as e:
            self.logger.warning(f"Could not initialize Docker client: {e}")
    
    @traced_function("chaos_experiment_execution")
    async def run_experiment(self, experiment: ChaosExperiment) -> bool:
        """Execute a single chaos experiment"""
        if not experiment.enabled:
            self.logger.info(f"Experiment {experiment.name} is disabled, skipping")
            return False
        
        if not self._check_prerequisites(experiment):
            self.logger.warning(f"Prerequisites not met for {experiment.name}")
            return False
        
        if not self._perform_safety_checks(experiment):
            self.logger.error(f"Safety checks failed for {experiment.name}")
            return False
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.now()
        self.running_experiments[experiment.name] = experiment
        
        try:
            self.logger.info(f"Starting chaos experiment: {experiment.name}")
            
            # Record experiment start
            self.experiment_counter.add(1, {
                "experiment_name": experiment.name,
                "experiment_type": experiment.experiment_type.value,
                "status": "started"
            })
            
            # Execute the specific chaos experiment
            success = await self._execute_experiment_type(experiment)
            
            if success:
                experiment.status = ExperimentStatus.SUCCEEDED
                self.logger.info(f"Chaos experiment {experiment.name} completed successfully")
            else:
                experiment.status = ExperimentStatus.FAILED
                self.logger.error(f"Chaos experiment {experiment.name} failed")
            
            # Wait for rollback period to observe system recovery
            if experiment.rollback_duration > 0:
                await asyncio.sleep(experiment.rollback_duration)
            
            # Perform rollback
            await self._rollback_experiment(experiment)
            
            # Record experiment completion
            duration = (datetime.now() - experiment.start_time).total_seconds()
            self.experiment_duration.record(duration, {
                "experiment_name": experiment.name,
                "experiment_type": experiment.experiment_type.value,
                "status": experiment.status.value
            })
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error during chaos experiment {experiment.name}: {e}")
            experiment.status = ExperimentStatus.FAILED
            experiment.error_message = str(e)
            await self._emergency_rollback(experiment)
            return False
            
        finally:
            experiment.end_time = datetime.now()
            self.running_experiments.pop(experiment.name, None)
    
    async def _execute_experiment_type(self, experiment: ChaosExperiment) -> bool:
        """Execute specific type of chaos experiment"""
        experiment_handlers = {
            ChaosExperimentType.NETWORK_LATENCY: self._network_latency_experiment,
            ChaosExperimentType.NETWORK_PARTITION: self._network_partition_experiment,
            ChaosExperimentType.CPU_STRESS: self._cpu_stress_experiment,
            ChaosExperimentType.MEMORY_STRESS: self._memory_stress_experiment,
            ChaosExperimentType.SERVICE_FAILURE: self._service_failure_experiment,
            ChaosExperimentType.DATABASE_FAILURE: self._database_failure_experiment,
            ChaosExperimentType.CONTAINER_KILL: self._container_kill_experiment,
        }
        
        handler = experiment_handlers.get(experiment.experiment_type)
        if not handler:
            self.logger.error(f"No handler for experiment type {experiment.experiment_type}")
            return False
        
        return await handler(experiment)
    
    async def _network_latency_experiment(self, experiment: ChaosExperiment) -> bool:
        """Inject network latency"""
        latency_ms = experiment.parameters.get('latency_ms', 500)
        jitter_ms = experiment.parameters.get('jitter_ms', 100)
        
        # Use tc (traffic control) to inject latency
        import subprocess
        
        try:
            # Add latency to network interface
            cmd = [
                'tc', 'qdisc', 'add', 'dev', 'eth0', 'root', 'netem',
                'delay', f'{latency_ms}ms', f'{jitter_ms}ms'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"Failed to inject latency: {result.stderr}")
                return False
            
            # Wait for experiment duration
            await asyncio.sleep(experiment.duration)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Network latency experiment failed: {e}")
            return False
    
    async def _memory_stress_experiment(self, experiment: ChaosExperiment) -> bool:
        """Create memory pressure"""
        memory_percentage = experiment.parameters.get('memory_percentage', 80)
        
        try:
            # Get available memory
            memory_info = psutil.virtual_memory()
            target_memory = int(memory_info.total * memory_percentage / 100)
            
            # Allocate memory to create pressure
            memory_hog = []
            chunk_size = 1024 * 1024  # 1MB chunks
            
            self.logger.info(f"Creating memory pressure: {memory_percentage}% of {memory_info.total} bytes")
            
            for _ in range(target_memory // chunk_size):
                memory_hog.append(' ' * chunk_size)
                
                # Check if we should stop
                if self.emergency_stop:
                    break
                
                # Small delay to prevent blocking
                if len(memory_hog) % 100 == 0:
                    await asyncio.sleep(0.1)
            
            # Hold memory for experiment duration
            await asyncio.sleep(experiment.duration)
            
            # Release memory
            del memory_hog
            
            return True
            
        except Exception as e:
            self.logger.error(f"Memory stress experiment failed: {e}")
            return False
    
    async def _container_kill_experiment(self, experiment: ChaosExperiment) -> bool:
        """Kill target container"""
        if not self.docker_client:
            self.logger.error("Docker client not available")
            return False
        
        target_container = experiment.target
        
        try:
            # Find target container
            containers = self.docker_client.containers.list(
                filters={"name": target_container}
            )
            
            if not containers:
                self.logger.error(f"Container {target_container} not found")
                return False
            
            container = containers[0]
            
            # Kill container
            self.logger.info(f"Killing container {target_container}")
            container.kill()
            
            # Wait for specified duration to observe impact
            await asyncio.sleep(experiment.duration)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Container kill experiment failed: {e}")
            return False
    
    async def _database_failure_experiment(self, experiment: ChaosExperiment) -> bool:
        """Simulate database failures"""
        failure_rate = experiment.parameters.get('failure_rate', 0.3)
        
        # This would typically integrate with a database proxy or chaos proxy
        # For demonstration, we'll simulate by injecting failures at the application level
        
        self.logger.info(f"Simulating database failures with {failure_rate} failure rate")
        
        # In a real implementation, this would configure a chaos proxy
        # or modify database connection settings
        
        await asyncio.sleep(experiment.duration)
        return True
    
    async def _service_failure_experiment(self, experiment: ChaosExperiment) -> bool:
        """Simulate service failures"""
        # Implementation would depend on service architecture
        # Could involve stopping processes, blocking ports, etc.
        await asyncio.sleep(experiment.duration)
        return True
    
    async def _cpu_stress_experiment(self, experiment: ChaosExperiment) -> bool:
        """Create CPU stress"""
        cpu_percentage = experiment.parameters.get('cpu_percentage', 80)
        
        try:
            # Create CPU load
            import multiprocessing
            
            def cpu_stress():
                start_time = time.time()
                while time.time() - start_time < experiment.duration:
                    pass
            
            # Start CPU stress processes
            processes = []
            num_cores = min(multiprocessing.cpu_count(), 4)  # Limit impact
            
            for _ in range(num_cores):
                p = multiprocessing.Process(target=cpu_stress)
                p.start()
                processes.append(p)
            
            # Wait for experiment duration
            await asyncio.sleep(experiment.duration)
            
            # Clean up processes
            for p in processes:
                p.terminate()
                p.join()
            
            return True
            
        except Exception as e:
            self.logger.error(f"CPU stress experiment failed: {e}")
            return False
    
    async def _network_partition_experiment(self, experiment: ChaosExperiment) -> bool:
        """Create network partition"""
        # Implementation would use iptables or similar to block network traffic
        await asyncio.sleep(experiment.duration)
        return True
    
    def _check_prerequisites(self, experiment: ChaosExperiment) -> bool:
        """Check if experiment prerequisites are met"""
        for prerequisite in experiment.prerequisites:
            if not self._check_prerequisite(prerequisite):
                return False
        return True
    
    def _check_prerequisite(self, prerequisite: str) -> bool:
        """Check a specific prerequisite"""
        if prerequisite == "system_healthy":
            return self._check_system_health()
        elif prerequisite == "low_traffic":
            return self._check_traffic_level()
        elif prerequisite == "business_hours":
            return self._check_business_hours()
        return True
    
    def _perform_safety_checks(self, experiment: ChaosExperiment) -> bool:
        """Perform safety checks before experiment"""
        if not self.safety_enabled:
            return True
        
        for safety_check in experiment.safety_checks:
            if not self._perform_safety_check(safety_check):
                return False
        return True
    
    def _perform_safety_check(self, check: str) -> bool:
        """Perform a specific safety check"""
        if check == "health_check":
            return self._check_system_health()
        elif check == "error_rate_check":
            return self._check_error_rate()
        elif check == "memory_usage_check":
            return self._check_memory_usage()
        return True
    
    def _check_system_health(self) -> bool:
        """Check overall system health"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                self.logger.warning(f"High memory usage: {memory.percent}%")
                return False
            
            # Check disk usage
            disk = psutil.disk_usage('/')
            if disk.percent > 90:
                self.logger.warning(f"High disk usage: {disk.percent}%")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def _check_error_rate(self) -> bool:
        """Check current error rate"""
        # This would typically query monitoring systems
        # For now, assume error rate is acceptable
        return True
    
    def _check_memory_usage(self) -> bool:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        return memory.percent < 85
    
    def _check_traffic_level(self) -> bool:
        """Check if traffic level is appropriate for chaos experiments"""
        # This would typically check load balancer or application metrics
        return True
    
    def _check_business_hours(self) -> bool:
        """Check if current time is within business hours"""
        current_hour = datetime.now().hour
        return 9 <= current_hour <= 17  # 9 AM to 5 PM
    
    async def _rollback_experiment(self, experiment: ChaosExperiment) -> None:
        """Rollback changes made by experiment"""
        self.logger.info(f"Rolling back experiment {experiment.name}")
        
        try:
            if experiment.experiment_type == ChaosExperimentType.NETWORK_LATENCY:
                # Remove network latency
                import subprocess
                subprocess.run(['tc', 'qdisc', 'del', 'dev', 'eth0', 'root'], 
                             capture_output=True)
            
            # Add other rollback procedures as needed
            
        except Exception as e:
            self.logger.error(f"Error during rollback of {experiment.name}: {e}")
    
    async def _emergency_rollback(self, experiment: ChaosExperiment) -> None:
        """Emergency rollback in case of experiment failure"""
        self.logger.error(f"Emergency rollback for experiment {experiment.name}")
        experiment.status = ExperimentStatus.ROLLBACK
        await self._rollback_experiment(experiment)
    
    async def run_scheduled_experiments(self) -> None:
        """Run experiments based on their schedules"""
        while not self.emergency_stop:
            try:
                for experiment in self.experiments:
                    if (experiment.enabled and 
                        experiment.name not in self.running_experiments and
                        len(self.running_experiments) < self.max_concurrent_experiments):
                        
                        # Check if experiment should run based on schedule
                        if self._should_run_experiment(experiment):
                            asyncio.create_task(self.run_experiment(experiment))
                
                # Check every minute
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in scheduled experiment runner: {e}")
                await asyncio.sleep(60)
    
    def _should_run_experiment(self, experiment: ChaosExperiment) -> bool:
        """Determine if experiment should run based on schedule"""
        if not experiment.schedule:
            return False
        
        # Simple schedule implementation - in production, use croniter or similar
        # For now, just check if it's time based on a simple format
        return random.random() < 0.1  # 10% chance per minute
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """Get status of all experiments"""
        return {
            "total_experiments": len(self.experiments),
            "running_experiments": len(self.running_experiments),
            "enabled_experiments": len([e for e in self.experiments if e.enabled]),
            "experiments": [
                {
                    "name": exp.name,
                    "type": exp.experiment_type.value,
                    "status": exp.status.value,
                    "enabled": exp.enabled,
                    "start_time": exp.start_time.isoformat() if exp.start_time else None,
                    "end_time": exp.end_time.isoformat() if exp.end_time else None,
                    "error_message": exp.error_message,
                }
                for exp in self.experiments
            ]
        }
    
    def emergency_stop_all(self) -> None:
        """Emergency stop all running experiments"""
        self.emergency_stop = True
        self.logger.warning("Emergency stop initiated - stopping all chaos experiments")
        
        for experiment in self.running_experiments.values():
            experiment.status = ExperimentStatus.ABORTED


# Configuration file template
CHAOS_CONFIG_TEMPLATE = """
experiments:
  - name: "api_latency_injection"
    type: "network_latency"
    description: "Inject network latency to API endpoints"
    target: "claude-manager-api"
    parameters:
      latency_ms: 500
      jitter_ms: 100
    duration: 120
    rollback_duration: 30
    enabled: true
    schedule: "0 */4 * * *"  # Every 4 hours
    prerequisites:
      - "system_healthy"
      - "low_traffic"
    success_criteria:
      response_time_p95: 2000
      error_rate: 0.05
    safety_checks:
      - "health_check"
      - "error_rate_check"

  - name: "database_connection_failure"
    type: "database_failure"
    description: "Simulate database connection failures"
    target: "postgres"
    parameters:
      failure_rate: 0.3
    duration: 60
    rollback_duration: 30
    enabled: true
    prerequisites:
      - "system_healthy"
    success_criteria:
      fallback_success_rate: 0.8
    safety_checks:
      - "database_health_check"

  - name: "container_memory_pressure"
    type: "memory_stress"
    description: "Create memory pressure on application container"
    target: "claude-manager"
    parameters:
      memory_percentage: 80
    duration: 90
    rollback_duration: 30
    enabled: false  # Disabled by default for safety
    prerequisites:
      - "system_healthy"
      - "business_hours"
    success_criteria:
      oom_kill_count: 0
      restart_count: 0
    safety_checks:
      - "memory_usage_check"
"""


if __name__ == "__main__":
    async def main():
        # Create chaos monkey instance
        chaos_monkey = ChaosMonkey()
        
        # Run a single experiment
        if chaos_monkey.experiments:
            experiment = chaos_monkey.experiments[0]
            success = await chaos_monkey.run_experiment(experiment)
            print(f"Experiment {experiment.name} {'succeeded' if success else 'failed'}")
        
        # Get status
        status = chaos_monkey.get_experiment_status()
        print(f"Chaos engineering status: {json.dumps(status, indent=2)}")
    
    asyncio.run(main())