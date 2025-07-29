#!/usr/bin/env python3
"""
Resource monitoring script for performance analysis.
Monitors CPU, memory, disk, and network usage over time.
"""

import argparse
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Any

try:
    import psutil
except ImportError:
    print("Error: psutil is required. Install with: pip install psutil")
    sys.exit(1)


class ResourceMonitor:
    """Monitor system resource usage over time."""
    
    def __init__(self):
        self.data: List[Dict[str, Any]] = []
        
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        timestamp = datetime.now().isoformat()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        # Process metrics
        process_count = len(psutil.pids())
        
        return {
            'timestamp': timestamp,
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'load_avg_1min': load_avg[0],
                'load_avg_5min': load_avg[1],
                'load_avg_15min': load_avg[2]
            },
            'memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent': memory.percent,
                'cached_gb': round(memory.cached / (1024**3), 2) if hasattr(memory, 'cached') else 0,
                'buffers_gb': round(memory.buffers / (1024**3), 2) if hasattr(memory, 'buffers') else 0
            },
            'swap': {
                'total_gb': round(swap.total / (1024**3), 2),
                'used_gb': round(swap.used / (1024**3), 2),
                'percent': swap.percent
            },
            'disk': {
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'used_gb': round(disk_usage.used / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'percent': round((disk_usage.used / disk_usage.total) * 100, 2),
                'read_mb': round(disk_io.read_bytes / (1024**2), 2) if disk_io else 0,
                'write_mb': round(disk_io.write_bytes / (1024**2), 2) if disk_io else 0
            },
            'network': {
                'bytes_sent_mb': round(network_io.bytes_sent / (1024**2), 2) if network_io else 0,
                'bytes_recv_mb': round(network_io.bytes_recv / (1024**2), 2) if network_io else 0,
                'packets_sent': network_io.packets_sent if network_io else 0,
                'packets_recv': network_io.packets_recv if network_io else 0
            },
            'processes': {
                'count': process_count
            }
        }
    
    def monitor(self, duration: int, interval: int = 10) -> None:
        """Monitor resources for specified duration."""
        print(f"Starting resource monitoring for {duration} seconds...")
        print(f"Collection interval: {interval} seconds")
        
        end_time = time.time() + duration
        
        while time.time() < end_time:
            try:
                metrics = self.collect_metrics()
                self.data.append(metrics)
                
                # Print current status
                cpu_pct = metrics['cpu']['percent']
                mem_pct = metrics['memory']['percent']
                disk_pct = metrics['disk']['percent']
                
                print(f"[{metrics['timestamp']}] CPU: {cpu_pct:5.1f}% | "
                      f"Memory: {mem_pct:5.1f}% | "
                      f"Disk: {disk_pct:5.1f}%")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\nMonitoring interrupted by user")
                break
            except Exception as e:
                print(f"Error collecting metrics: {e}")
                time.sleep(interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from collected data."""
        if not self.data:
            return {}
        
        # Extract time series data
        cpu_data = [d['cpu']['percent'] for d in self.data]
        memory_data = [d['memory']['percent'] for d in self.data]
        disk_data = [d['disk']['percent'] for d in self.data]
        
        # Calculate statistics
        def calc_stats(data: List[float]) -> Dict[str, float]:
            return {
                'min': min(data),
                'max': max(data),
                'avg': sum(data) / len(data),
                'count': len(data)
            }
        
        return {
            'duration_seconds': len(self.data) * 10,  # Assuming 10s intervals
            'data_points': len(self.data),
            'cpu_stats': calc_stats(cpu_data),
            'memory_stats': calc_stats(memory_data),
            'disk_stats': calc_stats(disk_data),
            'peak_usage': {
                'timestamp': max(self.data, key=lambda x: x['cpu']['percent'])['timestamp'],
                'cpu_percent': max(cpu_data),
                'memory_percent': max(memory_data),
                'disk_percent': max(disk_data)
            },
            'system_info': {
                'cpu_count': self.data[0]['cpu']['count'] if self.data else 0,
                'total_memory_gb': self.data[0]['memory']['total_gb'] if self.data else 0,
                'total_disk_gb': self.data[0]['disk']['total_gb'] if self.data else 0
            }
        }
    
    def save_data(self, output_file: str) -> None:
        """Save collected data to JSON file."""
        output = {
            'collection_info': {
                'start_time': self.data[0]['timestamp'] if self.data else None,
                'end_time': self.data[-1]['timestamp'] if self.data else None,
                'data_points': len(self.data)
            },
            'summary': self.get_summary(),
            'raw_data': self.data
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Resource monitoring data saved to: {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Monitor system resource usage over time"
    )
    parser.add_argument(
        '--duration', 
        type=int, 
        default=300,
        help='Monitoring duration in seconds (default: 300)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Collection interval in seconds (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='resource-usage.json',
        help='Output file for collected data (default: resource-usage.json)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.duration <= 0:
        print("Error: Duration must be positive")
        sys.exit(1)
    
    if args.interval <= 0:
        print("Error: Interval must be positive")
        sys.exit(1)
    
    # Initialize monitor
    monitor = ResourceMonitor()
    
    try:
        # Start monitoring
        monitor.monitor(args.duration, args.interval)
        
        # Save results
        monitor.save_data(args.output)
        
        # Print summary
        summary = monitor.get_summary()
        if summary:
            print("\n" + "="*50)
            print("MONITORING SUMMARY")
            print("="*50)
            print(f"Duration: {summary['duration_seconds']} seconds")
            print(f"Data points: {summary['data_points']}")
            print(f"\nCPU Usage:")
            print(f"  Min: {summary['cpu_stats']['min']:.1f}%")
            print(f"  Max: {summary['cpu_stats']['max']:.1f}%")
            print(f"  Avg: {summary['cpu_stats']['avg']:.1f}%")
            print(f"\nMemory Usage:")
            print(f"  Min: {summary['memory_stats']['min']:.1f}%")
            print(f"  Max: {summary['memory_stats']['max']:.1f}%")
            print(f"  Avg: {summary['memory_stats']['avg']:.1f}%")
            print(f"\nDisk Usage:")
            print(f"  Min: {summary['disk_stats']['min']:.1f}%")
            print(f"  Max: {summary['disk_stats']['max']:.1f}%")
            print(f"  Avg: {summary['disk_stats']['avg']:.1f}%")
            print(f"\nSystem Info:")
            print(f"  CPU Cores: {summary['system_info']['cpu_count']}")
            print(f"  Total Memory: {summary['system_info']['total_memory_gb']:.1f} GB")
            print(f"  Total Disk: {summary['system_info']['total_disk_gb']:.1f} GB")
        
    except Exception as e:
        print(f"Error during monitoring: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()