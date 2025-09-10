#!/usr/bin/env python3
"""
System Resource Monitor
Мониторинг системных ресурсов
"""

import psutil
import time
import json
import os
from datetime import datetime
from typing import Dict, Any

class SystemMonitor:
    """Мониторинг системных ресурсов"""
    
    def __init__(self):
        self.start_time = time.time()
        
    def get_cpu_stats(self) -> Dict[str, Any]:
        """Статистика CPU"""
        cpu_stats = {
            'usage_percent': psutil.cpu_percent(interval=1),
            'core_count': psutil.cpu_count()
        }
        
        # Load average (Unix only) - skip type checking for dynamic attribute
        if hasattr(os, 'getloadavg'):
            try:
                cpu_stats['load_average'] = getattr(os, 'getloadavg')()
            except:
                cpu_stats['load_average'] = None
        else:
            # Windows doesn't have getloadavg
            cpu_stats['load_average'] = None
            
        return cpu_stats
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Статистика памяти"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': round(memory.total / (1024**3), 2),
            'available_gb': round(memory.available / (1024**3), 2),
            'used_gb': round(memory.used / (1024**3), 2),
            'usage_percent': memory.percent
        }
    
    def get_disk_stats(self) -> Dict[str, Any]:
        """Статистика диска"""
        disk = psutil.disk_usage('/')
        return {
            'total_gb': round(disk.total / (1024**3), 2),
            'free_gb': round(disk.free / (1024**3), 2),
            'used_gb': round(disk.used / (1024**3), 2),
            'usage_percent': round((disk.used / disk.total) * 100, 2)
        }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Статистика сети"""
        net = psutil.net_io_counters()
        return {
            'bytes_sent': net.bytes_sent,
            'bytes_received': net.bytes_recv,
            'packets_sent': net.packets_sent,
            'packets_received': net.packets_recv
        }
    
    def get_process_stats(self) -> Dict[str, Any]:
        """Статистика процессов Python"""
        stats = {
            'total_processes': len(psutil.pids()),
            'python_processes': []
        }
        
        for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
            try:
                if 'python' in proc.info['name'].lower():
                    stats['python_processes'].append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'memory_percent': proc.info['memory_percent'],
                        'cpu_percent': proc.info['cpu_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return stats
    
    def monitor_loop(self, interval: int = 60):
        """Основной цикл мониторинга"""
        print("🚀 Starting system resource monitoring...")
        
        while True:
            try:
                timestamp = datetime.now().isoformat()
                
                # Collect all metrics
                metrics = {
                    'timestamp': timestamp,
                    'uptime_seconds': int(time.time() - self.start_time),
                    'cpu': self.get_cpu_stats(),
                    'memory': self.get_memory_stats(),
                    'disk': self.get_disk_stats(),
                    'network': self.get_network_stats(),
                    'processes': self.get_process_stats()
                }
                
                # Print summary
                print(f"\n🖥️  System Resources - {timestamp}")
                print(f"🔥 CPU Usage: {metrics['cpu']['usage_percent']:.1f}%")
                print(f"💾 Memory Usage: {metrics['memory']['usage_percent']:.1f}% ({metrics['memory']['used_gb']:.1f}GB/{metrics['memory']['total_gb']:.1f}GB)")
                print(f"💿 Disk Usage: {metrics['disk']['usage_percent']:.1f}% ({metrics['disk']['used_gb']:.1f}GB/{metrics['disk']['total_gb']:.1f}GB)")
                print(f"🐍 Python Processes: {len(metrics['processes']['python_processes'])}")
                
                # Save metrics
                metrics_file = os.path.join(
                    os.path.dirname(__file__), 
                    '..', 
                    'metrics', 
                    'system_metrics.json'
                )
                
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\n👋 System monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ System monitoring error: {e}")
                time.sleep(interval)

def main():
    """Запуск мониторинга"""
    try:
        monitor = SystemMonitor()
        monitor.monitor_loop()
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install psutil: pip install psutil")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
