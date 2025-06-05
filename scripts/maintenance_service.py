#!/usr/bin/env python3
"""
Maintenance and Health Check Service for Supply Chain Risk Intelligence System
Handles system maintenance, health checks, and performance optimization
"""

import os
import sys
import time
import logging
import asyncio
import psutil
import aiohttp
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import schedule

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager
from cache.manager import CacheManager

class MaintenanceService:
    """Comprehensive maintenance and health monitoring service"""
    
    def __init__(self):
        self.setup_logging()
        self.config = self.load_config()
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        
        # Health check endpoints
        self.health_endpoints = [
            {'name': 'API', 'url': 'http://api:8000/health'},
            {'name': 'Dashboard', 'url': 'http://dashboard:8501/_stcore/health'},
            {'name': 'Grafana', 'url': 'http://grafana:3000/api/health'}
        ]
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./logs/maintenance.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> Dict:
        """Load configuration from environment variables"""
        return {
            'DATABASE_URL': os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@postgres:5432/supply_chain_risk'),
            'REDIS_URL': os.getenv('REDIS_URL', 'redis://redis:6379'),
            'HEALTH_CHECK_INTERVAL': int(os.getenv('HEALTH_CHECK_INTERVAL', '300')),  # 5 minutes
            'CLEANUP_INTERVAL': int(os.getenv('CLEANUP_INTERVAL', '3600')),  # 1 hour
            'MAINTENANCE_ENABLED': os.getenv('MAINTENANCE_ENABLED', 'true').lower() == 'true',
            'ALERT_THRESHOLDS': {
                'cpu_usage': float(os.getenv('CPU_THRESHOLD', '80')),
                'memory_usage': float(os.getenv('MEMORY_THRESHOLD', '85')),
                'disk_usage': float(os.getenv('DISK_THRESHOLD', '90')),
                'response_time': float(os.getenv('RESPONSE_TIME_THRESHOLD', '5000'))
            }
        }
    
    async def check_system_health(self) -> Dict:
        """Comprehensive system health check"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {},
            'system_metrics': {},
            'alerts': []
        }
        
        try:
            # System resource checks
            health_status['system_metrics'] = await self._check_system_resources()
            
            # Database health
            health_status['components']['database'] = await self._check_database_health()
            
            # Cache health
            health_status['components']['cache'] = await self._check_cache_health()
            
            # Service health
            health_status['components']['services'] = await self._check_service_health()
            
            # Determine overall status
            health_status['overall_status'] = self._determine_overall_status(health_status)
            
            # Generate alerts
            health_status['alerts'] = self._generate_alerts(health_status)
            
            self.logger.info(f"Health check completed: {health_status['overall_status']}")
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check error: {str(e)}")
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
            return health_status
    
    async def _check_system_resources(self) -> Dict:
        """Check system resource usage"""
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory Usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk Usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load Average (Unix systems)
            try:
                load_avg = os.getloadavg()
            except (OSError, AttributeError):
                load_avg = (0, 0, 0)
            
            # Network I/O
            network = psutil.net_io_counters()
            
            return {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'load_average': {
                        '1min': load_avg[0],
                        '5min': load_avg[1],
                        '15min': load_avg[2]
                    }
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'usage_percent': memory_percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'usage_percent': disk_percent
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
            
        except Exception as e:
            self.logger.error(f"System resource check error: {str(e)}")
            return {'error': str(e)}
    
    async def _check_database_health(self) -> Dict:
        """Check database health and performance"""
        try:
            start_time = time.time()
            
            # Test database connection
            async with self.db_manager.get_async_session() as session:
                # Simple query to test connection
                result = await session.execute("SELECT 1")
                await result.fetchone()
                
                # Get database stats
                stats_query = """
                SELECT 
                    schemaname,
                    tablename,
                    n_tup_ins,
                    n_tup_upd,
                    n_tup_del,
                    n_live_tup,
                    n_dead_tup
                FROM pg_stat_user_tables
                ORDER BY n_live_tup DESC
                LIMIT 10;
                """
                
                stats_result = await session.execute(stats_query)
                table_stats = [dict(row) for row in stats_result.fetchall()]
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'table_stats': table_stats,
                'connection_pool': {
                    'size': self.db_manager.engine.pool.size(),
                    'checked_in': self.db_manager.engine.pool.checkedin(),
                    'checked_out': self.db_manager.engine.pool.checkedout()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Database health check error: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def _check_cache_health(self) -> Dict:
        """Check Redis cache health and performance"""
        try:
            start_time = time.time()
            
            # Test cache connection
            test_key = "health_check_test"
            test_value = {"timestamp": datetime.now().isoformat()}
            
            await self.cache_manager.set(test_key, test_value, ttl=60)
            retrieved_value = await self.cache_manager.get(test_key)
            await self.cache_manager.delete(test_key)
            
            response_time = (time.time() - start_time) * 1000
            
            # Get cache statistics
            cache_stats = await self.cache_manager.get_stats()
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'test_successful': retrieved_value == test_value,
                'statistics': cache_stats
            }
            
        except Exception as e:
            self.logger.error(f"Cache health check error: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    async def _check_service_health(self) -> Dict:
        """Check health of external services"""
        service_health = {}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for endpoint in self.health_endpoints:
                try:
                    start_time = time.time()
                    async with session.get(endpoint['url']) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        service_health[endpoint['name']] = {
                            'status': 'healthy' if response.status == 200 else 'unhealthy',
                            'status_code': response.status,
                            'response_time_ms': response_time
                        }
                        
                except Exception as e:
                    service_health[endpoint['name']] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
        
        return service_health
    
    def _determine_overall_status(self, health_status: Dict) -> str:
        """Determine overall system health status"""
        # Check for critical component failures
        if 'database' in health_status['components']:
            db_status = health_status['components']['database'].get('status')
            if db_status == 'unhealthy':
                return 'critical'
        
        if 'cache' in health_status['components']:
            cache_status = health_status['components']['cache'].get('status')
            if cache_status == 'unhealthy':
                return 'degraded'
        
        # Check system resources
        if 'system_metrics' in health_status:
            metrics = health_status['system_metrics']
            thresholds = self.config['ALERT_THRESHOLDS']
            
            cpu_usage = metrics.get('cpu', {}).get('usage_percent', 0)
            memory_usage = metrics.get('memory', {}).get('usage_percent', 0)
            disk_usage = metrics.get('disk', {}).get('usage_percent', 0)
            
            if (cpu_usage > thresholds['cpu_usage'] or 
                memory_usage > thresholds['memory_usage'] or 
                disk_usage > thresholds['disk_usage']):
                return 'degraded'
        
        return 'healthy'
    
    def _generate_alerts(self, health_status: Dict) -> List[Dict]:
        """Generate alerts based on health status"""
        alerts = []
        thresholds = self.config['ALERT_THRESHOLDS']
        
        # System resource alerts
        if 'system_metrics' in health_status:
            metrics = health_status['system_metrics']
            
            cpu_usage = metrics.get('cpu', {}).get('usage_percent', 0)
            if cpu_usage > thresholds['cpu_usage']:
                alerts.append({
                    'type': 'resource',
                    'severity': 'warning',
                    'message': f'High CPU usage: {cpu_usage:.1f}%',
                    'threshold': thresholds['cpu_usage']
                })
            
            memory_usage = metrics.get('memory', {}).get('usage_percent', 0)
            if memory_usage > thresholds['memory_usage']:
                alerts.append({
                    'type': 'resource',
                    'severity': 'warning',
                    'message': f'High memory usage: {memory_usage:.1f}%',
                    'threshold': thresholds['memory_usage']
                })
            
            disk_usage = metrics.get('disk', {}).get('usage_percent', 0)
            if disk_usage > thresholds['disk_usage']:
                alerts.append({
                    'type': 'resource',
                    'severity': 'critical',
                    'message': f'High disk usage: {disk_usage:.1f}%',
                    'threshold': thresholds['disk_usage']
                })
        
        # Service alerts
        if 'components' in health_status:
            for component, status in health_status['components'].items():
                if isinstance(status, dict) and status.get('status') == 'unhealthy':
                    alerts.append({
                        'type': 'service',
                        'severity': 'critical',
                        'message': f'{component.title()} service is unhealthy',
                        'error': status.get('error', 'Unknown error')
                    })
        
        return alerts
    
    async def perform_maintenance_tasks(self):
        """Perform routine maintenance tasks"""
        self.logger.info("Starting maintenance tasks")
        
        maintenance_results = {
            'timestamp': datetime.now().isoformat(),
            'tasks': []
        }
        
        # Database maintenance
        db_result = await self._perform_database_maintenance()
        maintenance_results['tasks'].append(db_result)
        
        # Cache maintenance
        cache_result = await self._perform_cache_maintenance()
        maintenance_results['tasks'].append(cache_result)
        
        # Log cleanup
        log_result = await self._perform_log_cleanup()
        maintenance_results['tasks'].append(log_result)
        
        # System optimization
        system_result = await self._perform_system_optimization()
        maintenance_results['tasks'].append(system_result)
        
        self.logger.info("Maintenance tasks completed")
        return maintenance_results
    
    async def _perform_database_maintenance(self) -> Dict:
        """Perform database maintenance tasks"""
        try:
            self.logger.info("Starting database maintenance")
            
            async with self.db_manager.get_async_session() as session:
                # Analyze tables for query optimization
                await session.execute("ANALYZE;")
                
                # Vacuum to reclaim space
                await session.execute("VACUUM;")
                
                # Update table statistics
                await session.execute("UPDATE pg_stat_user_tables SET n_tup_ins = n_tup_ins;")
            
            return {
                'task': 'database_maintenance',
                'status': 'completed',
                'message': 'Database maintenance completed successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Database maintenance error: {str(e)}")
            return {
                'task': 'database_maintenance',
                'status': 'failed',
                'error': str(e)
            }
    
    async def _perform_cache_maintenance(self) -> Dict:
        """Perform cache maintenance tasks"""
        try:
            self.logger.info("Starting cache maintenance")
            
            # Get cache statistics before cleanup
            stats_before = await self.cache_manager.get_stats()
            
            # Clear expired keys
            await self.cache_manager.cleanup_expired()
            
            # Get statistics after cleanup
            stats_after = await self.cache_manager.get_stats()
            
            return {
                'task': 'cache_maintenance',
                'status': 'completed',
                'message': 'Cache maintenance completed successfully',
                'stats': {
                    'before': stats_before,
                    'after': stats_after
                }
            }
            
        except Exception as e:
            self.logger.error(f"Cache maintenance error: {str(e)}")
            return {
                'task': 'cache_maintenance',
                'status': 'failed',
                'error': str(e)
            }
    
    async def _perform_log_cleanup(self) -> Dict:
        """Clean up old log files"""
        try:
            self.logger.info("Starting log cleanup")
            
            log_dir = Path('./logs')
            if not log_dir.exists():
                return {
                    'task': 'log_cleanup',
                    'status': 'skipped',
                    'message': 'Log directory does not exist'
                }
            
            # Remove logs older than 30 days
            cutoff_date = datetime.now() - timedelta(days=30)
            deleted_count = 0
            
            for log_file in log_dir.glob('*.log*'):
                if log_file.is_file():
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        log_file.unlink()
                        deleted_count += 1
            
            return {
                'task': 'log_cleanup',
                'status': 'completed',
                'message': f'Deleted {deleted_count} old log files'
            }
            
        except Exception as e:
            self.logger.error(f"Log cleanup error: {str(e)}")
            return {
                'task': 'log_cleanup',
                'status': 'failed',
                'error': str(e)
            }
    
    async def _perform_system_optimization(self) -> Dict:
        """Perform system optimization tasks"""
        try:
            self.logger.info("Starting system optimization")
            
            # Clear Python cache
            import gc
            gc.collect()
            
            # Optimize memory usage
            optimizations = []
            
            # Check if we can free up memory
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                optimizations.append("High memory usage detected")
            
            return {
                'task': 'system_optimization',
                'status': 'completed',
                'message': 'System optimization completed',
                'optimizations': optimizations
            }
            
        except Exception as e:
            self.logger.error(f"System optimization error: {str(e)}")
            return {
                'task': 'system_optimization',
                'status': 'failed',
                'error': str(e)
            }
    
    def schedule_maintenance(self):
        """Schedule automated maintenance tasks"""
        if not self.config['MAINTENANCE_ENABLED']:
            self.logger.info("Maintenance is disabled")
            return
        
        # Schedule health checks every 5 minutes
        schedule.every(5).minutes.do(
            lambda: asyncio.run(self.check_system_health())
        )
        
        # Schedule maintenance tasks daily at 3 AM
        schedule.every().day.at("03:00").do(
            lambda: asyncio.run(self.perform_maintenance_tasks())
        )
        
        self.logger.info("Maintenance tasks scheduled")

async def main():
    """Main function to run maintenance service"""
    service = MaintenanceService()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--health-check':
            health_status = await service.check_system_health()
            print(json.dumps(health_status, indent=2))
            return
        elif sys.argv[1] == '--maintenance':
            maintenance_results = await service.perform_maintenance_tasks()
            print(json.dumps(maintenance_results, indent=2))
            return
    
    # Schedule and run continuous maintenance service
    service.schedule_maintenance()
    
    print("Maintenance service started. Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("Maintenance service stopped.")

if __name__ == "__main__":
    asyncio.run(main())
