#!/usr/bin/env python3
"""
Backup Service for Supply Chain Risk Intelligence System
Handles automated backups of database, Redis cache, and application data
"""

import os
import sys
import time
import shutil
import gzip
import json
import logging
import schedule
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
import aiofiles
import psutil

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.manager import DatabaseManager
from cache.manager import CacheManager

class BackupService:
    """Comprehensive backup service for the supply chain risk system"""
    
    def __init__(self):
        self.setup_logging()
        self.config = self.load_config()
        self.backup_dir = Path(self.config.get('BACKUP_DIRECTORY', './backups'))
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('./logs/backup.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> Dict:
        """Load configuration from environment variables"""
        return {
            'DATABASE_URL': os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@postgres:5432/supply_chain_risk'),
            'REDIS_URL': os.getenv('REDIS_URL', 'redis://redis:6379'),
            'BACKUP_DIRECTORY': os.getenv('BACKUP_DIRECTORY', './backups'),
            'BACKUP_RETENTION_DAYS': int(os.getenv('BACKUP_RETENTION_DAYS', '30')),
            'BACKUP_COMPRESSION': os.getenv('BACKUP_COMPRESSION', 'true').lower() == 'true',
            'BACKUP_SCHEDULE': os.getenv('BACKUP_SCHEDULE', '0 2 * * *'),
            'BACKUP_ENABLED': os.getenv('BACKUP_ENABLED', 'true').lower() == 'true'
        }
    
    async def backup_database(self) -> Optional[str]:
        """Backup PostgreSQL database"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"database_backup_{timestamp}.sql"
            
            self.logger.info(f"Starting database backup to {backup_file}")
            
            # Extract database connection details
            db_url = self.config['DATABASE_URL']
            # Parse connection string
            if db_url.startswith('postgresql://'):
                # Simple parsing for demo - in production use proper URL parsing
                parts = db_url.replace('postgresql://', '').split('@')
                auth_part = parts[0]
                host_part = parts[1]
                
                user, password = auth_part.split(':')
                host, db_name = host_part.split('/')
                host = host.split(':')[0]  # Remove port if present
                
                # Use pg_dump for backup
                cmd = [
                    'pg_dump',
                    '-h', host,
                    '-U', user,
                    '-d', db_name,
                    '--no-password',
                    '-f', str(backup_file)
                ]
                
                env = os.environ.copy()
                env['PGPASSWORD'] = password
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    # Compress if enabled
                    if self.config['BACKUP_COMPRESSION']:
                        compressed_file = f"{backup_file}.gz"
                        with open(backup_file, 'rb') as f_in:
                            with gzip.open(compressed_file, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        os.remove(backup_file)
                        backup_file = compressed_file
                    
                    self.logger.info(f"Database backup completed: {backup_file}")
                    return str(backup_file)
                else:
                    self.logger.error(f"Database backup failed: {stderr.decode()}")
                    return None
            
        except Exception as e:
            self.logger.error(f"Database backup error: {str(e)}")
            return None
    
    async def backup_redis(self) -> Optional[str]:
        """Backup Redis data"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"redis_backup_{timestamp}.json"
            
            self.logger.info(f"Starting Redis backup to {backup_file}")
            
            # Get all keys and their data
            redis_data = {}
            
            # Use cache manager to get all cached data
            all_keys = await self.cache_manager._get_all_keys()
            
            for key in all_keys:
                try:
                    value = await self.cache_manager.get(key)
                    if value is not None:
                        redis_data[key] = value
                except Exception as e:
                    self.logger.warning(f"Failed to backup key {key}: {str(e)}")
            
            # Save to JSON file
            async with aiofiles.open(backup_file, 'w') as f:
                await f.write(json.dumps(redis_data, indent=2, default=str))
            
            # Compress if enabled
            if self.config['BACKUP_COMPRESSION']:
                compressed_file = f"{backup_file}.gz"
                with open(backup_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(backup_file)
                backup_file = compressed_file
            
            self.logger.info(f"Redis backup completed: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Redis backup error: {str(e)}")
            return None
    
    async def backup_application_data(self) -> Optional[str]:
        """Backup application data, models, and configurations"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_dir / f"app_data_backup_{timestamp}.tar.gz"
            
            self.logger.info(f"Starting application data backup to {backup_file}")
            
            # Directories to backup
            data_dirs = [
                './data',
                './checkpoints',
                './config',
                './logs'
            ]
            
            # Create tar archive
            cmd = ['tar', '-czf', str(backup_file)]
            for data_dir in data_dirs:
                if os.path.exists(data_dir):
                    cmd.append(data_dir)
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.logger.info(f"Application data backup completed: {backup_file}")
                return str(backup_file)
            else:
                self.logger.error(f"Application data backup failed: {stderr.decode()}")
                return None
                
        except Exception as e:
            self.logger.error(f"Application data backup error: {str(e)}")
            return None
    
    async def cleanup_old_backups(self):
        """Remove backup files older than retention period"""
        try:
            retention_days = self.config['BACKUP_RETENTION_DAYS']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            self.logger.info(f"Cleaning up backups older than {retention_days} days")
            
            deleted_count = 0
            for backup_file in self.backup_dir.glob('*'):
                if backup_file.is_file():
                    file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        backup_file.unlink()
                        deleted_count += 1
                        self.logger.info(f"Deleted old backup: {backup_file}")
            
            self.logger.info(f"Cleanup completed: {deleted_count} files deleted")
            
        except Exception as e:
            self.logger.error(f"Backup cleanup error: {str(e)}")
    
    async def create_backup_manifest(self, backup_files: List[str]) -> str:
        """Create a manifest file with backup information"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            manifest_file = self.backup_dir / f"backup_manifest_{timestamp}.json"
            
            # Get system information
            system_info = {
                'timestamp': datetime.now().isoformat(),
                'hostname': os.uname().nodename,
                'python_version': sys.version,
                'disk_usage': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free
                },
                'memory_usage': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'used': psutil.virtual_memory().used
                }
            }
            
            # Create manifest
            manifest = {
                'backup_info': {
                    'created_at': datetime.now().isoformat(),
                    'backup_version': '1.0',
                    'system_info': system_info
                },
                'backup_files': []
            }
            
            # Add file information
            for backup_file in backup_files:
                if backup_file and os.path.exists(backup_file):
                    file_stat = os.stat(backup_file)
                    manifest['backup_files'].append({
                        'filename': os.path.basename(backup_file),
                        'full_path': backup_file,
                        'size_bytes': file_stat.st_size,
                        'created_at': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        'type': self._get_backup_type(backup_file)
                    })
            
            # Save manifest
            async with aiofiles.open(manifest_file, 'w') as f:
                await f.write(json.dumps(manifest, indent=2))
            
            self.logger.info(f"Backup manifest created: {manifest_file}")
            return str(manifest_file)
            
        except Exception as e:
            self.logger.error(f"Manifest creation error: {str(e)}")
            return ""
    
    def _get_backup_type(self, filename: str) -> str:
        """Determine backup type from filename"""
        if 'database' in filename:
            return 'database'
        elif 'redis' in filename:
            return 'cache'
        elif 'app_data' in filename:
            return 'application_data'
        else:
            return 'unknown'
    
    async def perform_full_backup(self):
        """Perform complete system backup"""
        self.logger.info("Starting full system backup")
        start_time = time.time()
        
        backup_files = []
        
        # Backup database
        db_backup = await self.backup_database()
        if db_backup:
            backup_files.append(db_backup)
        
        # Backup Redis
        redis_backup = await self.backup_redis()
        if redis_backup:
            backup_files.append(redis_backup)
        
        # Backup application data
        app_backup = await self.backup_application_data()
        if app_backup:
            backup_files.append(app_backup)
        
        # Create manifest
        manifest_file = await self.create_backup_manifest(backup_files)
        if manifest_file:
            backup_files.append(manifest_file)
        
        # Cleanup old backups
        await self.cleanup_old_backups()
        
        duration = time.time() - start_time
        self.logger.info(f"Full backup completed in {duration:.2f} seconds")
        self.logger.info(f"Backup files created: {len(backup_files)}")
        
        return backup_files
    
    def schedule_backups(self):
        """Schedule automated backups"""
        if not self.config['BACKUP_ENABLED']:
            self.logger.info("Backups are disabled")
            return
        
        # Parse cron schedule (simplified - supports daily at specific time)
        schedule_time = "02:00"  # Default 2 AM
        
        self.logger.info(f"Scheduling daily backup at {schedule_time}")
        schedule.every().day.at(schedule_time).do(
            lambda: asyncio.run(self.perform_full_backup())
        )
        
        # Also schedule weekly cleanup
        schedule.every().sunday.at("03:00").do(
            lambda: asyncio.run(self.cleanup_old_backups())
        )

async def main():
    """Main function to run backup service"""
    backup_service = BackupService()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--backup-now':
            await backup_service.perform_full_backup()
            return
        elif sys.argv[1] == '--cleanup':
            await backup_service.cleanup_old_backups()
            return
    
    # Schedule and run continuous backup service
    backup_service.schedule_backups()
    
    print("Backup service started. Press Ctrl+C to stop.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("Backup service stopped.")

if __name__ == "__main__":
    asyncio.run(main())
