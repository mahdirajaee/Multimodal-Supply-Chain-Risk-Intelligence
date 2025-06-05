#!/usr/bin/env python3

import sys
import subprocess
import time
import signal
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_api_server():
    logger.info("Starting API server...")
    try:
        result = subprocess.run([
            sys.executable, "main.py", "mode=api"
        ], cwd=Path(__file__).parent.parent, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"API server failed: {e}")
    except KeyboardInterrupt:
        logger.info("API server stopped")

def run_dashboard():
    logger.info("Starting Streamlit dashboard...")
    time.sleep(5)
    try:
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", "scripts/dashboard.py",
            "--server.port", "8501", "--server.address", "0.0.0.0"
        ], cwd=Path(__file__).parent.parent, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Dashboard failed: {e}")
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")

def run_training():
    logger.info("Starting model training...")
    try:
        result = subprocess.run([
            sys.executable, "main.py", "mode=train"
        ], cwd=Path(__file__).parent.parent, check=True)
        logger.info("Training completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")

def run_tests():
    logger.info("Running test suite...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v"
        ], cwd=Path(__file__).parent.parent, check=True)
        logger.info("All tests passed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Tests failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Supply Chain Risk Intelligence Runner")
    parser.add_argument('--mode', choices=['train', 'api', 'dashboard', 'full', 'test'], 
                       default='full', help='Run mode')
    parser.add_argument('--no-dashboard', action='store_true', help='Skip dashboard in full mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        run_training()
    elif args.mode == 'api':
        run_api_server()
    elif args.mode == 'dashboard':
        run_dashboard()
    elif args.mode == 'test':
        run_tests()
    elif args.mode == 'full':
        logger.info("Starting full system...")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            futures.append(executor.submit(run_api_server))
            
            if not args.no_dashboard:
                futures.append(executor.submit(run_dashboard))
            
            try:
                for future in futures:
                    future.result()
            except KeyboardInterrupt:
                logger.info("Shutting down system...")
                for future in futures:
                    future.cancel()

if __name__ == "__main__":
    main()
