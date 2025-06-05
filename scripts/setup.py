#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_sample_data():
    logger.info("Downloading sample satellite imagery and economic data")
    
    data_dirs = [
        "data/cache/satellite",
        "data/cache/social", 
        "data/cache/weather",
        "data/cache/economic",
        "data/cache/news"
    ]
    
    base_path = Path(__file__).parent.parent
    for data_dir in data_dirs:
        full_path = base_path / data_dir
        full_path.mkdir(parents=True, exist_ok=True)
        
        sample_file = full_path / "sample_data.txt"
        with open(sample_file, 'w') as f:
            f.write(f"Sample data for {data_dir}\n")
            f.write("This would contain actual data in production\n")
    
    logger.info("Sample data structure created")

def setup_environment():
    logger.info("Setting up development environment")
    
    env_file = Path(__file__).parent.parent / ".env"
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("# API Keys for data sources\n")
            f.write("TWITTER_API_KEY=your_twitter_api_key_here\n")
            f.write("REDDIT_CLIENT_ID=your_reddit_client_id_here\n") 
            f.write("WEATHER_API_KEY=your_weather_api_key_here\n")
            f.write("NEWS_API_KEY=your_news_api_key_here\n")
        
        logger.info("Created .env file template")
    
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    logger.info("Environment setup complete")

def main():
    parser = argparse.ArgumentParser(description="Setup script for Supply Chain Risk Intelligence")
    parser.add_argument('--download-data', action='store_true', help='Download sample data')
    parser.add_argument('--setup-env', action='store_true', help='Setup development environment')
    parser.add_argument('--all', action='store_true', help='Run all setup tasks')
    
    args = parser.parse_args()
    
    if args.all or args.setup_env:
        setup_environment()
        
    if args.all or args.download_data:
        download_sample_data()
    
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()
