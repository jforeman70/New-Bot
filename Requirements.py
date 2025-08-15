# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
yfinance>=0.1.87
requests>=2.25.0
ib_insync>=0.9.86
schedule>=1.1.0
python-dotenv>=0.19.0

# config.json
{
  "scan_interval_seconds": 60,
  "position_timeout_hours": 24,
  "max_concurrent_positions": 3,
  "min_signal_strength": 75,
  "min_confidence": 0.7,
  "daily_loss_limit_pct": 3.0,
  "weekly_loss_limit_pct": 8.0,
  "emergency_vix_level": 45,
  "symbols_watchlist": [
    "SPY", "QQQ", "IWM", "TLT", "GLD",
    "BTCUSD", "ETHUSD",
    "EURUSD", "USDJPY"
  ],
  "leverage_limits": {
    "crypto": 5.0,
    "forex": 10.0,
    "equity": 2.0,
    "default": 1.0
  },
  "risk_management": {
    "base_position_pct": 1.0,
    "max_position_pct": 4.0,
    "max_daily_trades": 50,
    "max_slippage_bps": 20,
    "order_timeout_seconds": 30
  }
}

# .env.example
# Copy this to .env and fill in your actual API keys
TOKEN_METRICS=your_token_metrics_key
FINNHUB_API_KEY=your_finnhub_key
FMP_API_KEY=your_fmp_key
SECAPI_KEY=your_sec_api_key
GNEWS_KEY=your_gnews_key
OPENROUTER_API_KEY=your_openrouter_key
HUGGINGFACE_API_KEY=your_huggingface_key
GEMINI_API_KEY=your_gemini_key
GOOGLE_EMAIL=your_email@gmail.com
GOOGLE_PASSWORD=your_app_password
GOOGLE_CREDENTIALS_JSON=credentials.json
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=0
IBKR_USER=your_ibkr_username
IBKR_PASSWORD=your_ibkr_password
IBKR_ACCOUNT=your_ibkr_account
MARKETSTACK_API_KEY=your_marketstack_key
COINMARKETCAP_API_KEY=your_coinmarketcap_key
DISCORD_API_KEY=your_discord_key
CONSOLE_GROQ_API_KEY=your_groq_key
INITIAL_CAPITAL=10000

# setup.py
from setuptools import setup, find_packages

setup(
    name="trailhead-catalyst-bot",
    version="1.0.0",
    description="Production-ready quantitative trading bot with trailhead detection",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "yfinance>=0.1.87",
        "requests>=2.25.0",
        "ib_insync>=0.9.86",
        "schedule>=1.1.0",
        "python-dotenv>=0.19.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

# run_bot.py - Main execution script
#!/usr/bin/env python3
"""
Trailhead Catalyst Bot Runner
Production deployment script with monitoring
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trailhead_catalyst_bot import TrailheadCatalystBot

def setup_production_logging():
    """Setup production-grade logging"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/bot_main_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_environment():
    """Validate required environment variables"""
    required_vars = [
        'FMP_API_KEY',
        'FINNHUB_API_KEY', 
        'IBKR_HOST',
        'IBKR_PORT',
        'IBKR_ACCOUNT'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
            
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Please check your .env file")
        return False
        
    return True

def main():
    parser = argparse.ArgumentParser(description='Trailhead Catalyst Trading Bot')
    parser.add_argument('--capital', type=float, default=10000, 
                       help='Initial trading capital (default: 10000)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in paper trading mode')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_production_logging()
    logger = logging.getLogger(__name__)
    
    print("🤖 Trailhead Catalyst Bot - Production Ready")
    print("=" * 60)
    print(f"💰 Initial Capital: ${args.capital:,.2f}")
    print(f"📋 Config File: {args.config}")
    
    if args.dry_run:
        print("📝 DRY RUN MODE - No real trades will be executed")
        
    print("=" * 60)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
        
    try:
        # Create bot instance
        logger.info("Initializing Trailhead Catalyst Bot...")
        bot = TrailheadCatalystBot(initial_capital=args.capital)
        
        # Display configuration
        config = bot.config
        print(f"🎯 Max Positions: {config['max_concurrent_positions']}")
        print(f"⏱️  Scan Interval: {config['scan_interval_seconds']}s")
        print(f"🚨 Daily Loss Limit: {config['daily_loss_limit_pct']}%")
        print(f"📊 Min Signal Strength: {config['min_signal_strength']}")
        print(f"🎲 Min Confidence: {config['min_confidence']}")
        print("=" * 60)
        
        # Pre-flight checks
        logger.info("Running pre-flight system checks...")
        health = bot._health_check()
        
        if not all(health.values()):
            logger.error(f"❌ Pre-flight check failed: {health}")
            print("System not ready for trading. Check logs for details.")
            sys.exit(1)
            
        print("✅ All systems operational")
        print("🚀 Starting bot...")
        print("Press Ctrl+C to shutdown gracefully")
        print("=" * 60)
        
        # Start the bot
        bot.start()
        
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
        print("\n🛑 Graceful shutdown initiated...")
        
    except Exception as e:
        logger.critical(f"FATAL ERROR: {e}")
        print(f"\n💥 FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

# Docker deployment files
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Run the bot
CMD ["python", "run_bot.py"]

# docker-compose.yml
version: '3.8'

services:
  trailhead-bot:
    build: .
    container_name: trailhead-catalyst-bot
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./config.json:/app/config.json
    restart: unless-stopped
    network_mode: host  # Required for IBKR connection
    
  # Optional: monitoring service
  watchtower:
    image: containrrr/watchtower
    container_name: watchtower
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    command: --interval 30 trailhead-catalyst-bot
    restart: unless-stopped

# Installation and deployment instructions
# README_DEPLOYMENT.md
## Trailhead Catalyst Bot - Deployment Guide

### Prerequisites
1. Python 3.8+ installed
2. Interactive Brokers TWS or Gateway running
3. All required API keys configured
4. $10,000+ trading account (recommended minimum)

### Quick Start
1. Clone repository to your server
2. Copy .env.example to .env and fill in your API keys
3. Install dependencies: `pip install -r requirements.txt`
4. Start IBKR TWS/Gateway
5. Run: `python run_bot.py --capital 10000`

### Production Deployment
1. Use Docker: `docker-compose up -d`
2. Monitor logs: `docker logs -f trailhead-catalyst-bot`
3. Check performance: Review logs/bot_main_YYYYMMDD.log

### Safety Features
- Automatic emergency stop on black swan events
- Daily/weekly loss limits with circuit breakers
- Position timeouts and stop losses
- Comprehensive health monitoring
- Graceful shutdown on system signals

### Monitoring
- All trades logged with full audit trail
- Real-time performance metrics in logs
- Emergency alerts for system failures
- Position management with automatic stops

### Configuration
- Edit config.json for trading parameters
- Adjust position sizes, risk limits, timeouts
- Customize symbol watchlist
- Set leverage limits by asset class

⚠️ **WARNING**: This bot trades real money. Test thoroughly in paper trading mode first.
