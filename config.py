"""
Configuration module for the autonomous trading system.
Centralizes all configurable parameters with validation.
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class TradingConfig:
    """Trading system configuration with validation"""
    # Exchange configuration
    EXCHANGE_NAME: str = "binance"
    SYMBOL: str = "BTC/USDT"
    TIMEFRAME: str = "1h"
    INITIAL_BALANCE: float = 10000.0
    
    # Data collection
    HISTORICAL_DAYS: int = 365
    DATA_WINDOW_SIZE: int = 100
    
    # RL Agent parameters
    STATE_DIM: int = 15  # Number of features per state
    ACTION_SPACE: int = 3  [0: hold, 1: buy, 2: sell]
    GAMMA: float = 0.99
    EPSILON_START: float = 1.0
    EPSILON_END: float = 0.01
    EPSILON_DECAY: float = 0.995
    BATCH_SIZE: int = 64
    MEMORY_CAPACITY: int = 10000
    TARGET_UPDATE_FREQ: int = 100
    
    # Training parameters
    EPISODES: int = 1000
    MAX_STEPS_PER_EPISODE: int = 1000
    LEARNING_RATE: float = 0.001
    
    # Risk management
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    STOP_LOSS_PERCENT: float = 0.02  # 2%
    TAKE_PROFIT_PERCENT: float = 0.05  # 5%
    
    # Firebase configuration
    FIREBASE_CREDENTIALS_PATH: Optional[str] = os.getenv("FIREBASE_CREDENTIALS_PATH")
    FIREBASE_PROJECT_ID: Optional[str] = os.getenv("FIREBASE_PROJECT_ID")
    
    # Telegram alerts
    TELEGRAM_BOT_TOKEN: Optional[str] = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID: Optional[str] = os.getenv("TELEGRAM_CHAT_ID")
    
    def validate(self) -> bool:
        """Validate all configuration parameters"""
        validations = [
            (self.INITIAL_BALANCE > 0, "Initial balance must be positive"),
            (0 < self.MAX_POSITION_SIZE <= 1, "Max position size must be between 0 and 1"),
            (self.STOP_LOSS_PERCENT > 0, "Stop loss must be positive"),
            (self.TAKE_PROFIT_PERCENT > self.STOP_LOSS_PERCENT, 
             "Take profit must exceed stop loss"),
            (0 <= self.EPSILON_END < self.EPSILON_START <= 1, 
             "Epsilon values invalid"),
            (self.BATCH_SIZE <= self.MEMORY_CAPACITY, 
             "Batch size cannot exceed memory capacity"),
        ]
        
        for condition, error_msg in validations:
            if not condition:
                raise ValueError(f"Config validation failed: {error_msg}")
        
        return True

# Global configuration instance
CONFIG = TradingConfig()