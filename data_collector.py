"""
Real-time market data collector with error handling and rate limiting.
Uses CCXT for exchange connectivity with Firebase state persistence.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import ccxt
from firebase_admin import firestore

from config import CONFIG

logger = logging.getLogger(__name__)

class MarketDataCollector:
    """Collects and manages market data with robust error handling"""
    
    def __init__(self, exchange_name: str = None):
        """Initialize data collector with exchange connection"""
        self.exchange_name = exchange_name or CONFIG.EXCHANGE_NAME
        self.symbol = CONFIG.SYMBOL
        self.timeframe = CONFIG.TIMEFRAME
        self.exchange = None
        self.db = None
        self._initialize_exchange()
        self._initialize_firebase()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
        
    def _initialize_exchange(self) -> None:
        """Initialize exchange connection with error handling"""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            
            # Test connection
            self.exchange.fetch_status()
            logger.info(f"Connected to {self.exchange_name} exchange")
            
        except AttributeError:
            logger.error(f"Exchange {self.exchange_name} not found in CCXT")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {str(e)}")
            raise
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection for data persistence"""
        try:
            import firebase_admin
            from firebase_admin import credentials
            
            if CONFIG.FIREBASE_CREDENTIALS_PATH:
                cred = credentials.Certificate(CONFIG.FIREBASE_CREDENTIALS_PATH)
                firebase_admin.initialize_app(cred)
            else:
                # Use default credentials if available
                firebase_admin.initialize_app()
                
            self.db = firestore.client()
            logger.info("Firebase Firestore initialized")
            
        except Exception as e:
            logger.warning(f"Firebase initialization failed: {str(e)}")
            logger.info("Continuing without Firebase persistence")
            self.db = None
    
    def _respect_rate_limit(self) -> None:
        """Enforce rate limiting to avoid API bans"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_historical_data(self, days: int = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with error handling and retries
        """
        days = days or CONFIG.HISTORICAL_DAYS