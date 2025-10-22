import io
import time
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import config
import investpy

import asyncio
import aiohttp
from datetime import datetime, timedelta
import config

class EconomicCalendar:
    """
    Manages fetching high-impact economic events asynchronously and provides
    time-to-event data for risk management.
    """
    def __init__(self):
        self.events_df = pd.DataFrame()
        self.last_fetch_date = None
        self.lock = asyncio.Lock()

    async def _fetch_events_async(self):
        """
        Asynchronously fetches high-impact economic events for the current day.
        Caches the result to avoid redundant calls.
        """
        async with self.lock:
            today = datetime.utcnow().date()
            if self.last_fetch_date == today and not self.events_df.empty:
                return

            print("[Calendar] Fetching high-impact economic events...")
            try:
                loop = asyncio.get_running_loop()
                # investpy is synchronous, so we run it in a thread pool
                events = await loop.run_in_executor(
                    None, 
                    lambda: investpy.economic_calendar(
                        importance_filter=['high'],
                        time_filter='this_week' # Fetch for the week to be safe
                    )
                )
                
                if not events.empty:
                    # Convert time and date to a single UTC datetime column
                    events['datetime'] = pd.to_datetime(events['date'] + ' ' + events['time'], format='%d/%m/%Y %H:%M', errors='coerce')
                    # Assuming the fetched time is GMT, convert to UTC (often they are the same, but this is safer)
                    events['datetime'] = events['datetime'].dt.tz_localize('GMT').dt.tz_convert('UTC').dt.tz_localize(None)
                    
                    self.events_df = events[['datetime', 'event', 'currency']].copy()
                    self.last_fetch_date = today
                    print(f"[Calendar] Successfully fetched {len(self.events_df)} high-impact events.")
                else:
                    self.events_df = pd.DataFrame() # Ensure it's an empty frame on failure
                    
            except Exception as e:
                print(f"[Calendar] ERROR: Could not fetch economic events: {e}")
                self.events_df = pd.DataFrame() # Reset on error

    async def get_time_to_next_event(self, current_time_utc: datetime) -> float:
        """
        Calculates the time in minutes until the next high-impact event.

        Returns:
            float: Minutes until the next event, or float('inf') if no events are scheduled.
        """
        if not config.EVENT_GUARDRAILS_ACTIVE:
            return float('inf')
        
        await self._fetch_events_async()
        if self.events_df.empty:
            return float('inf')

        imminent_events = self.events_df[self.events_df['datetime'] > current_time_utc]
        if imminent_events.empty:
            return float('inf')

        next_event_time = imminent_events['datetime'].min()
        time_to_event_minutes = (next_event_time - current_time_utc).total_seconds() / 60
        return time_to_event_minutes

    async def is_blackout_period(self, current_time_utc: datetime) -> bool:
        """Checks if the current time is within the blackout window of any event."""
        # This method now checks both past and future events within the window
        if not config.EVENT_GUARDRAILS_ACTIVE: return False
        await self._fetch_events_async()
        if self.events_df.empty: return False

        for event_time in self.events_df['datetime']:
            minutes_diff = abs((current_time_utc - event_time).total_seconds() / 60)
            if minutes_diff < config.EVENT_BLACKOUT_WINDOW:
                return True
        return False

    async def should_force_close(self, current_time_utc: datetime) -> bool:
        """Checks if positions should be closed ahead of an event."""
        time_to_event = await self.get_time_to_next_event(current_time_utc)
        return time_to_event <= config.EVENT_FORCE_CLOSE_WINDOW

def add_bot_log(message, bot_status):
    log_line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}"
    print(log_line)
    if 'logs' in bot_status:
        bot_status["logs"].insert(0, log_line)
        if len(bot_status["logs"]) > 200: bot_status["logs"].pop()
    try:
        with open(config.BOT_LOG_FILE, 'a') as f: f.write(log_line + '\n')
    except Exception as e: print(f"Error writing to log file: {e}")
