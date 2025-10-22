# forge/data_processing/sentiment_engine.py
import asyncio
import aiohttp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import os

class SentimentEngine:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.sentiment_score = 0.5  # Neutral sentiment
        self.sentiment_momentum = 0

    async def fetch_and_analyze(self):
        """
        Fetches news headlines from CryptoPanic and analyzes their sentiment.
        """
        api_key = os.getenv('CRYPTOPANIC_API_KEY')
        if not api_key:
            print("[Sentiment] WARNING: 'CRYPTOPANIC_API_KEY' not found in .env file. Sentiment analysis is disabled.")
            return

        print("[Sentiment] Fetching and analyzing sentiment...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&public=true") as response:
                    if response.status == 200:
                        data = await response.json()
                        headlines = [post['title'] for post in data['results']]
                        
                        if headlines:
                            sentiment_scores = [self.analyzer.polarity_scores(h)['compound'] for h in headlines]
                            new_sentiment_score = pd.Series(sentiment_scores).mean()
                            
                            # Update momentum
                            self.sentiment_momentum = (new_sentiment_score - self.sentiment_score)
                            self.sentiment_score = new_sentiment_score
                            
                            print(f"[Sentiment] Sentiment updated: {self.sentiment_score:.2f} (Momentum: {self.sentiment_momentum:.2f})")
                        else:
                            print("[Sentiment] No headlines found.")
                    else:
                        print(f"[Sentiment] Error fetching sentiment data: {response.status}")
        except Exception as e:
            print(f"[Sentiment] An error occurred during sentiment analysis: {e}")

    def get_sentiment(self):
        """Returns the current sentiment score and momentum."""
        return self.sentiment_score, self.sentiment_momentum