import requests
import json
import time
import csv
from typing import Dict, List, Tuple
import openai
import os
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import asyncio
import aiohttp
from datetime import timedelta

# Load environment variables at the start of the file
load_dotenv()

# Configure OpenAI-compatible endpoint
openai.api_base = "http://x9dri.local:8080/v1"  # Adjust if your llama.cpp server is on a different port
openai.api_key = "dummy"  # llama.cpp doesn't need a real key

class RedditAPI:
    def __init__(self):
        # Verify required environment variables are present
        required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_SECRET', 'REDDIT_USER_AGENT']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                "Please check your .env file."
            )
            
        self.access_token = self.get_access_token()
        
    def get_access_token(self):
        """Get OAuth access token from Reddit"""
        data = {'grant_type': 'client_credentials'}
        auth = HTTPBasicAuth(
            os.environ.get('REDDIT_CLIENT_ID'),
            os.environ.get('REDDIT_SECRET')
        )
        headers = {'User-Agent': os.environ.get('REDDIT_USER_AGENT', 'SubredditCategorizer/1.0')}
        response = requests.post(
            'https://www.reddit.com/api/v1/access_token',
            auth=auth,
            data=data,
            headers=headers
        )
        return response.json().get('access_token')

    def make_request(self, endpoint):
        """Make authenticated request to Reddit API"""
        headers = {
            'Authorization': f'bearer {self.access_token}',
            'User-Agent': os.environ.get('REDDIT_USER_AGENT', 'SubredditCategorizer/1.0')
        }
        response = requests.get(f'https://oauth.reddit.com{endpoint}', headers=headers)
        return response.json()

    async def check_rate_limit(self, rate_limits: dict):
        """Check and wait if we're approaching rate limits based on Reddit's headers"""
        if not rate_limits:
            return

        remaining = float(rate_limits['remaining'])
        reset_seconds = float(rate_limits['reset'])

        # If we're approaching the limit (less than 50 requests remaining), wait
        if remaining < 50:
            wait_time = reset_seconds
            print(f"\nApproaching Reddit's rate limit. Waiting {int(wait_time)} seconds...")
            await asyncio.sleep(wait_time)

    async def make_request_async(self, session: aiohttp.ClientSession, endpoint: str) -> Tuple[dict, dict]:
        """Make authenticated request to Reddit API asynchronously"""
        headers = {
            'Authorization': f'bearer {self.access_token}',
            'User-Agent': os.environ.get('REDDIT_USER_AGENT', 'SubredditCategorizer/1.0')
        }
        
        async with session.get(f'https://oauth.reddit.com{endpoint}', headers=headers) as response:
            # Extract rate limit headers
            rate_limits = {
                'remaining': response.headers.get('x-ratelimit-remaining', 'N/A'),
                'used': response.headers.get('x-ratelimit-used', 'N/A'),
                'reset': response.headers.get('x-ratelimit-reset', 'N/A')
            }
            
            # Check rate limits after each request
            await self.check_rate_limit(rate_limits)
            
            return await response.json(), rate_limits

def load_categories() -> Dict[str, List[str]]:
    """Load the category lists from subreddits.json"""
    with open('subreddits.json', 'r') as f:
        categories = json.load(f)
    return categories

async def get_reddit_data_async(reddit_api: RedditAPI, session: aiohttp.ClientSession, subreddit: str) -> Tuple[dict, dict, dict]:
    """Fetch subreddit data asynchronously"""
    try:
        # Get both requests concurrently
        top_task = reddit_api.make_request_async(session, f'/r/{subreddit}/top')
        about_task = reddit_api.make_request_async(session, f'/r/{subreddit}/about')
        (top_data, top_limits), (about_data, about_limits) = await asyncio.gather(top_task, about_task)
        
        return top_data, about_data, about_limits  # Return the last rate limit info
    
    except Exception as e:
        print(f"Error fetching data for r/{subreddit}: {e}")
        return None, None, None

def create_prompt(subreddit: str, top_data: dict, about_data: dict, categories: Dict[str, List[str]]) -> str:
    """Create a prompt for the LLM with more explicit instructions"""
    
    if about_data.get('data', {}).get('over_18', False):
        return None
    
    description = about_data.get('data', {}).get('public_description', '')
    title = about_data.get('data', {}).get('title', '')
    
    top_posts = []
    try:
        for post in top_data.get('data', {}).get('children', [])[:5]:
            top_posts.append(post['data']['title'])
    except:
        pass

    # Format existing subcategories as a numbered list
    existing_subcats = '\n'.join(f"{i+1}. {cat}" for i, cat in enumerate(categories['third_column']))

    prompt = f"""Analyze this subreddit and categorize it:

SUBREDDIT INFO:
Name: r/{subreddit}
Title: {title}
Description: {description}

TOP POSTS:
{chr(10).join('- ' + post for post in top_posts)}

TASK:
1. Choose ONE main category from this list:
{', '.join(categories['second_column'])}

2. For the subcategory, FIRST check if any of these existing subcategories fit:
{existing_subcats}

If none of the existing subcategories fit well, suggest a NEW one that's concise and follows the same style.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
MAIN_CATEGORY: <category>
SUB_CATEGORY: <existing_number or NEW:suggested_category>"""

    return prompt

def update_subcategories(new_subcategory: str, categories: Dict[str, List[str]]) -> None:
    """Update subreddits.json with new subcategory if it's unique"""
    if new_subcategory not in categories['third_column']:
        categories['third_column'].append(new_subcategory)
        with open('subreddits.json', 'w', encoding='utf-8') as f:
            json.dump(categories, f, indent=2)
        print(f"Added new subcategory: {new_subcategory}")

def get_llm_categorization(prompt: str, about_data: dict, categories: Dict[str, List[str]]) -> Tuple[str, str]:
    """Get categorization from LLM and handle subcategory updates"""
    if about_data.get('data', {}).get('over_18', False):
        return "Adult and NSFW", ""
        
    try:
        formatted_prompt = f"""<|system|>
You are a helpful assistant that categorizes subreddits accurately and concisely.
<|end|>
<|user|>
{prompt}
<|end|>
<|assistant|>"""

        # Updated to use new OpenAI API format
        client = openai.OpenAI(
            base_url="http://x9dri.local:8080/v1",
            api_key="dummy"
        )
        
        response = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0.1,  # Lower temperature for more consistent categorization
            max_tokens=100,
            stop=["<|end|>"]  # Add explicit stop token
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse the response
        main_category = ""
        sub_category = ""
        
        for line in result.split('\n'):
            if line.startswith('MAIN_CATEGORY:'):
                main_category = line.split(':')[1].strip()
            elif line.startswith('SUB_CATEGORY:'):
                sub_value = line.split(':')[1].strip()
                
                # Handle numbered existing category
                if sub_value.split('.')[0].isdigit():
                    index = int(sub_value.split('.')[0]) - 1
                    if 0 <= index < len(categories['third_column']):
                        sub_category = categories['third_column'][index]
                # Handle new category suggestion
                elif sub_value.startswith('NEW:'):
                    new_sub = sub_value.replace('NEW:', '').strip()
                    update_subcategories(new_sub, categories)
                    sub_category = new_sub
        
        return main_category, sub_category
    
    except Exception as e:
        print(f"Error getting LLM categorization: {e}")
        return None, None

async def process_batch(reddit_api: RedditAPI, batch: List[str], categories: dict) -> dict:
    """Process a batch of subreddits concurrently"""
    last_rate_limits = None
    async with aiohttp.ClientSession() as session:
        # Fetch Reddit data for all subreddits in batch
        data_tasks = [get_reddit_data_async(reddit_api, session, subreddit) for subreddit in batch]
        results = await asyncio.gather(*data_tasks)
        
        # Process results with LLM
        for subreddit, (top_data, about_data, rate_limits) in zip(batch, results):
            if not top_data or not about_data:
                continue
            
            last_rate_limits = rate_limits
            print(f"Processing r/{subreddit}...")
            prompt = create_prompt(subreddit, top_data, about_data, categories)
            main_category, sub_category = get_llm_categorization(prompt, about_data, categories)
            
            if main_category and sub_category:
                # Modified file opening to handle Unicode
                with open('categorised.csv', 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([subreddit, main_category, sub_category] + [''] * 5)
                print(f"Categorized r/{subreddit} as {main_category} / {sub_category}")
        
        return last_rate_limits

async def main_async():
    start_time = time.time()
    processed_count = 0
    error_count = 0
    
    # Load categories
    categories = load_categories()
    
    # Initialize Reddit API
    reddit_api = RedditAPI()
    
    # Read subreddits to categorize
    with open('to_categorise.txt', 'r') as f_in:
        subreddits = [line.strip() for line in f_in if line.strip()]
    
    total_subs = len(subreddits)
    print(f"\nStarting processing of {total_subs} subreddits...")
    
    # Process subreddits in batches
    batch_size = 8
    for i in range(0, len(subreddits), batch_size):
        batch = subreddits[i:i + batch_size]
        rate_limits = await process_batch(reddit_api, batch, categories)
        processed_count += len(batch)
        
        # Calculate and display progress
        elapsed_time = time.time() - start_time
        avg_time_per_sub = elapsed_time / processed_count
        remaining_subs = total_subs - processed_count
        estimated_remaining = remaining_subs * avg_time_per_sub
        
        print(f"\nProgress Update:")
        print(f"Processed: {processed_count}/{total_subs} subreddits")
        print(f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}")
        print(f"Average Time per Subreddit: {avg_time_per_sub:.2f} seconds")
        print(f"Estimated Time Remaining: {timedelta(seconds=int(estimated_remaining))}")
        if rate_limits:
            print("\nReddit API Status:")
            print(f"Official API Remaining: {rate_limits['remaining']}")
            print(f"Official API Used: {rate_limits['used']}")
            print(f"Official Reset in: {rate_limits['reset']} seconds")
        print("-" * 50)
        
        await asyncio.sleep(0.2)  # Rate limiting between batches

def main():
    """Entry point that runs the async main function"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 