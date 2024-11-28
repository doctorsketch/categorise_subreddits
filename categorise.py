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
    subscribers = about_data.get('data', {}).get('subscribers', 'N/A')
    
    top_posts = []
    try:
        for post in top_data.get('data', {}).get('children', [])[:5]:
            top_posts.append(post['data']['title'])
    except:
        pass

    # Format existing subcategories as a numbered list with descriptions
    existing_subcats = '\n'.join(f"{i+1}. {cat} - Choose this for topics about {cat.lower()}" 
                                for i, cat in enumerate(categories['third_column']))

    prompt = f"""You are a subreddit categorization expert. Your task is to accurately categorize the subreddit based on its ACTUAL CONTENT, not superficial similarities.

IMPORTANT RULES:
1. Focus on the MAIN THEME of the content, not metaphorical similarities
2. Read the subreddit description and top posts carefully
3. If unsure, prefer broader categories over specific ones
4. Don't be misled by metaphorical or humorous references
5. Look for patterns in the actual discussion topics

Subreddit Details:
Name: r/{subreddit}
Title: {title}
Description: {description}
Active Users: {subscribers}

Recent Content Examples:
{chr(10).join('- ' + post for post in top_posts)}

Available Main Categories (pick ONE):
{', '.join(categories['second_column'])}

Available Subcategories (use number or suggest NEW):
{existing_subcats}

Respond ONLY with these three lines:
MAIN_CATEGORY: <select from available main categories>
SUB_CATEGORY: <use number from list OR write NEW:suggested_category>
SPECIALIST: <specific focus area or if not enough info available you can leave this blank>

Remember: Focus on ACTUAL content themes, not metaphorical similarities!"""

    return prompt

def update_subcategories(new_subcategory: str, categories: Dict[str, List[str]]) -> None:
    """Update subreddits.json with new subcategory if it's unique"""
    if new_subcategory not in categories['third_column']:
        categories['third_column'].append(new_subcategory)
        with open('subreddits.json', 'w', encoding='utf-8') as f:
            json.dump(categories, f, indent=2)
        print(f"Added new subcategory: {new_subcategory}")

async def process_llm_batch(prompts_data: List[Tuple[str, dict, dict, str]], categories: Dict[str, List[str]]) -> List[Tuple[str, str, str, str]]:
    """Process multiple LLM queries concurrently using a semaphore to control concurrency"""
    
    # Create a semaphore to limit concurrent LLM requests
    semaphore = asyncio.Semaphore(8)  # Allow 8 concurrent LLM requests
    
    async def process_single_prompt(prompt_data):
        async with semaphore:  # This ensures we don't exceed our concurrent request limit
            subreddit, prompt, about_data = prompt_data
            if not prompt:  # Handle None prompts (e.g., NSFW)
                return subreddit, "Adult and NSFW", "", ""
                
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Create client inside the function to ensure thread safety
                    client = openai.AsyncOpenAI(
                        base_url="http://x9dri.local:8080/v1",
                        api_key="dummy"
                    )
                    
                    formatted_prompt = f"""<|system|>
You are a categorization assistant. Respond ONLY with the three required lines. No explanations or additional text.
<|end|>
<|user|>
{prompt}
<|end|>
<|assistant|>"""
                    
                    # Use asyncio.create_task to run the API call asynchronously
                    response = await client.chat.completions.create(
                        model="local-model",
                        messages=[{"role": "user", "content": formatted_prompt}],
                        temperature=0.1,
                        max_tokens=100,
                        stop=["<|end|>"]
                    )
                    
                    # Clean up the response more strictly
                    result = response.choices[0].message.content.strip()
                    
                    # Only take lines that start with our expected prefixes and are properly formatted
                    valid_lines = []
                    for line in result.split('\n'):
                        line = line.strip()
                        if any(line.startswith(prefix) for prefix in ['MAIN_CATEGORY:', 'SUB_CATEGORY:', 'SPECIALIST:']):
                            # Verify the line has a colon and content after it
                            if ':' in line and line.split(':', 1)[1].strip():
                                valid_lines.append(line)
                    
                    if len(valid_lines) == 3:
                        main_category = ""
                        sub_category = ""
                        specialist = ""
                        
                        for line in valid_lines:
                            prefix, value = line.split(':', 1)
                            value = value.strip()
                            
                            if prefix == 'MAIN_CATEGORY':
                                main_category = value
                            elif prefix == 'SUB_CATEGORY':
                                if value.split('.')[0].isdigit():
                                    index = int(value.split('.')[0]) - 1
                                    if 0 <= index < len(categories['third_column']):
                                        sub_category = categories['third_column'][index]
                                elif value.startswith('NEW:'):
                                    new_sub = value.replace('NEW:', '').strip()
                                    update_subcategories(new_sub, categories)
                                    sub_category = new_sub
                                elif value.lower() in ['-', 'blank', 'none', 'n/a']:
                                    sub_category = ''
                                else:
                                    sub_category = value
                            elif prefix == 'SPECIALIST':
                                if value.lower() not in ['-', 'blank', 'none', 'n/a']:
                                    specialist = value
                        
                        # Clean up any remaining explanations or parentheticals
                        main_category = main_category.split('(')[0].strip()
                        sub_category = sub_category.split('(')[0].strip()
                        specialist = specialist.split('(')[0].strip()
                        
                        return subreddit, main_category, sub_category, specialist
                    
                    else:
                        print(f"Invalid response format for r/{subreddit} on attempt {attempt + 1}. Retrying...")
                
                except Exception as e:
                    print(f"Error processing r/{subreddit} on attempt {attempt + 1}: {e}")
            
            print(f"Failed to get valid response for r/{subreddit} after {max_retries} attempts.")
            return subreddit, None, None, None

    # Create tasks for all prompts simultaneously
    tasks = [asyncio.create_task(process_single_prompt(data)) for data in prompts_data]
    
    # Wait for all tasks to complete
    return await asyncio.gather(*tasks)

async def process_batch(reddit_api: RedditAPI, batch: List[str], categories: dict) -> dict:
    """Process a batch of subreddits concurrently"""
    last_rate_limits = None
    async with aiohttp.ClientSession() as session:
        # Fetch Reddit data for all subreddits in batch
        data_tasks = [get_reddit_data_async(reddit_api, session, subreddit) for subreddit in batch]
        results = await asyncio.gather(*data_tasks)
        
        # Prepare prompts for all valid results
        prompts_data = []
        for subreddit, (top_data, about_data, rate_limits) in zip(batch, results):
            if not top_data or not about_data:
                continue
                
            last_rate_limits = rate_limits
            print(f"Preparing prompt for r/{subreddit}...")
            prompt = create_prompt(subreddit, top_data, about_data, categories)
            prompts_data.append((subreddit, prompt, about_data))
        
        # Process all LLM queries truly concurrently
        llm_results = await process_llm_batch(prompts_data, categories)
        
        # Write results to CSV using a thread-safe approach
        async with aiohttp.ClientSession() as session:  # New session for file operations
            for subreddit, main_category, sub_category, specialist in llm_results:
                if main_category:
                    # Use asyncio.to_thread for file operations to prevent blocking
                    await asyncio.to_thread(write_to_csv, subreddit, main_category, sub_category, specialist)
        
        return last_rate_limits

def write_to_csv(subreddit: str, main_category: str, sub_category: str, specialist: str):
    """Thread-safe CSV writing function"""
    with open('categorised.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([subreddit, main_category, sub_category, specialist] + [''] * 4)
    print(f"Categorized r/{subreddit} as {main_category} / {sub_category} / {specialist}")

async def main_async():
    start_time = time.time()
    processed_count = 0
    
    # Load categories
    categories = load_categories()
    
    # Initialize Reddit API
    reddit_api = RedditAPI()
    
    # Read subreddits to categorize
    with open('to_categorise.txt', 'r') as f_in:
        subreddits = [line.strip() for line in f_in if line.strip()]
    
    total_subs = len(subreddits)
    print(f"\nStarting processing of {total_subs} subreddits...")
    
    # Increase batch size significantly
    batch_size = 1  # Increased from 8
    
    # Process subreddits in larger batches
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
        
        # Reduced sleep time between batches
        await asyncio.sleep(0.1)  # Reduced from 0.2

def main():
    """Entry point that runs the async main function"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 