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
openai.api_base = "https://api.mistral.ai/v1"
openai.api_key = os.getenv("MISTRAL_API_KEY")

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

    # Format existing subcategories as a simple list without numbers
    existing_subcats = '\n'.join(f"- {cat}" for cat in categories['third_column'])

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

Available Subcategories (select one or suggest NEW):
{existing_subcats}

Respond ONLY with these three lines:
MAIN_CATEGORY: <select from available main categories>
SUB_CATEGORY: <select from list OR write NEW:suggested_category>
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
    """Process LLM queries one at a time"""
    results = []
    
    for prompt_data in prompts_data:
        # Process one prompt at a time
        result = await process_single_prompt(prompt_data, categories)
        results.append(result)
        # Respect Mistral's 1 request per second limit
        await asyncio.sleep(1.1)  # Slightly over 1 second to be safe
    
    return results

async def process_batch(reddit_api: RedditAPI, batch: List[str], categories: dict) -> dict:
    """Process subreddits one at a time"""
    last_rate_limits = None
    async with aiohttp.ClientSession() as session:
        results = []
        for subreddit in batch:
            # Fetch Reddit data for one subreddit
            top_data, about_data, rate_limits = await get_reddit_data_async(reddit_api, session, subreddit)
            
            if not top_data or not about_data:
                continue
                
            last_rate_limits = rate_limits
            print(f"Preparing prompt for r/{subreddit}...")
            prompt = create_prompt(subreddit, top_data, about_data, categories)
            
            # Process one LLM query
            llm_result = await process_llm_batch([(subreddit, prompt, about_data)], categories)
            results.extend(llm_result)
            
            # Write result to CSV
            for result in llm_result:
                if result[1]:  # if main_category exists
                    await asyncio.to_thread(write_to_csv, *result)
        
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
    
    # Process one subreddit at a time
    batch_size = 1
    
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
        
        # Reduced delay between subreddits
        await asyncio.sleep(1.1)  # Slightly over 1 second to be safe

def main():
    """Entry point that runs the async main function"""
    asyncio.run(main_async())

async def process_single_prompt(prompt_data, categories):
    """Process a single LLM prompt"""
    subreddit, prompt, about_data = prompt_data
    if not prompt:
        return subreddit, "Adult and NSFW", "", ""
        
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = openai.AsyncOpenAI(
                base_url="https://api.mistral.ai/v1",
                api_key=os.getenv("MISTRAL_API_KEY")
            )
            
            formatted_prompt = f"""<|system|>
You are a categorization assistant. Respond ONLY with the three required lines. No explanations or additional text.
<|end|>
<|user|>
{prompt}
<|end|>
<|assistant|>"""
            
            print(f"\nAttempting to process r/{subreddit}...")
            
            # Add delay between requests
            if attempt > 0:
                wait_time = 2 ** attempt
                print(f"Rate limit hit. Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            
            response = await client.chat.completions.create(
                model="mistral-tiny",
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            print(f"\nRaw response for r/{subreddit}:")
            print(response)
            
            if hasattr(response, 'error'):
                error_message = response.error.get('message', 'Unknown error')
                if 'rate limit' in error_message.lower():
                    print(f"Rate limit error: {error_message}")
                    continue  # Try again after waiting
                else:
                    print(f"API error: {error_message}")
                    
            if not response or not response.choices:
                print(f"Empty or invalid response received for r/{subreddit}")
                continue
            
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
                
                await asyncio.sleep(1.1)  # Slightly over 1 second to be safe
                return subreddit, main_category, sub_category, specialist
            
            else:
                print(f"Invalid response format for r/{subreddit} on attempt {attempt + 1}. Retrying...")
        
        except Exception as e:
            print(f"Error processing r/{subreddit} on attempt {attempt + 1}: {e}")
        
    print(f"Failed to get valid response for r/{subreddit} after {max_retries} attempts.")
    return subreddit, None, None, None

if __name__ == "__main__":
    main() 