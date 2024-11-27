# Subreddit Categorizer

An automated tool that uses a local LLM (Phi 3.5 Mini Instruct) and the Reddit API to categorize subreddits based on their content, description, and top posts.

## Features

- Automated subreddit categorization using local LLM
- Reddit API integration for fetching subreddit data
- CSV output with categorization results
- Built-in rate limiting for API compliance
- Support for custom categories via JSON configuration

## Prerequisites

- Python 3.7+
- llama.cpp server with Phi 3.5 Mini Instruct model
- Internet connection

## Quick Start

1. **Install Dependencies**

pip install openai requests

2. **Start LLM Server**

./llama-server \
-m ./models/Phi-3.5-mini-instruct-Q6_K.gguf \
--temp 0.1 \
--ctx-size 16384 \
--batch-size 512 \
--n-gpu-layers 100 \
--parallel 2 \
--cont-batching \
--threads 8 \
--host 0.0.0.0

3. **Prepare Input Files**
   - Create `to_categorise.txt` with one subreddit name per line (without "r/" prefix):
     ```
     programming
     AskReddit
     science
     funny
     ```
   - Ensure `subreddits.json` is present in the working directory

4. **Run the Script**

python categorise.py


## Output Format

The script generates `categorised.csv` with the following structure:

subreddit,main_category,sub_category
programming,Technology,Software Development
AskReddit,General,Discussion
science,Science,Scientific Research


## Configuration

### Rate Limits
- Reddit API calls: 2 second delay
- Subreddit processing: 1 second delay

### Files
| File | Description |
|------|-------------|
| `categorise.py` | Main script |
| `to_categorise.txt` | Input subreddit list |
| `subreddits.json` | Category definitions |
| `categorised.csv` | Results output |

## Troubleshooting

### LLM Server Issues
- Verify server is running at `x9dri.local:8080`
- Check model loading status
- Confirm server configuration

### Reddit API Problems
- Verify internet connectivity
- Check subreddit name spelling
- Confirm subreddit accessibility

### Categorization Accuracy
- Review `subreddits.json` category definitions
- Adjust LLM temperature setting
- Verify input data quality

## Limitations

- Uses unauthenticated Reddit API (rate-limited)
- Category suggestions limited by `subreddits.json` definitions
- Processing speed dependent on:
  - Number of subreddits
  - Server performance
  - Network conditions

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

[Add your license information here]