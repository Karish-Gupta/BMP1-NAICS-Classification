import asyncio
import logging
import warnings
from webscraping.scraping_utils import append_business_description_data

# Suppress logging noise
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# CONFIG
INPUT_FILE = 'data/train_data/unprocessed_data_part_7.csv'
OUTPUT_FILE = 'website_summaries_train_part_7.csv'

if __name__ == "__main__":
    asyncio.run(append_business_description_data(INPUT_FILE, OUTPUT_FILE))