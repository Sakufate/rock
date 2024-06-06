#python-dotenv
from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv('MY_API_KEY'))