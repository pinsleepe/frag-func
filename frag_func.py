import gspread
from oauth2client.service_account import ServiceAccountCredentials

from os.path import join, dirname
from dotenv import load_dotenv
import os

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

SECRET_TOKEN = os.getenv("CLIENT_SECRET_JSON")

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name(SECRET_TOKEN,
                                                         scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
# Make sure you use the right name here.
spreadsheet = client.open("MARKET FRAGMENTATION RESEARCH_copy")

# Extract and print all of the values
# list_of_hashes = sheet.get_all_records()
# print(list_of_hashes)
