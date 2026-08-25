import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --------------------------------------------------
# 1. Load CSV
# --------------------------------------------------

csv_file_path = os.path.join(
    os.path.dirname(__file__),
    "spamham.csv"
)

df = pd.read_csv(
    csv_file_path,
    engine="python"
)

print(f"CSV loaded: {len(df)} rows")
print("Columns:", df.columns.tolist())


# --------------------------------------------------
# 2. Connect to MongoDB
# --------------------------------------------------

mongo_uri = os.getenv("MONGODB_URL")

if mongo_uri is None:
    raise Exception("MONGODB_URL is not set in .env")

client = MongoClient(mongo_uri)

# Test MongoDB connection
client.admin.command("ping")

print("MongoDB connection successful!")


# --------------------------------------------------
# 3. Select Database and Collection
# --------------------------------------------------

db = client["Firstdatabase"]

collection = db["emaildata"]


# --------------------------------------------------
# 4. Convert CSV data to MongoDB documents
# --------------------------------------------------

data = df.to_dict(orient="records")


# --------------------------------------------------
# 5. Insert entire dataset
# --------------------------------------------------

if len(data) > 0:

    result = collection.insert_many(data)

    print(
        f"Successfully inserted {len(result.inserted_ids)} "
        "documents into MongoDB!"
    )

else:

    print("No data found in CSV.")


# --------------------------------------------------
# 6. Close connection
# --------------------------------------------------

client.close()

print("MongoDB connection closed.")