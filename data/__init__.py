import pymongo
from configparser import ConfigParser


config = ConfigParser()
config.read("config.ini")
host = config.get("DATABASE", "host")
username = config.get("DATABASE", "username")
password = config.get("DATABASE", "password")
db_name = config.get("DATABASE", "db")

class BMediaDB:
    def __init__(self):
        self.db_client = pymongo.MongoClient(
            f"mongodb://"
            f"{username}:{password}@{host}"
        )
        self.db = self.db_client[f"{db_name}"]