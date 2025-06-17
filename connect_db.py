from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

uri = "mongodb://localhost:27017"  # Thay URI của bạn vào đây

class CheckDB:
    def __init__(self, uri="mongodb://localhost:27017", db_name="chatbot_db", col_name="chunks"):
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            self.client.server_info()  # Test kết nối
            self.db = self.client[db_name]
            self.collection = self.db[col_name]
            print("✅ Đã kết nối MongoDB thành công!")

        except ConnectionFailure as e:
            print("❌ Kết nối MongoDB thất bại:", e)
            self.db = None
            self.collection = None





