import pyodbc
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

class BulkOperation(object):
	def __init__(self, collection):
		self.bulkSize = 1000
		self.collection = collection
		self.dataset = []

	def insert(self, data):
		self.dataset.append(data)
		if (len(self.dataset) == self.bulkSize):
			try:
				result = self.collection.insert_many(self.dataset)
				print result.inserted_ids[-1]
				self.dataset = []
			except BulkWriteError as bwe:
				print bwe.details

	def flush(self):
		if len(self.dataset) != 0:
			try:
				result = self.collection.insert_many(self.dataset)
				print result.inserted_ids[-1]
				self.dataset = []
			except BulkWriteError as bwe:
				print bwe.details

# remote db url
remoteDBUrl = 'mongodb://localhost-am:BGnd03kntrHL@172.16.0.6:27017/am'
#remoteDBUrl = 'mongodb://ec2-am:BGnd03kntrHL@52.27.193.168:27017/am'
remoteClient = MongoClient(remoteDBUrl)
remoteDB = remoteClient.am

# local db file
localDBStr = 'DRIVER={Microsoft Access Driver (*.mdb)};DBQ=Database.mdb'
localConn = pyodbc.connect(localDBStr)
cursor = localConn.cursor()

# Load 13209506
# Load2 632223
# Temp 1162837
collections = ['Load', 'Load2', 'Temp']
for collection in collections:
	SQL = 'SELECT * from {0}'.format(collection)
	bulkOp = BulkOperation(remoteDB[collection])
	for row in cursor.execute(SQL):
		descriptions = row.cursor_description
		data = {}
		for index, description in enumerate(descriptions):
			if description[0] == 'ID':
				data['_id'] = row.ID
			else:
				data[description[0]] = row[index]
		bulkOp.insert(data)
	bulkOp.flush()

cursor.close()
localConn.close()
remoteClient.close()
