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