from pymongo import MongoClient
from pymongo.errors import BulkWriteError
from bulk_op import BulkOperation

def insert(dataset):
  dataset_variant = [0 for i in range(0, len(dataset))]
  #print 'dataset        ', dataset
  for index, data in enumerate(dataset):
    if data == 0:
      # when index is 0, first one.
      if index == 0:
        for data1 in dataset:
          if data1 != 0:
            dataset_variant[0] = data1
            break
      # when index is between [1, 10].
      elif index < len(dataset) - 1:
        # [43, 0, 0], index-1 => 43, index => 0, index+1 => 0.
        if dataset[index + 1] == 0:
          dataset_variant[index] = dataset_variant[index - 1]
        # [43, 0, 45], index-1 => 43, index => 0, index+1 => 45.
        else:
          dataset_variant[index] = (dataset_variant[index - 1] + dataset[index + 1]) / 2
      # when index is 11, last one.
      else:
        dataset_variant[index] = dataset_variant[index - 1]
    else:
      dataset_variant[index] = data
  #print 'dataset_variant', dataset_variant
  return dataset_variant

def output():
  # db_url = 'mongodb://localhost-am:BGnd03kntrHL@172.16.0.6:27017/am'
  db_url = 'mongodb://ec2-am:BGnd03kntrHL@52.27.193.168:27017/am'
  db_client = MongoClient(db_url)
  db = db_client.am

  # find the record, wherein Both TList and KwhList are not empty
  results = db.Merged.find()
  index = 0
  for result in results:
    index = index + 1
    print index
    # preallocate an empty t list of the size of 12, with all the values of 0
    t_list = [0 for i in range(0, 12)]
    for record in result['TList']:
      timestamp = record['Timestamp']
      hour = timestamp.hour
      t_list[hour / 2] = record['T']

    # preallocate an empty kwh list of the size of 12, with all the values of 0
    k_list = [0 for i in range(0, 12)]
    for record in result['KList']:
      timestamp = record['Timestamp']
      hour = timestamp.hour
      k_list[hour / 2] = record['Kwh']

    # preallocate an empty voltage list of the size of 12, with all the values of 0
    v_list = [0 for i in range(0, 12)]
    for record in result['VList']:
      timestamp = record['Timestamp']
      hour = timestamp.hour
      v_list[hour / 2] = record['Voltage']

    # insert
    t_list = insert(t_list)
    k_list = insert(k_list)
    v_list = insert(v_list)
    data = {
      '_id': index,
      'town': result['Town'],
      'loc': result['Loc'],
      'longitude': result['Longitude'],
      'latitude': result['Latitude'],
      'amiModel': result['AMIModel'],
      'meter': result['Meter'],
      'year': result['Year'],
      'month': result['Month'],
      'day': result['Day']
    }
    for values, keyword in zip([t_list, k_list, v_list], ['t', 'k', 'v']):
      values_len = len(values)
      for i in xrange(0, values_len):
        data[keyword + str(i)] = values[i]

    db.Merged_Output.insert(data)

if __name__ == '__main__':
  output()