from pymongo import MongoClient

class MongoAPI:
	def  __init__(self, data):
		self.client = MongoClient('mongodb+srv://admin:skillsmetter2020@cluster0.1njxt.gcp.mongodb.net/sm-web?retryWrites=true&w=majority')

		database = data['database']
		collection = data['collection']

		cursor = self.client[database]
		self.collection = cursor[collection]
		self.data = data

	def read(self):
		documents = self.collection.find(self.data['filter'], self.data['projection'])
		output = [{item: data[item] for item in data} for data in documents]

		return output

	# def write(self, data):
	# 	"""Rework!!!"""

	# 	log.info('Writing Data')
	# 	new_document = data['Document']
	# 	response = self.collection.insert_one(new_document)
	# 	output = {'Status': 'Successfully inserted', 'Document_ID': str(response.inserted_id)}

	# 	return output

	def update(self, data):
		filt = data['filter']
		updated_data = data['updated_data']
		response = self.collection.update_one(filt, updated_data)
		output = {'Status': 'Successfully updated' if response.modified_count > 0 else 'Nothing was updated'}

		return output

	# def delete(self, data):
	# 	"""Rework!!!"""
		
	# 	filt = data['Document']
	# 	response = self.collection.delete_one(filt)
	# 	output = {'Status': 'Successfully deleted' if response.deleted_count > 0 else 'Document not found'}

	# 	return output