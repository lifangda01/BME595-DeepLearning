from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Save Drive file as a local file
# file3.GetContentFile(file3['title'])  

class Drive(object):
	def __init__(self):
		super(Drive, self).__init__()
		gauth = GoogleAuth()
		gauth.LocalWebserverAuth()
		self.drive = GoogleDrive(gauth)
		self.train_tumor_id = '0BzsdkU4jWx9BUzVXeUg0dUNOR1U'
		self.train_normal_id = '0BzsdkU4jWx9BNUFqRE81QS04eDg'
		self.ground_truth_mask_id = '0BzsdkU4jWx9BVlM5UHNHMXNjbE0'

	# Auto-iterate through all files in the root folder.
	def get_file_list(self, parent_id):
		return self.drive.ListFile({'q': "'{0}' in parents and trashed=False".format(parent_id)}).GetList()
