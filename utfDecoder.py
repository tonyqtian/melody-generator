import os
from six.moves import cPickle as pickle

from argparse import ArgumentParser
parser = ArgumentParser(description='text to unicode int')
parser.add_argument('--source', default='', help='path to coded text file')
args = parser.parse_args()

filehandle = args.source

#with open("wiki_cn_chunk.txt", 'r') as f:
def Decoder(sourcefile="midi2text_out.txt.utf8", rev_dict_file="reverse_dictionary.pickle"):
			
	if sourcefile.endswith('.utf8'):
		rev_dict_file = sourcefile[:-5]+'_reverse_dictionary.pickle'
	else:
		rev_dict_file = sourcefile+'_reverse_dictionary.pickle'
	target_filename = sourcefile+'decode'
	
	if os.path.exists(rev_dict_file):
		try:
			print("Pickle files found, try to load data from pickle files...")
			with open(rev_dict_file, 'rb') as f:
				reverse_dictionary = pickle.load(f)
			print("Data loaded from pickle files successfully")
		except Exception as e:
			print('Unable to load data', ':', e)
	else:
		print('Pickle file ',rev_dict_file , 'not found...')
		return
	with open(sourcefile, 'r') as f:
		print("Loading words from text file...")
		lines = f.read().strip().split()
		#print(lines[:10])
		words = []
		for line in lines:
			words.extend(list(line))
		print('Data size %d' % len(words))
		print(words[:10])
		
		decoded = []
		for w in words:
			decoded.append(reverse_dictionary[ord(w)])
	
	with open(target_filename, 'w') as fout:
		for w in decoded:
			fout.write(w)
			fout.write(' ')

Decoder(sourcefile=filehandle)