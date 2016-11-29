import os
import collections
from six.moves import cPickle as pickle

from argparse import ArgumentParser
parser = ArgumentParser(description='text to unicode int')
parser.add_argument('--source', default='', help='path to coded text file')
args = parser.parse_args()

filehandle = args.source
	
vocabulary_size = 4000
min_count = 2

def build_dataset(words):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()
	for word, count_num in count:
		if count_num > min_count:
			dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
			data.append(index)
		else:
			index = 0	# dictionary['UNK']
			unk_count = unk_count + 1
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
	return data, count, dictionary, reverse_dictionary

def maybe_pickle(target_data, set_filename, force=False):
	if os.path.exists(set_filename) and not force:
		if os.path.getsize(set_filename) > 0:
			# You may override by setting force=True.
			print('%s already present - Skipping pickling.' % set_filename)
			return set_filename
	print('Pickling %s.' % set_filename)
	try:
		with open(set_filename, 'wb') as f:
			pickle.dump(target_data, f, pickle.HIGHEST_PROTOCOL)
	except Exception as e:
		print('Unable to save data to', set_filename, ':', e)

#with open("wiki_cn_chunk.txt", 'r') as f:
def loadData(sourcefile="midi2text_out.txt", force=False, \
			data_file="data.pickle", count_file="count.pickle", \
			dict_file="dictionary.pickle", rev_dict_file="reverse_dictionary.pickle"):
			
	data_file = sourcefile+"_data.pickle"
	count_file = sourcefile+"_count.pickle"
	dict_file = sourcefile+"_dictionary.pickle"
	rev_dict_file = sourcefile+"_reverse_dictionary.pickle"
	
	if os.path.exists(data_file) and os.path.exists(count_file) and os.path.exists(dict_file) and \
		 os.path.exists(rev_dict_file) and not force:
		try:
			print("Pickle files found, try to load data from pickle files...")
			with open(data_file, 'rb') as f:
				data = pickle.load(f)
			with open(count_file, 'rb') as f:
				count = pickle.load(f)
			with open(dict_file, 'rb') as f:
				dictionary = pickle.load(f)
			with open(rev_dict_file, 'rb') as f:
				reverse_dictionary = pickle.load(f)
			print("Data loaded from pickle files successfully")
			print('Most common words (+UNK)', count[:5])
			print('Dictionary Size: ', len(dictionary))
			print('Least common words', count[len(dictionary)-5:len(dictionary)])
			print('Sample data', data[:10])
			return data, count, dictionary, reverse_dictionary
		except Exception as e:
			print('Unable to load data', ':', e)
	with open(sourcefile, 'r') as f:
		print("Loading words from text file...")
		lines = f.read().strip().split()
		#print(lines[:10])
		words = []
		for line in lines:
			words.append(line)
		print('Data size %d' % len(words))
		print(words[:10])
		
		print("Cooking data from words loaded...")
		data, count, dictionary, reverse_dictionary = build_dataset(words)
		print('Most common words (+UNK)', count[:5])
		print('Dictionary Size: ', len(dictionary))
		print('Least common words', count[len(dictionary)-5:len(dictionary)])
		print('Sample data', data[:10])
		del words	# Hint to reduce memory.
	
		print("Saving cooked data into pickle files...")
		maybe_pickle(dictionary, dict_file, force=force)
		maybe_pickle(reverse_dictionary, rev_dict_file, force=force)
		maybe_pickle(count, count_file, force=force)
		maybe_pickle(data, data_file, force=force)
		
	with open(sourcefile+".utf8", 'w') as fout:
		for w in data:
			fout.write(chr(w))
		
	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = loadData(sourcefile=filehandle, force=True)