
# Melody Generator

A midi melody generator. This model generator is based on a poetry lstm model generator coded by dvictor. I modified for midi words generator. https://github.com/dvictor/lstm-poetry-word-based

## Understanding LSTM Networks

This is the best introduction about LSTM networks I found.

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Structure of this model

<!---
![MODEL STRUCTURE](/demo/model-lstm-layer.png?raw=true)
-->

I used a 2 layer LSTM each with 400 node and 0.6 dropout

[ ](/demo/model-lstm-layer.png?raw=true)
<a href="url"><img src="/demo/model-lstm-layer.png" height="300" ></a>

some tensorflow pesudo-code like this:
```python
g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 400, return_seq=True)
g = tflearn.dropout(g, 0.6)
g = tflearn.lstm(g, 400)
g = tflearn.dropout(g, 0.6)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.0)
```
-------------------------------------------------
## Quick Start

### 0. get some midi file

Google "midi download" and get them :-)

### 1. midi → text string

```
python midi2text.py --source test.mid
```
this will generate a txt sufficed file which looks like this
```
0_b0_65_00 0_b0_64_02 0_b0_06_40 60_b0_65_00 0_b0_64_01 0_b0_06_40 0_b0_26_00 ...
```

### 2. train your model

put your encoded source file in a specific folder and rename it to input.txt

```
python3 src/train.py -source ../data/your_input_file_folder --num_layers 2 --hidden_size 400 --dropout 0.6
```

### 3. sample some coded melody

you will get some tensorflow dump files after the training, use the final one to sample some output

```
python3 src/generate.py --source ../data/your_input_file_folder --output sample.txt --length 100
```

if you'd like to use your own sample seed, you can use '--header' arg

### 4. text string → midi

```
python text2midi.py --source ../your_folder/sample.utf8decode
```
then you will get your generated midi file

### 5. play it!

-----------------------------------------
## To-do list (already done)

~~I'm planning to add the following features in the future~~

1. Embedding will be added to make the training memory friendly

2. Add more co-related midi melodies to enlarge the learning material

3. Use GPU to speed up the training

-----------------------------------------------
## Karpathy's raw Readme

this is where my original idea came from...

please follow Karpathy's readme to setup your experiment for training and sampling.

https://github.com/karpathy/char-rnn

---------------------------------------------------------
## License

MIT
