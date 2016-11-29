
# Melody Generator

A midi melody generator based on LSTM. The model generator is a char based LSTM model trainer coded in Lua by Karpathy. I adopted a modified version of it which added utf code support by zhangzibin. https://github.com/karpathy/char-rnn and Nal Kalchbrenner's paper http://arxiv.org/abs/1507.01526

## Understanding LSTM Networks

This is the best introduction about LSTM networks I found.

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Structure of this model

<!---
![MODEL STRUCTURE](/demo/model-lstm-layer.png?raw=true)
-->

I adopted a 4 layer LSTM each with 512 node and 0.5 dropout

[ ](/demo/model-lstm-layer.png?raw=true)
<a href="url"><img src="/demo/model-lstm-layer.png" height="300" ></a>

some tensorflow pesudo-code like this:
```python
g = tflearn.input_data([None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
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

### 2. encode text string with utf-8 code

```
python3 utfCoder.py --source test.mid3.txt
```
it will generate a dictionary to encode each string into an id, and encode those ids into utf-8
so that your model will treat your input as those id strings and generate new ones after training

keep the reverse dictionary pickle file, you will use it to decode generated sample files after training

### 3. train your model

read karpathy's readme to get your environment ready → https://github.com/karpathy/char-rnn
 
put your encoded source file in a specific folder and rename it to input.txt

```
train.lua -data_dir data/your_input_file_folder -rnn_size 512 -num_layers 4 -dropout 0.5
```

### 4. sample some coded melody

you will get some .t7 files after the training, use the final one to sample some output
```
th sample.lua cv/lm_lstm_epoch50.00_1.0000.t7 -length 100 -verbose 0 >sample.utf8
```

### 5. decode sample to text string

find your reverse dictionary pickle file, rename it to specific file name so that this script can find and load it
```
python3 utfDecoder.py --source ../your_folder/sample.utf8
```

### 6. text string → midi

```
python text2midi.py --source ../your_folder/sample.utf8decode
```
then you will get your generated midi file

### 7. play it!

-----------------------------------------
## To-do list

I'm planning to add the following features in the future

1. Embedding will be added to make the training memory friendly

2. Add more co-related midi melodies to enlarge the learning material

3. Use GPU to speed up the training

-----------------------------------------------
## Karpathy's raw Readme
please follow Karpathy's readme to setup your experiment for training and sampling.

https://github.com/karpathy/char-rnn

-----------------------------------------------
## zhangzibin's Readme
The following are zhangzibin's modifications of Karpathy's char-rnn
https://github.com/zhangzibin/char-rnn-chinese

### Chinese process
Make the code can process both English and Chinese characters.
This is my first touch of Lua, so the string process seems silly, but it works well.

### opt.min_freq
I also add an option called 'min_freq' because the vocab size in Chinese is very big, which makes the parameter num increase a lot.
So delete some rare character may help.

### Scheduled Sampling
Samy Bengio's paper [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](http://arxiv.org/abs/1506.03099) in NIPS15 propose a simple but power method to implove RNN.

In my experiment, I find it helps a lot to avoid overfitting and make the test loss go deeper. I only use linear decay.

Use `-use_ss` to turn on or turn off scheduled sampling, default is on. `-start_ss` is the start aomunt of real data, I suggest to use 1 because our model should learn data without noise at the very beginning. `-min_ss` is also very important as too much noise will hurt performance. Finally, `-decay_ss` is the linear decay rate.

### Model conversion between cpu and gpu
I add a script to convert a model file trained by gpu to cpu model.
You can try it as follow:
```bash
$ th convert.lua gpu_model cpu_model
```

---------------------------------------------------------
## License

MIT
