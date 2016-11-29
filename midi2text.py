#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import struct
from binascii import *
from types import *
reload(sys)
sys.setdefaultencoding('utf-8')

def is_eq_0x2f(b):
  return int(b2a_hex(b), 16) == int('2f', 16)

def is_gte_0x80(b):
  return int(b2a_hex(b), 16) >= int('80', 16)

def is_eq_0xff(b):
  return int(b2a_hex(b), 16) == int('ff', 16)

def is_eq_0xf0(b):
  return int(b2a_hex(b), 16) == int('f0', 16)

def is_eq_0xf7(b):
  return int(b2a_hex(b), 16) == int('f7', 16)

def is_eq_0x8n(b):
  return int(b2a_hex(b), 16) >= int('80', 16) and int(b2a_hex(b), 16) <= int('8f', 16)

def is_eq_0x9n(b):
  return int(b2a_hex(b), 16) >= int('90', 16) and int(b2a_hex(b), 16) <= int('9f', 16)

def is_eq_0xan(b): # An: 3byte
  return int(b2a_hex(b), 16) >= int('a0', 16) and int(b2a_hex(b), 16) <= int('af', 16)

def is_eq_0xbn(b): # Bn: 3byte
  return int(b2a_hex(b), 16) >= int('b0', 16) and int(b2a_hex(b), 16) <= int('bf', 16)

def is_eq_0xcn(b): # Cn: 2byte
  return int(b2a_hex(b), 16) >= int('c0', 16) and int(b2a_hex(b), 16) <= int('cf', 16)

def is_eq_0xdn(b): # Dn: 2byte
  return int(b2a_hex(b), 16) >= int('d0', 16) and int(b2a_hex(b), 16) <= int('df', 16)

def is_eq_0xen(b): # En: 3byte
  return int(b2a_hex(b), 16) >= int('e0', 16) and int(b2a_hex(b), 16) <= int('ef', 16)

def is_eq_0xfn(b):
  return int(b2a_hex(b), 16) >= int('f0', 16) and int(b2a_hex(b), 16) <= int('ff', 16)

def mutable_lengths_to_int(bs):
  length = 0
  for i, b in enumerate(bs):
    if is_gte_0x80(b):
      length += ( int(b2a_hex(b), 16) - int('80', 16) ) * pow(int('80', 16), len(bs) - i - 1)
    else:
      length += int(b2a_hex(b), 16)
  return length

def int_to_mutable_lengths(length):
  length = int(length)
  bs = []
  append_flag = False
  for i in range(3, -1, -1):
    a = length / pow(int('80', 16), i)
    length -= a * pow(int('80', 16), i)
    if a > 0:
      append_flag = True
    if append_flag:
      if i > 0:
        bs.append(hex(a + int('80', 16))[2:].zfill(2))
      else:
        bs.append(hex(a)[2:].zfill(2))
  return bs if len(bs) > 0 else ['00']

def read_midi(path_to_midi):
  midi = open(path_to_midi, 'rb')
  data = {'header': [], 'tracks': []}
  track = {'header': [], 'chunks': []}
  chunk = {'delta': [], 'status': [], 'meta': [], 'length': [], 'body': []}
  current_status = None

  """
  Load data.header
  """
  bs = midi.read(14)
  data['header'] = [b for b in bs]

  while 1:
    """
    Load data.tracks[0].header
    """
    if len(track['header']) == 0:
      bs = midi.read(8)
      if bs == '':
        break
      track['header'] = [b for b in bs]

    """
    Load data.tracks[0].chunks[0]
    """
    # delta time
    # ----------
    b = midi.read(1)
    while 1:
      chunk['delta'].append(b)
      if is_gte_0x80(b):
        b = midi.read(1)
      else:
        break

    # status
    # ------
    b = midi.read(1)
    if is_gte_0x80(b):
      chunk['status'].append(b)
      current_status = b
    else:
      midi.seek(-1, os.SEEK_CUR)
      chunk['status'].append(current_status)

    # meta and length
    # ---------------
    if is_eq_0xff(current_status): # meta event
      b = midi.read(1)
      chunk['meta'].append(b)
      b = midi.read(1)
      while 1:
        chunk['length'].append(b)
        if is_gte_0x80(b):
          b = midi.read(1)
        else:
          break
      length = mutable_lengths_to_int(chunk['length'])
    elif is_eq_0xf0(current_status) or is_eq_0xf7(current_status): # sysex event
      b = midi.read(1)
      while 1:
        chunk['length'].append(b)
        if is_gte_0x80(b):
          b = midi.read(1)
        else:
          break
      length = mutable_lengths_to_int(chunk['length'])
    else: # midi event
      if is_eq_0xcn(current_status) or is_eq_0xdn(current_status):
        length = 1
      else:
        length = 2

    # body
    # ----
    for i in range(0, length):
      b = midi.read(1)
      chunk['body'].append(b)

    track['chunks'].append(chunk)


    if is_eq_0xff(chunk['status'][0]) and is_eq_0x2f(chunk['meta'][0]):
      data['tracks'].append(track)
      track = {'header': [], 'chunks': []}
    chunk = {'delta': [], 'status': [], 'meta': [], 'length': [], 'body': []}

  return data

def write_text(tracks, outputfilename):
  midi = open(outputfilename, 'w')
  for track in tracks:
    for chunks in track:
      midi.write('{} '.format(chunks))

if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser(description='audio RNN')
  parser.add_argument('--source', type=unicode, default='', help='path to the MIDI file')
  args = parser.parse_args()

  data = read_midi(args.source)

  track_length = len(data['tracks'])
  print('Total Tracks: ', str(track_length))
  # extract midi track
  track_list = range(track_length)

  for n in track_list:
    print('Processing track No. ', str(n))
    tracks = []
    raw_data = []
    chunks = data['tracks'][n]['chunks']
    for i in range(0, len(chunks)):
      chunk = chunks[i]
      if is_eq_0xff(chunk['status'][0]) or \
         is_eq_0xf0(chunk['status'][0]) or \
         is_eq_0xf7(chunk['status'][0]) :
        continue
      raw_data.append('_'.join(
        [str(mutable_lengths_to_int(chunk['delta']))] +
        [str(b2a_hex(chunk['status'][0]))] +
        [str(b2a_hex(body)) for body in chunk['body']]
      ))
    tracks.append(raw_data)
    write_text(tracks, args.midi+str(n)+'.txt')
