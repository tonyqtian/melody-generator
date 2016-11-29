#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import struct
from binascii import *
from types import *
reload(sys)
sys.setdefaultencoding('utf-8')

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

def write_midi(tracks, outputfilename):
  print len(tracks)
  midi = open(outputfilename, 'wb')

  """
  MIDI Header
  """
  header_bary = bytearray([])
  header_bary.extend([0x4d, 0x54, 0x68, 0x64, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00])
  header_bary.extend([int(hex(len(tracks))[2:].zfill(4)[i:i+2], 16) for i in range(0, 4, 2)])
  header_bary.extend([0x01, 0xe0])
  midi.write(header_bary)

  for track in tracks:
    track_bary = bytearray([])
    for chunk in track:
      # It is assumed that each chunk consists of just 4 elements
      if len(chunk.split('_')) != 4:
        continue
      int_delta, status, data1, data2 = chunk.split('_')

      if status[0] == '8' or status[0] == '9' or status[0] == 'a' or status[0] == 'b' or status[0] == 'e': # 3byte
        delta = int_to_mutable_lengths(int_delta)
        track_bary.extend([int(d, 16) for d in delta])
        track_bary.extend([int(status, 16)]) 
        track_bary.extend([int(data1, 16)])  
        track_bary.extend([int(data2, 16)])  
      elif status[0] == 'c' or status[0] == 'd':
        delta = int_to_mutable_lengths(int_delta)
        track_bary.extend([int(d, 16) for d in delta])
        track_bary.extend([int(status, 16)]) 
        track_bary.extend([int(data1, 16)])  
      else:
        print status[0]

    """
    Track header
    """
    header_bary = bytearray([])
    header_bary.extend([0x4d, 0x54, 0x72, 0x6b])
    header_bary.extend([int(hex(len(track_bary)+4)[2:].zfill(8)[i:i+2], 16) for i in range(0, 8, 2)])
    midi.write(header_bary)

    """
    Track body
    """
    print len(track_bary)
    midi.write(track_bary)

    """
    Track footer
    """
    footer_bary = bytearray([])
    footer_bary.extend([0x00, 0xff, 0x2f, 0x00])
    midi.write(footer_bary)

if __name__ == '__main__':
  from argparse import ArgumentParser
  parser = ArgumentParser(description='audio RNN')
  parser.add_argument('--source', type=unicode, default='', help='path to text file')
  args = parser.parse_args()
  
  filename = args.source
  with open(filename, 'r') as f:
    txt = f.read().strip()
  
  #txt = '0_80_40_00 0_90_4f_64 0_90_43_64 120_80_4f_00 0_80_43_00 0_90_51_64 0_90_45_64 480_80_51_00 0_80_45_00 0_90_4c_64 0_90_44_64 120_80_4c_00 0_80_44_00 0_90_4f_64 0_90_43_64 60_80_4f_00 0_80_43_00 0_90_4d_64 0_90_41_64 120_80_4d_00'
  tracks = [txt.split(' ')]
  write_midi(tracks, filename+'.mid')
