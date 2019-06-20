#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:00:31 2019

@author: sleek_eagle
"""
from os import walk

#get all the file names with the extenstion ext
def get_ext_paths(path,ext):
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        #files.extend(filenames)
        files.append([dirpath,filenames])
        
    ext_files = []
    for item in files:
        dirname = item[0]
        filenames = item[1]
        for filename in filenames:
            if (filename[-3:] != ext):
                continue
            ext_files.append(dirname + "/" + filename)
    
    return ext_files