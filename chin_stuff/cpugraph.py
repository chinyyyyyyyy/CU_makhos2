#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 01:19:34 2020

@author: chin
"""
import matplotlib.pyplot as plt
import math

class cpustat():
    def __init__(self,cpunum,totaltask):
        self.cpunum = cpunum
        self.totaltask = totaltask
        self.data = dict()
        self.min = 0
        self.max = 0
        self.duration  = 0
        
    def add(self,cpuname,iterth,start,end):
        if cpuname not in self.data:
            self.data[cpuname] = []
        self.data[cpuname].append((iterth,start,end,end - start))
        if iterth == 0:
            self.min = start
        if iterth == (self.totaltask - 1) :
            self.max = end
            self.duration  = self.max  - self.min
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(self.duration*2,4))
        for i in self.data.keys():
            temp = list()
            for j in self.data[i]:
                temp.append((j[1],j[3]))
            ax.broken_barh(temp,(int(i)-0.25,0.5))
        plt.show()