# _*_ coding=UTF-8 -*-

'''
Created on Jan 23, 2018

@author: Administrator
'''


class test(object):
    
    def method1(self):
        print('method1')
    
    @staticmethod
    def method2(self):
        print('method2')
        
    @classmethod
    def method3(self):
        print('method3')
        

f = open('test.txt')
text=f.readlines()

words = list()
[words.extend(sentence.split()) for sentence in text]
vocab = set(words)
word_dict = {word:words.count(word)  for word in vocab}
print(word_dict)




