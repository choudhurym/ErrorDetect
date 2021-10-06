#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:58:56 2020

@author: Muntabir Choudhury
"""

#importing libraries
import glob
import fileinput
import re
import pandas as pd
from nltk import word_tokenize
from nltk.util import ngrams
import stanza
import csv
from nltk import pos_tag
from spellchecker import SpellChecker


#read the files in a sorted order
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#noisy-dataset path
data_path = '/Users/muntabir/Desktop/Graduate-School-ODU/CS895/project/source/noisy/*.txt'
dict_path = '/Users/muntabir/Desktop/Graduate-School-ODU/CS895/project/source/clean/*txt'
noisy_files = sorted(glob.glob(data_path), key=numericalSort)
clean_files = sorted(glob.glob(dict_path), key=numericalSort)

#write all the noisy data files from a directory and merged it in a sorted order
with open('noisy_data.txt', 'w', encoding = 'utf-8') as outfile:
    for lines in fileinput.input(noisy_files):
        outfile.write(lines)  

#write all the clean data files from the dicrectory and merged it in a sorted order
with open('dictionary.txt', 'w', encoding = 'utf-8') as outfile:
    for lines in fileinput.input(clean_files):
        outfile.write(lines)
        outfile.write("\n")                        
            

################## LEXICON LOOKUP #################################

#create a curated dictionary from OCR output which is clean and without misspellings
def lexicon():
    dictionaryWords = []
    files = open('dictionary.txt', 'r', encoding = 'utf-8') # this is test dictionary file
    for line in files:
        word = line.lower().split()
        word = [words.strip(', . -') for words in word]
        for words in word:
            dictionaryWords.append(words)
    files.close()
    return dictionaryWords

#read the noisy data
def noisy_data():
    tokens = []
    files = open('noisy_data.txt', 'r', encoding ='utf-8')
    for line in files:
        token_lower = line.lower().strip()
        tokenized_word = word_tokenize(token_lower)
        for words in tokenized_word:
            tokens.append(words.strip('\n\n~,.*;/\\(){}<>?@#$%^&_+--+:""'))
    files.close()
    return tokens

def ocr_ERROR(lexicon_words, noisy_words):
    misspelled = []
    for words in noisy_words:
        if words not in lexicon_words:
            misspelled.append(words)
    return misspelled


def check_ERROR(error_check):
    for words in error_check:
        error_words = words
    return error_words
        
#################### Regular Expressions #############################

#spliting the text on delimeter
def split_regx(text):
    text = re.split('[\n\n,.]', text)
    return text

#pre-process the delimeted text
def list_to_string(list_str):
    str1 = " ".join(list_str)
    return str1

def process(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"!", "", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\]", "", string)
    string = re.sub(r"\[", "", string)
    string = re.sub(r"\}", "", string)
    string = re.sub(r"\{", "", string)
    string = re.sub(r"\*", "", string)
    string = re.sub(r"\;", "", string)
    string = re.sub(r"\~", "", string)
    string = re.sub(r"\@", "", string)
    string = re.sub(r"\&", "", string)
    string = re.sub(r"\%", "", string)
    string = re.sub(r"\#", "", string)
    string = re.sub(r"\^", "", string)
    string = re.sub(r"\+", "", string)
    string = re.sub(r"\=", "", string)
    string = re.sub(r"\_", "", string)
    string = re.sub(r"\>", "", string)
    string = re.sub(r"\<", "", string)
    string = re.sub(r"\$", "", string)
    string = re.sub(r"/", "", string)
    
    return string

def regex_err():
    in_file = open("noisy_data.txt", "r", encoding = "utf-8")
    tokens = []
    for lines in in_file:
        token = lines.lower()
        split_text = split_regx(token)
        listString = list_to_string(split_text)
        processed = process(listString)
        tokenized = word_tokenize(processed)
        for words in tokenized:
            tokens.append(words)
    in_file.close()
    return tokens

#################### n-GRAM #######################
def n_gram(noisy, clean):
    tokens1 = list(ngrams(noisy, 3))
    tokens2 = list(ngrams(clean, 3))
    uncommon = []
    for gram in tokens1:
        if gram not in tokens2:
            uncommon.append(gram) 
    return uncommon


################ Real Word Error -- POS tagging and Candidate Generate ###################  
# def nlp_stanza_pos(tokens):
#     #print("In stanza")
#     pos_nlp = stanza.Pipeline('en', processors='tokenize,pos', use_gpu=True, pos_batch_size=5000, tokenize_pretokenized=True)
#     pos_document = pos_nlp((" ").join(tokens))
#     sent_pos = [(word.xpos) for sent in pos_document.sentences for word in sent.words]
#     zipped = list(zip(tokens, sent_pos))
#     #return sent_pos
#     return zipped    
        
        #print(word)
        # Get the one `most likely` answer
        #print(spell.correction(word))

        # Get a list of `likely` options
        #print(spell.candidates(word))

################# Appending tag #####################################
def append_lexicon_tag(read_noisy, read_dict):
    
    df1 = pd.DataFrame(read_noisy)
    df2 = pd.DataFrame(read_dict)
    
    check = pd.Series(df1[0].isin(df2[0]).values.astype(int), df1[0].values).replace([True, False], [0,1])
    check.to_csv("lexicon_error-word.csv", encoding = "utf-8")
    

def append_regex_tag(read_regex, read_dict):
    
    df1 = pd.DataFrame(read_regex)
    df2 = pd.DataFrame(read_dict)
    
    check = pd.Series(df1[0].isin(df2[0]).values.astype(int), df1[0].values).replace([True, False], [0,1])
    check.to_csv("regex_error-word.csv", encoding = "utf-8")

            
def append_realword_tag(misspelled, read_dict):
    
    df1 = pd.DataFrame(misspelled)
    df2 = pd.DataFrame(read_dict)
    
    check = pd.Series(df1[0].isin(df2[0]).values.astype(int), df1[0].values).replace([True, False], [0,1])
    check.to_csv("real_error-word.csv", encoding = "utf-8")
            
    
if __name__ == "__main__":
    read_dict = lexicon() # preprocessed dictionary data
    read_noisy = noisy_data() # preprocessed noisy data
    #print(read_noisy[0])
    read_regex = regex_err() # preprocessed noisy data using regex
    error_check = ocr_ERROR(read_dict, read_noisy) # Non word -- Lexicon Error Check: returning all ocr'd error

    regex_check = ocr_ERROR(read_dict, read_regex) # Non word -- Regular Expression Error Check: returning all ocr'd errors
    
    check_ngram = n_gram(read_noisy, read_dict) #N-Gram check
    #print(check_ngram)
    
    #real-word error check
    spell = SpellChecker()
    misspelled = spell.unknown(read_noisy) # find those words that may be misspelled
    
    lexicon_tag = append_lexicon_tag(read_noisy, read_dict)
    regex_tag = append_regex_tag(read_regex, read_dict)
    realWord_tag = append_realword_tag(misspelled, read_dict)
            
    