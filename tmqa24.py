#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('autosave', '180')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#import convex as cx
import requests
import time
import itertools as it
import re
#import numpy
from copy import copy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#from pprint import pprint
import time
import json
import os
import networkx as nx
from math import sqrt
import spacy
from hdt import HDTDocument
import multiprocessing as mp

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
from deepcorrect import DeepCorrect

#import deepcorrect
#print(deepcorrect.__file__)
corrector = DeepCorrect('data/deep_punct/deeppunct_params_en', 'data/deep_punct/deeppunct_checkpoint_wikipedia')


# In[3]:


#corrector.correct('of what nationality is ken mcgoogan')


# In[4]:


hdt_wd = HDTDocument("data/kb/wikidata2018_09_11.hdt")


# In[5]:


#nlp = spacy.load("en_core_web_lg")
nlp = spacy.load("/data/users/romain.claret/tm/wiki-kb-linked-entities/nlp_custom_6")
#print(nlp.pipeline)


# In[6]:


# load settings
with open( "settings-graphqa.json", "r") as settings_data:
    settings = json.load(settings_data)
    use_cache = settings['use_cache']
    save_cache = settings['save_cache']
    cache_path = settings['cache_path']
#cache_path


# In[7]:


save_cache = True
def save_cache_data():
    if save_cache:
        with open(os.path.join(cache_path,'wd_local_statements_dict.json'), 'wb') as outfile:
            outfile.write(json.dumps(wd_local_statements_dict, separators=(',',':')).encode('utf8'))
        with open(os.path.join(cache_path,'wd_labels_dict.json'), 'wb') as outfile:
            outfile.write(json.dumps(wd_labels_dict, separators=(',',':')).encode('utf8'))
        with open(os.path.join(cache_path,'wd_local_word_ids_dict.json'), 'wb') as outfile:
            outfile.write(json.dumps(wd_local_word_ids_dict, separators=(',',':')).encode('utf8'))
        with open(os.path.join(cache_path,'wd_online_word_ids_dict.json'), 'wb') as outfile:
            outfile.write(json.dumps(wd_online_word_ids_dict, separators=(',',':')).encode('utf8'))
        with open(os.path.join(cache_path,'wd_local_predicate_ids_dict.json'), 'wb') as outfile:
            outfile.write(json.dumps(wd_local_predicate_ids_dict, separators=(',',':')).encode('utf8'))
        with open(os.path.join(cache_path,'wd_online_predicate_ids_dict.json'), 'wb') as outfile:
            outfile.write(json.dumps(wd_online_predicate_ids_dict, separators=(',',':')).encode('utf8'))
        with open(os.path.join(cache_path,'word_similarities_dict.json'), 'wb') as outfile:
            outfile.write(json.dumps(word_similarities_dict, separators=(',',':')).encode('utf8'))


# In[8]:


# Load statements cache
use_cache = True
if use_cache:
    path_wd_local_statements_dict = "wd_local_statements_dict.json"
    path_wd_labels_dict = 'wd_labels_dict.json'
    path_wd_local_word_ids_dict = 'wd_local_word_ids_dict.json'
    path_wd_online_word_ids_dict = 'wd_online_word_ids_dict.json'
    path_wd_local_predicate_ids_dict = 'wd_local_predicate_ids_dict.json'
    path_wd_online_predicate_ids_dict = 'wd_online_predicate_ids_dict.json'
    path_word_similarities_dict = 'word_similarities_dict.json'
else:
    path_wd_local_statements_dict = "wd_local_statements_dict_empty.json"
    path_wd_labels_dict = 'wd_labels_dict_empty.json'
    path_wd_local_word_ids_dict = 'wd_local_word_ids_dict_empty.json'
    path_wd_online_word_ids_dict = 'wd_online_word_ids_dict_empty.json'
    path_wd_local_predicate_ids_dict = 'wd_local_predicate_ids_dict_empty.json'
    path_wd_online_predicate_ids_dict = 'wd_online_predicate_ids_dict_empty.json'
    path_word_similarities_dict = 'word_similarities_dict_empty.json'

with open(os.path.join(cache_path,path_wd_local_statements_dict), "rb") as data:
    wd_local_statements_dict = json.load(data)
with open(os.path.join(cache_path,path_wd_labels_dict), "rb") as data:
    wd_labels_dict = json.load(data)
with open(os.path.join(cache_path,path_wd_local_word_ids_dict), "rb") as data:
    wd_local_word_ids_dict = json.load(data)
with open(os.path.join(cache_path,path_wd_online_word_ids_dict), "rb") as data:
    wd_online_word_ids_dict = json.load(data)
with open(os.path.join(cache_path,path_wd_local_predicate_ids_dict), "rb") as data:
    wd_local_predicate_ids_dict = json.load(data)
with open(os.path.join(cache_path,path_wd_online_predicate_ids_dict), "rb") as data:
    wd_online_predicate_ids_dict = json.load(data)
with open(os.path.join(cache_path,path_word_similarities_dict), "rb") as data:
    word_similarities_dict = json.load(data)

#print("len(wd_local_statements_dict)",len(wd_local_statements_dict))
#print("len(wd_labels_dict)",len(wd_labels_dict))
#print("len(wd_local_word_ids_dict)",len(wd_local_word_ids_dict))
#print("len(wd_online_word_ids_dict)",len(wd_online_word_ids_dict))
#print("len(wd_local_predicate_ids_dict)",len(wd_local_word_ids_dict))
#print("len(wd_online_predicate_ids_dict)",len(wd_online_word_ids_dict))


# In[9]:


def get_kb_ents(text):
    #doc = nlp_kb(text)
    doc = nlp(text)
    #for ent in doc.ents:
    #    print(" ".join(["ent", ent.text, ent.label_, ent.kb_id_]))
    return doc.ents
        
#ent_text_test = (
#    "In The Hitchhiker's Guide to the Galaxy, written by Douglas Adams, "
#    "Douglas reminds us to always bring our towel, even in China or Brazil. "
#    "The main character in Doug's novel is the man Arthur Dent, "
#    "but Dougledydoug doesn't write about George Washington or Homer Simpson."
#)
#
#en_text_test_2 = ("Which actor voiced the Unicorn in The Last Unicorn?")
#
#print([ent.kb_id_ for ent in get_kb_ents(ent_text_test)])
#[ent.kb_id_ for ent in get_kb_ents(en_text_test_2)]


# In[10]:


def get_nlp(sentence, autocorrect=False, banning_str=False):
    
    sentence = sentence.replace("’", "\'")
    nlp_sentence = nlp(sentence)
    nlp_sentence_list = list(nlp_sentence)
    meaningful_punct = []
    
    for i_t, t in enumerate(nlp_sentence_list):
        #print(t,t.pos_, t.lemma_)
        if t.lemma_ == "year":
            nlp_sentence_list[i_t] = "date"
        elif t.text == "\'s" or t.text == "s":
            if t.lemma_ == "be" or t.lemma_ == "s":
                nlp_sentence_list[i_t] = "is"
            else: nlp_sentence_list[i_t] = ""
        elif t.text == "\'ve" or t.text == "ve":
            if t.lemma_ == "have":
                nlp_sentence_list[i_t] = "have"
            else: nlp_sentence_list[i_t] = ""
        elif t.text == "\'re" or t.text == "re":
            if t.lemma_ == "be":
                nlp_sentence_list[i_t] = "are"
            else: nlp_sentence_list[i_t] = ""
        elif t.text == "\'ll" or t.text == "ll":
            if t.lemma_ == "will":
                nlp_sentence_list[i_t] = "will"
            else: nlp_sentence_list[i_t] = ""
        elif t.text == "\'d" or t.text == "d":
            if t.lemma_ == "have":
                nlp_sentence_list[i_t] = "had"
            elif t.lemma_ == "would":
                nlp_sentence_list[i_t] = "would"
            else: nlp_sentence_list[i_t] = ""   
        elif t.is_space:
            nlp_sentence_list[i_t] = ""
        elif t.pos_ == "PUNCT":
            if t.text.count(".") > 2:
                meaningful_punct.append((i_t,"..."))
                nlp_sentence_list[i_t] = "..."
            else:
                nlp_sentence_list[i_t] = ""
        else: nlp_sentence_list[i_t] = nlp_sentence_list[i_t].text
    
    nlp_sentence_list = [w for w in nlp_sentence_list if w]
    
    #print("nlp_sentence_list",nlp_sentence_list)
    
    if autocorrect:
        nlp_sentence = " ".join(nlp_sentence_list)
        nlp_sentence = (nlp_sentence.replace("’", "\'").replace("€", "euro").replace("ç", "c")
                        .replace("à", "a").replace("é","e").replace("ä","a").replace("ö","o")
                       .replace("ü","u").replace("è","e").replace("¨","").replace("ê","e")
                       .replace("â","a").replace("ô","o").replace("î","i").replace("û","u")
                        .replace("_"," ").replace("°","degree").replace("§","section").replace("š","s")
                       .replace("Š","S").replace("ć","c").replace("Ç", "C")
                        .replace("À", "A").replace("É","E").replace("Ä","A").replace("Ö","O")
                       .replace("Ü","U").replace("È","E").replace("Ê","E")
                       .replace("Â","A").replace("Ô","O").replace("Î","I").replace("Û","U")
                        .replace("á","a").replace("Á","Á").replace("ó","o").replace("Ó","O")
                        .replace("ú","u").replace("Ú","U").replace("í","i").replace("Í","I")
                        .replace("–","-").replace("×","x").replace("“","\"")
                       )
        
        if banning_str:
            for ban in banning_str:
                nlp_sentence = nlp_sentence.replace(ban[0],ban[1])
                
        nlp_sentence = corrector.correct(nlp_sentence)
        nlp_sentence = nlp_sentence[0]["sequence"]
    
        nlp_sentence = nlp(nlp_sentence)
        nlp_sentence_list = list(nlp_sentence)

        for i_t, t in enumerate(nlp_sentence_list):
            if t.pos_ == "PUNCT":
                if i_t in [mpunct[0] for mpunct in meaningful_punct]:
                    for mpunct in meaningful_punct:
                        if i_t == mpunct[0]:
                            nlp_sentence_list[mpunct[0]] = mpunct[1]
                else: nlp_sentence_list[i_t] = ''

            else:
                nlp_sentence_list[i_t] = nlp_sentence_list[i_t].text

        for mpunct in meaningful_punct:
            if mpunct[0] < len(nlp_sentence_list):
                if nlp_sentence_list[mpunct[0]] != mpunct[1]:
                    nlp_sentence_list.insert(mpunct[0], mpunct[1])
        
    return nlp(" ".join(nlp_sentence_list).replace("  ", " ").replace(". &",".").replace(". /","."))


#get_nlp("Which genre of album is harder.....faster?", autocorrect=True)
#get_nlp("Which genre of album is Harder ... Faster", autocorrect=True)
#get_nlp("Which home is an example of italianate architecture?", autocorrect=True)
#get_nlp("Your mom's father, were nice in the Years.!?\'\":`’^!$£€\(\)ç*+%&/\\\{\};,àéäöüè¨êâôîû~-_<>°§...@.....", autocorrect=True)
#get_nlp("of what nationality is ken mcgoogan", autocorrect=True)
#get_nlp("you're fun", autocorrect=True)
#get_nlp("where's the fun", autocorrect=True)
#get_nlp("whats the name of the organization that was founded by  frei otto", True)
#get_nlp("Hurry! We’re late!",True)
#get_nlp("Who was an influential figure for miško Šuvaković",True)
#get_nlp("what is the second level division of the division crixás do tocantins",True)


# In[11]:


def is_wd_entity(to_check):
    pattern = re.compile('^Q[0-9]*$')
    if pattern.match(to_check.strip()): return True
    else: return False

def is_wd_predicate(to_check):
    pattern = re.compile('^P[0-9]*$')
    if pattern.match(to_check.strip()): return True
    else: return False
    
def is_valide_wd_id(to_check):
    if is_wd_entity(to_check) or is_wd_predicate(to_check): return True
    else: return False

#print(is_wd_entity("Q155"))


# In[12]:


# TODO redo the functions and optimize

def is_entity_or_literal(to_check):
    if is_wd_entity(to_check.strip()):
        return True
    pattern = re.compile('^[A-Za-z0-9]*$')
    if len(to_check) == 32 and pattern.match(to_check.strip()):
        return False
    return True

# return if the given string is a literal or a date
def is_literal_or_date(to_check): 
    return not('www.wikidata.org' in to_check)

# return if the given string describes a year in the format YYYY
def is_year(year):
    pattern = re.compile('^[0-9][0-9][0-9][0-9]$')
    if not(pattern.match(year.strip())):
        return False
    else:
        return True

# return if the given string is a date
def is_date(date):
    pattern = re.compile('^[0-9]+ [A-z]+ [0-9][0-9][0-9][0-9]$')
    if not(pattern.match(date.strip())):
        return False
    else:
        return True

# return if the given string is a timestamp
def is_timestamp(timestamp):
    pattern = re.compile('^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T00:00:00Z')
    if not(pattern.match(timestamp.strip())):
        return False
    else:
        return True

# convert the given month to a number
def convert_month_to_number(month):
    return{
        "january" : "01",
        "february" : "02",
        "march" : "03",
        "april" : "04",
        "may" : "05",
        "june" : "06",
        "july" : "07",
        "august" : "08",
        "september" : "09", 
        "october" : "10",
        "november" : "11",
        "december" : "12"
    }[month.lower()]

# convert a date from the wikidata frontendstyle to timestamp style
def convert_date_to_timestamp (date):
    sdate = date.split(" ")
    # add the leading zero
    if (len(sdate[0]) < 2):
        sdate[0] = "0" + sdate[0]
    return sdate[2] + '-' + convert_month_to_number(sdate[1]) + '-' + sdate[0] + 'T00:00:00Z'

# convert a year to timestamp style
def convert_year_to_timestamp(year):
    return year + '-01-01T00:00:00Z'

# get the wikidata id of a wikidata url
def wikidata_url_to_wikidata_id(url):
    if not url:
        return False
    if "XMLSchema#dateTime" in url or "XMLSchema#decimal" in url:
        date = url.split("\"", 2)[1]
        date = date.replace("+", "")
        return date
    if(is_literal_or_date(url)):
        if is_year(url):
            return convert_year_to_timestamp(url)
        if is_date(url):
            return convert_date_to_timestamp(url)
        else:
            url = url.replace("\"", "")
            return url
    else:
        url_array = url.split('/')
        # the wikidata id is always in the last component of the id
        return url_array[len(url_array)-1]
    
# fetch all statements where the given qualifier statement occurs as subject
def get_all_statements_with_qualifier_as_subject(qualifier):
    statements = []
    triples, cardinality = hdt_wd.search_triples(qualifier, "", "")
    for triple in triples:
        sub, pre, obj = triple
        # only consider triples with a wikidata-predicate
        if pre.startswith("http://www.wikidata.org/"):
            statements.append({'entity': sub, 'predicate': pre, 'object': obj})
    return statements

# fetch the statement where the given qualifier statement occurs as object
def get_statement_with_qualifier_as_object(qualifier):
    triples, cardinality = hdt_wd.search_triples("", "", qualifier)
    for triple in triples:
        sub, pre, obj = triple
        # only consider triples with a wikidata-predicate
        if pre.startswith("http://www.wikidata.org/") and sub.startswith("http://www.wikidata.org/entity/Q"):
            return (sub, pre, obj)
    return False

# returns all statements that involve the given entity
def get_all_statements_of_entity(entity_id):
    # check entity pattern
    if not is_wd_entity(entity_id.strip()):
        return False
    if wd_local_statements_dict.get(entity_id) != None:
        #print("saved statement")
        return wd_local_statements_dict[entity_id]
    entity = "http://www.wikidata.org/entity/"+entity_id
    statements = []
    # entity as subject
    triples_sub, cardinality_sub = hdt_wd.search_triples(entity, "", "")
    # entity as object
    triples_obj, cardinality_obj = hdt_wd.search_triples("", "", entity)
    if cardinality_sub + cardinality_obj > 5000:
        wd_local_statements_dict[entity_id] = []
        return []
    # iterate through all triples in which the entity occurs as the subject
    for triple in triples_sub:
        sub, pre, obj = triple
        # only consider triples with a wikidata-predicate or if it is an identifier predicate
        if not pre.startswith("http://www.wikidata.org/"):# or (wikidata_url_to_wikidata_id(pre) in identifier_predicates):
            continue
        # object is statement
        if obj.startswith("http://www.wikidata.org/entity/statement/"):
            qualifier_statements = get_all_statements_with_qualifier_as_subject(obj)
            qualifiers = []
            for qualifier_statement in qualifier_statements:
                if qualifier_statement['predicate'] == "http://www.wikidata.org/prop/statement/" + wikidata_url_to_wikidata_id(pre):
                        obj = qualifier_statement['object']
                elif is_entity_or_literal(wikidata_url_to_wikidata_id(qualifier_statement['object'])):
                    qualifiers.append({
                        "qualifier_predicate":{
                            "id": wikidata_url_to_wikidata_id(qualifier_statement['predicate'])
                        }, 
                        "qualifier_object":{
                            "id": wikidata_url_to_wikidata_id(qualifier_statement['object'])
                        }})
            statements.append({'entity': {'id': wikidata_url_to_wikidata_id(sub)}, 'predicate': {'id': wikidata_url_to_wikidata_id(pre)}, 'object': {'id': wikidata_url_to_wikidata_id(obj)}, 'qualifiers': qualifiers})
        else:
            statements.append({'entity': {'id': wikidata_url_to_wikidata_id(sub)}, 'predicate': {'id': wikidata_url_to_wikidata_id(pre)}, 'object': {'id': wikidata_url_to_wikidata_id(obj)}, 'qualifiers': []})
    # iterate through all triples in which the entity occurs as the object
    for triple in triples_obj:
        sub, pre, obj = triple
        # only consider triples with an entity as subject and a wikidata-predicate or if it is an identifier predicate
        if not sub.startswith("http://www.wikidata.org/entity/Q"):# or not pre.startswith("http://www.wikidata.org/") or wikidata_url_to_wikidata_id(pre) in identifier_predicates:
            continue
        if sub.startswith("http://www.wikidata.org/entity/statement/"):
            statements_with_qualifier_as_object =  get_statement_with_qualifier_as_object(sub, process)
            # if no statement was found continue
            if not statements_with_qualifier_as_object:
                continue
            main_sub, main_pred, main_obj = statements_with_qualifier_as_object
            qualifier_statements = get_all_statements_with_qualifier_as_subject(sub)
            qualifiers = []
            for qualifier_statement in qualifier_statements:
                if wikidata_url_to_wikidata_id(qualifier_statement['predicate']) == wikidata_url_to_wikidata_id(main_pred):
                    main_obj = qualifier_statement['object']
                elif is_entity_or_literal(wikidata_url_to_wikidata_id(qualifier_statement['object'])):
                    qualifiers.append({
                        "qualifier_predicate":{"id": wikidata_url_to_wikidata_id(qualifier_statement['predicate'])}, 
                        "qualifier_object":{"id": wikidata_url_to_wikidata_id(qualifier_statement['object'])}
                    })
            statements.append({
                            'entity': {'id': wikidata_url_to_wikidata_id(main_sub)},
                            'predicate': {'id': wikidata_url_to_wikidata_id(main_pred)},
                            'object': {'id': wikidata_url_to_wikidata_id(main_obj)},
                            'qualifiers': qualifiers
                              })
        else:
            statements.append({'entity': {'id': wikidata_url_to_wikidata_id(sub)}, 'predicate': {'id': wikidata_url_to_wikidata_id(pre)}, 'object': {'id': wikidata_url_to_wikidata_id(obj)}, 'qualifiers': []})
    # cache the data
    wd_local_statements_dict[entity_id] = statements
    return statements

#print(len(get_all_statements_of_entity("Q267721")))
#for s in get_all_statements_of_entity("Q267721"):
#    print(s)
#save_cache_data()


# In[13]:


def get_wd_ids_online(name, is_predicate=False, top_k=3):
    name = name.split('(')[0]
    
    if is_predicate and wd_online_predicate_ids_dict.get(name) != None and use_cache and len(wd_online_predicate_ids_dict)>0:
        #print("saved predicate online")
        return wd_online_predicate_ids_dict[name][:top_k]
    elif not is_predicate and wd_online_word_ids_dict.get(name) != None and use_cache and len(wd_online_word_ids_dict)>0:
        #print("saved word online")
        return wd_online_word_ids_dict[name][:top_k]

    request_successfull = False
    entity_ids = ""
    while not request_successfull:
        try:
            if is_predicate:
                entity_ids = requests.get('https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&type=property&limit=' + str(top_k) + '&search='+name).json()
            else:
                entity_ids = requests.get('https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&limit=' + str(top_k) + '&search='+name).json()
            request_successfull = True
        except:
            time.sleep(5)
    results = entity_ids.get("search")
    if not results:
        if is_predicate: wd_online_predicate_ids_dict[name] = [] 
        else: wd_online_word_ids_dict[name] = [] 
        return []
    if not len(results):
        if is_predicate: wd_online_predicate_ids_dict[name] = [] 
        else: wd_online_word_ids_dict[name] = []
        return [] 
    res = []
    for result in results:
        res.append(result['id'])
    
    if is_predicate: wd_online_predicate_ids_dict[name] = res
    else: wd_online_word_ids_dict[name] = res
    
    if res:
        return res[:top_k]
    else:
        return []

#print(get_wd_ids_online("be", is_predicate=False, top_k=1))


# In[14]:


# very computational
def get_most_similar(word, top_k=3):
    print("behold: get_most_similar started with:", word)
    word_text = str(word.lower())
    if word_similarities_dict.get(word) != None and use_cache and len(word_similarities_dict)>0:
        return word_similarities_dict[word][:top_k]
    
    word = nlp.vocab[word_text]
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    
    word_similarities = [(w.text.lower(),float(w.similarity(word))) for w in by_similarity[:10] if w.lower_ != word.lower_]

    word_similarities_dict[word_text] = word_similarities
    
    save_cache_data()
    
    return word_similarities[:top_k]

#print(get_most_similar("voiced", top_k=3))
#save_cache_data()


# In[15]:


def get_wd_ids(word, is_predicate=False, top_k=3, limit=6, online=False):
    if is_predicate and wd_local_predicate_ids_dict.get(word) != None and use_cache and len(wd_local_predicate_ids_dict)>0:
        #print("saved predicate local")
        return wd_local_predicate_ids_dict[word][:top_k]
    elif not is_predicate and wd_local_word_ids_dict.get(word) != None and use_cache and len(wd_local_word_ids_dict)>0:
        #print("saved word local")
        return wd_local_word_ids_dict[word][:top_k]
    
    language = "en"
    word_formated = str("\""+word+"\""+"@"+language)
    to_remove = len("http://www.wikidata.org/entity/")
    t_name, card_name = hdt_wd.search_triples("", "http://schema.org/name", word_formated, limit=top_k)
    #print("names cardinality of \"" + word+"\": %i" % card_name)
    t_alt, card_alt = hdt_wd.search_triples("", 'http://www.w3.org/2004/02/skos/core#altLabel', word_formated, limit=top_k)
    #print("alternative names cardinality of \"" + word+"\": %i" % card_alt)
    results = list(set(
        [t[0][to_remove:] for t in t_name if is_valide_wd_id(t[0][to_remove:])] + 
        [t[0][to_remove:] for t in t_alt if is_valide_wd_id(t[0][to_remove:])]
           ))
    
    if is_predicate: results = [r for r in results if is_wd_predicate(r)]
        
    # cache the data
    if is_predicate: wd_local_predicate_ids_dict[word] = results
    else: wd_local_word_ids_dict[word] = results
    
    return results if limit<=0 else results[:limit-1]
     
#print(get_wd_ids("", is_predicate=False, top_k=1))
#get_wd_ids("The Last Unicorn", top_k=0, limit=10)
#print(get_wd_ids("wife", is_predicate=False , top_k=0, limit=0))
#print(get_wd_ids("voiced", is_predicate=False , top_k=0, limit=0))


# In[16]:


def get_wd_label(from_id, language="en"):
    #print("from_id",from_id)
    if is_valide_wd_id(from_id):
        if wd_labels_dict.get(from_id) != None and use_cache and len(wd_labels_dict)>0:
            #print("saved label local")
            return wd_labels_dict[from_id]
        
        id_url = "http://www.wikidata.org/entity/"+from_id
        t_name, card_name = hdt_wd.search_triples(id_url, "http://schema.org/name", "")
        name = [t[2].split('\"@'+language)[0].replace("\"", "") for t in t_name if "@"+language in t[2]]
        #name = [t[2].split('@en')[0] for t in t_name if "@"+language in t[2]]
        result = name[0] if name else ''
        wd_labels_dict[from_id] = result #caching
        return result
        
    else:
        return from_id
    
#print(get_wd_label("P725"))
#get_wd_label("Q20789322")
#get_wd_label("Q267721")


# In[17]:


# Building colors from graph
def get_color(node_type):
    if node_type == "entity": return "violet"#"cornflowerblue"
    elif node_type == "predicate": return "yellow"
    else: return "red"

# Building labels for graph
def get_elements_from_graph(graph):
    node_names = nx.get_node_attributes(graph,"name")
    node_types = nx.get_node_attributes(graph,"type")
    colors = [get_color(node_types[n]) for n in node_names]
    return node_names, colors

# Plotting the graph
def plot_graph(graph, name, title="Graph"):
    fig = plt.figure(figsize=(14,14))
    ax = plt.subplot(111)
    ax.set_title(str("answer: "+title), fontsize=10)
    #pos = nx.spring_layout(graph)
    labels, colors = get_elements_from_graph(graph)
    nx.draw(graph, node_size=30, node_color=colors, font_size=10, font_weight='bold', with_labels=True, labels=labels)
    plt.tight_layout()
    plt.savefig("tmqa1_graphs_imgs/"+str(name)+".png", format="PNG", dpi = 300)
    plt.show()
    
#plot_graph(graph, "file_name_graph", "Graph_title")


# In[18]:


def make_statements_graph_worker(graph, predicate_nodes, turn, indexing_predicates, BANNED_WD_IDS, BANNED_WD_PRED_IDS, BANNED_WD_KEYWORDS, BANNED_WD_PRED_KEYWORDS, in_mp_queue, out_mp_queue, predicate_nodes_lock):
    #for statement in statements:
    sentinel = None
    
    for statement in iter(in_mp_queue.get, sentinel):
        #print("statement",statement)
        #if (statement['entity']['id'][0] != "Q"
        #    or statement['entity']['id'] in BANNED_WD_IDS
        #    or statement['predicate']['id'][0] != "P"
        #    or statement['predicate']['id'] in BANNED_WD_PRED_IDS
        #    or statement['object']['id'][0] != "Q"
        #    or statement['object']['id'] in BANNED_WD_IDS):
        #    continue
            
        if (
            statement['entity']['id'] in BANNED_WD_IDS
            or statement['predicate']['id'][0] != "P"
            or statement['predicate']['id'] in BANNED_WD_PRED_IDS
            or statement['object']['id'] in BANNED_WD_IDS
            ):
            continue
        
        continue_flag = False
        for key in BANNED_WD_PRED_KEYWORDS:
            if (get_wd_label(statement['predicate']['id']).find(key) != -1): continue_flag = True
                
        for key in BANNED_WD_KEYWORDS:
            if (get_wd_label(statement['entity']['id']).find(key) != -1): continue_flag = True
            if (get_wd_label(statement['object']['id']).find(key) != -1): continue_flag = True
                
        if continue_flag: continue
        
        #print(statement)
        if not statement['entity']['id'] in graph:
            graph.add_node(statement['entity']['id'], name=get_wd_label(statement['entity']['id']), type='entity', turn=turn)
        if not statement['object']['id'] in graph:
            graph.add_node(statement['object']['id'], name=get_wd_label(statement['object']['id']), type='entity', turn=turn)
        
        with predicate_nodes_lock:
            # increment index of predicate or set it at 0
            if not statement['predicate']['id'] in predicate_nodes or not indexing_predicates:
                predicate_nodes_index = 1
                predicate_nodes[statement['predicate']['id']] = 1
            else:
                predicate_nodes[statement['predicate']['id']] += 1
                predicate_nodes_index = predicate_nodes[statement['predicate']['id']]

        # add the predicate node
        predicate_node_id = (statement['predicate']['id'])
        if indexing_predicates: predicate_node_id += "-" + str(predicate_nodes_index)
        
        graph.add_node(predicate_node_id, name=get_wd_label(statement['predicate']['id']), type='predicate', turn=turn)

        # add the two edges (entity->predicate->object)
        #statement['entity']['id'] in BANNED_WD_IDS 
        #statement['object']['id'] in BANNED_WD_IDS
        #statement['predicate']['id'] in BANNED_WD_PRED_IDS
        #if (statement['predicate']['id'] in BANNED_WD_PRED_IDS): break
            
        graph.add_edge(statement['entity']['id'], predicate_node_id)
        graph.add_edge(predicate_node_id, statement['object']['id'])
        
    out_mp_queue.put(graph)


# In[19]:


# TODO: handle special literals? which one

def make_statements_graph(statements, indexing_predicates=True, cores=mp.cpu_count()):
    BANNED_WD_IDS = [
        "Q4167410","Q66087861","Q65932995","Q21281405","Q17442446","Q41770487","Q29548341",
        "Q29547399","Q25670"
    ]
    BANNED_WD_PRED_IDS = [
        "P1687","P7087","P1889","P646", "P227", "P1256", "P1257", "P1258", "P1260", "P301",
        "P18","P1266","P487","P1970","P2529", "P4390", "P4342", "P4213", "P487", "P2624",
        "P4953", "P2241", "P345","P703", "P2163", "P18", "P436", "P227", "P646", "P2581",
        "P1006", "P244", "P214", "P1051", "P1296","P461", "P2959", "P1657", "P3834","P243",
        "P3306","P6932","P356","P1630","P3303","P1921","P1793","P1628","P1184","P1662","P2704",
        "P4793","P1921","P2302","P6562","P6127","P4342","P6145","P5786","P5099","P4947","P5032",
        "P4933","P4632","P4529","P4277","P4282","P3135","P4276","P3593","P2638","P3804","P3145",
        "P2509","P3212","P2704","P480","P3844","P3141","P3808","P3933","P2346","P3077","P3417",
        "P2529","P3302","P3143","P2334","P3129","P3138","P3107","P2603","P2631","P2508","P2465",
        "P2014", "P1874", "P2518", "P1265", "P1237","P1712", "P1970","P1804","P905","P1562",
        "P1258","P646","P345"
    ]
    BANNED_WD_KEYWORDS = []
    BANNED_WD_PRED_KEYWORDS = [
        "ID", "ISBN","identifier", "IDENTIFIER", "isbn", "ISSN", "issn"
    ]
    
    graph = nx.Graph()
    turn=0
    
    predicate_nodes = mp.Manager().dict()
    predicate_nodes_lock = mp.Manager().Lock()
    
    paths_keyword_nodes = []
    if cores <= 0: cores = 1

    out_mp_queue = mp.Queue()
    in_mp_queue = mp.Queue()
    sentinel = None

    for statement in statements:
        in_mp_queue.put(statement)

    procs = [mp.Process(target = make_statements_graph_worker, args = (graph, predicate_nodes, turn, indexing_predicates, BANNED_WD_IDS, BANNED_WD_PRED_IDS, BANNED_WD_KEYWORDS, BANNED_WD_PRED_KEYWORDS, in_mp_queue, out_mp_queue, predicate_nodes_lock)) for i in range(cores)]

    for proc in procs:
        proc.daemon = True
        proc.start()
    for proc in procs:    
        in_mp_queue.put(sentinel)
    for proc in procs:
        local_g = out_mp_queue.get()
        graph = nx.compose(graph,local_g)
    for proc in procs:
        proc.join()
    
    return graph, predicate_nodes

#test_graph = make_statements_graph(test_unduplicate_statements, indexing_predicates=False)
#print(test_graph[1])
#plot_graph(test_graph[0],"test")
#print("len(filtered_statements)",len(filtered_statements))
#start_time = time.time()
#graph, predicate_nodes = make_statements_graph(filtered_statements, indexing_predicates=True, cores=1)
#print(time.time()-start_time)
#print("--> ",len(graph), "nodes and", graph.size(), "edges")
#print(predicate_nodes)


# In[20]:


def merge_lists(list_1, list_2):
    if len(list_1) == len(list_2):
        return [(list_1[i], list_2[i]) for i in range(0, len(list_1))]
    else:
        return "Error: lists are not the same lenght"

#print(merge_lists(["author"],['P50']))


# In[21]:


def get_themes_ids_from_chunks(noun_chunks, top_k=3, online=False):
    if online:
        theme_ids = [get_wd_ids_online(chunk.text, top_k=top_k)+get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks]
    else:
        theme_ids = [get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks]
    return theme_ids

def get_themes(nlp_question, raw_question, top_k=3, online=False, max_title_size=5):
    #nlp_raw_question = get_nlp(raw_question, autocorrect=False)
    #nlp_raw_question_lower = get_nlp(raw_question.lower(), autocorrect=False)
    #nlp_raw_question_captialize = get_nlp(" ".join([w.capitalize() for w in raw_question.split(" ")]), autocorrect=False)
    
    #nlp_raw_question_list = list(nlp_raw_question)
    #print([e for e in nlp_raw_question])
    #print([e for e in nlp_raw_question_lower])
    #print([e for e in nlp_raw_question_captialize])
    
    special_words = [w for w in raw_question.lower().split(" ") if w not in  nlp_question.text.lower().split()]
    special_words_capitalize = [w.capitalize() for w in special_words]
    #special_words_capitalize_per_2 = it.permutations(special_words_capitalize,2)
    
    #print("special_words",special_words)
    #print("special_words_capitalize",special_words_capitalize)
    
    special_words_titles = []
    for per_size in range(2,len(special_words_capitalize)+1):
        if per_size > max_title_size:
            break
        special_words_titles += [" ".join(p) for p in it.permutations(special_words_capitalize,per_size)]

    #print("special_words_titles",special_words_titles)
    
    #print(nlp_raw_question_list)
    
    # PART1: finding themes as the user typed it
    filter_list = ["PART", "PRON", "NUM"]
    nlp_list_src = list(nlp_question)
    nlp_list = []
    for w in nlp_question:
        if w.pos_ not in filter_list:
            nlp_list.append(w)
    nlp_question = get_nlp(" ".join([e.text for e in nlp_list]))
    
    themes = [(ent, [ent.kb_id_]) for ent in get_kb_ents(nlp_question.text) if ent.kb_id_ != "NIL"]
    #print("1 themes",themes)
    for w in special_words+special_words_capitalize+special_words_titles:
        themes += [(ent, [ent.kb_id_]) for ent in get_kb_ents(w) if ent.kb_id_ != "NIL"]
    #print("2 themes",themes)
    theme_complements = []
    
    noun_chunks = [chunk for chunk in nlp_question.noun_chunks]
    
    #print("1 noun_chunks",noun_chunks)
    for w in special_words+special_words_capitalize+special_words_titles:
        noun_chunks += [chunk for chunk in get_nlp(w, autocorrect=False).noun_chunks]
    
    #print("2 noun_chunks",noun_chunks)
    #theme_ids = [get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks][:top_k]
    theme_ids = get_themes_ids_from_chunks(noun_chunks, top_k=3, online=online)
            
    for i, chunk in enumerate(theme_ids):
        if chunk: themes.append((noun_chunks[i], chunk))
        else: theme_complements.append(noun_chunks[i])
    
    # PART2: finding themes with the question capitalized
    #print(nlp_question)
    nlp_list_cap = []
    nlp_list_low = []
    nlp_list_lemma = []
    nlp_list_no_det = []
    w_filter = ["WDT","WP","WP$","WRB"]
    for w in nlp_question:
        if w.tag_ not in w_filter:
            nlp_list_cap.append(w.text.capitalize())
            nlp_list_low.append(w.text.lower())
            nlp_list_lemma.append(w.lemma_)
        if w.pos_ != "DET":
            nlp_list_no_det.append(w.text)
            
    nlp_question_cap = get_nlp(" ".join([e for e in nlp_list_cap]))
    nlp_question_low = get_nlp(" ".join([e for e in nlp_list_low]))
    nlp_question_lemma = get_nlp(" ".join([e for e in nlp_list_lemma]))
    nlp_question_no_det = get_nlp(" ".join([e for e in nlp_list_no_det]))

    themes += [(ent, [ent.kb_id_]) for ent in get_kb_ents(nlp_question_cap.text) if ent.kb_id_ != "NIL" and (ent, [ent.kb_id_]) not in themes]
    themes += [(ent, [ent.kb_id_]) for ent in get_kb_ents(nlp_question_low.text) if ent.kb_id_ != "NIL" and (ent, [ent.kb_id_]) not in themes]
    themes += [(ent, [ent.kb_id_]) for ent in get_kb_ents(nlp_question_lemma.text) if ent.kb_id_ != "NIL" and (ent, [ent.kb_id_]) not in themes]
    themes += [(ent, [ent.kb_id_]) for ent in get_kb_ents(nlp_question_no_det.text) if ent.kb_id_ != "NIL" and (ent, [ent.kb_id_]) not in themes]
    
    if online:
        themes += [(ent, get_wd_ids_online(ent.text, is_predicate=False, top_k=top_k)) for ent in get_kb_ents(nlp_question_cap.text)]
        themes += [(ent, get_wd_ids_online(ent.text, is_predicate=False, top_k=top_k)) for ent in get_kb_ents(nlp_question_low.text)]
        themes += [(ent, get_wd_ids_online(ent.text, is_predicate=False, top_k=top_k)) for ent in get_kb_ents(nlp_question_lemma.text)]
        themes += [(ent, get_wd_ids_online(ent.text, is_predicate=False, top_k=top_k)) for ent in get_kb_ents(nlp_question_no_det.text)]
    
    noun_chunks = []
    
    previous_title_position = 0
    for i_t,t in enumerate(nlp_question):
        tmp_row = []
        if i_t > previous_title_position:
            if t.is_title:
                for i_p in range(previous_title_position,i_t+1):
                    tmp_row.append(nlp_question[i_p])

                noun_chunks.append(get_nlp(" ".join([w.text for w in tmp_row])))

        if t.is_title:
            previous_title_position = i_t
    
    noun_chunks += [chunk for chunk in nlp_question_cap.noun_chunks]
    
    #theme_ids = [get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks][:top_k]
    theme_ids = get_themes_ids_from_chunks(noun_chunks, top_k=3, online=online)
    
    for i, chunk in enumerate(theme_ids):
        if chunk: themes.append((noun_chunks[i], chunk))
        else: theme_complements.append(noun_chunks[i])
    
    noun_chunks = [chunk for chunk in nlp_question_low.noun_chunks]
    #theme_ids = [get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks][:top_k]
    theme_ids = get_themes_ids_from_chunks(noun_chunks, top_k=3, online=online)

    for i, chunk in enumerate(theme_ids):
        if chunk: themes.append((noun_chunks[i], chunk))
        else: theme_complements.append(noun_chunks[i])
            
    noun_chunks = [chunk for chunk in nlp_question_lemma.noun_chunks]
    #theme_ids = [get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks][:top_k]
    theme_ids = get_themes_ids_from_chunks(noun_chunks, top_k=3, online=online)

    for i, chunk in enumerate(theme_ids):
        if chunk: themes.append((noun_chunks[i], chunk))
        else: theme_complements.append(noun_chunks[i])
            
    noun_chunks = [chunk for chunk in nlp_question_no_det.noun_chunks]
    #theme_ids = [get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks][:top_k]
    theme_ids = get_themes_ids_from_chunks(noun_chunks, top_k=3, online=online)

    for i, chunk in enumerate(theme_ids):
        if chunk: themes.append((noun_chunks[i], chunk))
        else: theme_complements.append(noun_chunks[i])
    
    themes_filtered = []
    for t in themes:
        if t[0].text in [tf[0].text for tf in themes_filtered]:
            index = [tf[0].text for tf in themes_filtered].index(t[0].text)
            tmp = t[1]+[i for j in [tf[1] for index, tf in enumerate(themes_filtered) if tf[0].text == t[0].text] for i in j]
            themes_filtered[index] = (t[0],tmp)

        else:
            themes_filtered.append(t)
    
    # removing the same elments per rows and skipping already existing rows
    unique_ids = []
    themes_filtered_undupped = []
    for tf in themes_filtered:
        tmp_ids = []
        for tfid in tf[1]:
            if tfid not in unique_ids and tfid not in tmp_ids:
                tfname = get_wd_label(tfid)
                similarity = get_nlp(tfname).similarity(tf[0])
                if similarity >= 0.95:
                    tmp_ids.append(tfid)
                unique_ids.append(tfid)
        if tmp_ids and tmp_ids not in [tfu[1] for tfu in themes_filtered_undupped]:
            themes_filtered_undupped.append((tf[0],tmp_ids))
    
    #for tf in themes_filtered:
    #    tmp_ids = []
    #    for tfid in tf[1]:
    #        if tfid not in tmp_ids:
    #            tmp_ids.append(tfid)
    #    if tmp_ids not in [tfu[1] for tfu in themes_filtered_undupped]:
    #        themes_filtered_undupped.append((tf[0],tmp_ids))

        
    theme_complements_undupped = []
    [theme_complements_undupped.append(tc) for tc in theme_complements if tc.text not in [tcu.text for tcu in theme_complements_undupped]]
    
    
    #print(themes_filtered)
    return themes_filtered_undupped, theme_complements_undupped

#q0_themes = get_themes(q0_nlp, top_k=3)
#q0_themes_test = get_themes(q0_nlp_test)
#q0_themes_test_2 = get_themes(q0_nlp_test_2)
#print(q0_themes) 

#q_test_3 = get_nlp("the unicorn and the raccoons love obama barack's tacos")
#q_test_3_themes = get_themes(q_test_3, top_k=3)
#print(get_enhanced_themes(q_test_3_themes))
#print(q_test_3_themes)


#q_test_test = get_nlp("What is a tv action show?")
#q_test_test = get_nlp("Who voiced the Unicorn in The Last Unicorn")
#q_test_test = get_nlp("What is the name of the person who created Saved by the Bell?")
#q_test_test = get_nlp("When did the movie Grease come out?")
#q_test_question = "Who was an influential figure for miško Šuvaković"
#q_test_test = get_nlp(q_test_question,True)
#get_themes(q_test_test, q_test_question, top_k=3, online=True)


# In[22]:


BANNED_WORDS = ["..."]

def get_theme_tuples(theme_list, top_k=3, online=False):
    tuples = [(t, get_wd_ids(t, top_k=top_k)) for t in theme_list if t not in BANNED_WORDS]
    if online:
        tuples += [(t, get_wd_ids_online(t, is_predicate=False, top_k=top_k)) for t in theme_list if t not in BANNED_WORDS]
    return tuples

def get_theme_no_stopwords(theme_list):
    return [s for s in theme_list if not s.is_stop]

def get_theme_lemmatized(theme_list):
    return [s.lemma_ for s in theme_list]

def get_permutation_tuples(theme_list, start=2):   
    permutations = []
    for i in range(start, len(theme_list)+1):
        permutations += list(it.permutations(theme_list,i))
    return permutations

def get_lemma_permutation_tuples(theme_list, start=2):
    return get_permutation_tuples(get_theme_lemmatized(theme_list), start=2)

def get_non_token_tuples(theme_list):
    return [" ".join([e for e in list(l)]) for l in theme_list]

def get_non_token_lower_tuples(theme_list):
    return [" ".join([e.lower() for e in list(l)]) for l in theme_list]

def get_non_token_capitalize_tuples(theme_list):
    return [" ".join([c.capitalize() for c in [e for e in list(l)]]) for l in theme_list] 

def get_text_tuples(theme_list):
    return [" ".join([e.text for e in list(l)]) for l in theme_list]

def get_lower_tuples(theme_list):
    return [" ".join([e.lower_ for e in list(l)]) for l in theme_list]

def get_capitalized_tuples(theme_list):
    return [" ".join([c.capitalize() for c in [e.text for e in list(l)]]) for l in theme_list]

def get_enhanced_themes(themes, top_k=3, title_limit=5, aggressive=False, online=False):
    enhanced_themes = []
    # permute, capitalize, lowering of the words in the complements
    for c in themes[1]:
        if len(c) <= title_limit:
            per_lemma = get_theme_tuples(get_non_token_tuples([n for n in get_permutation_tuples(get_theme_lemmatized(c))]),top_k, online=online)
            [enhanced_themes.append(p) for p in per_lemma if p[1] and p not in enhanced_themes]
            del per_lemma

            per_nostop = get_theme_tuples(get_text_tuples(get_permutation_tuples(get_theme_no_stopwords(c),start=1)),top_k, online=online)
            [enhanced_themes.append(p) for p in per_nostop if p[1] and p not in enhanced_themes]
            del per_nostop

            per_lemma_nostop = get_theme_tuples(get_non_token_tuples([get_theme_lemmatized(s) for s in get_permutation_tuples(get_theme_no_stopwords(c),start=1)]),top_k, online=online)
            [enhanced_themes.append(p) for p in per_lemma_nostop if p[1] and p not in enhanced_themes]
            del per_lemma_nostop

            per_lemma_lower = get_theme_tuples(get_non_token_lower_tuples([n for n in get_permutation_tuples(get_theme_lemmatized(c))]),top_k, online=online)
            [enhanced_themes.append(p) for p in per_lemma_lower if p[1] and p not in enhanced_themes]
            del per_lemma_lower

            per_nostop_lower = get_theme_tuples(get_lower_tuples(get_permutation_tuples(get_theme_no_stopwords(c),start=1)),top_k)
            [enhanced_themes.append(p) for p in per_nostop_lower if p[1] and p not in enhanced_themes]
            del per_nostop_lower

            per_lemma_nostop_lower = get_theme_tuples(get_non_token_lower_tuples([get_theme_lemmatized(s) for s in get_permutation_tuples(get_theme_no_stopwords(c),start=1)]),top_k, online=online)
            [enhanced_themes.append(p) for p in per_lemma_nostop_lower if p[1] and p not in enhanced_themes]
            del per_lemma_nostop_lower

            per_lemma_capitalize = get_theme_tuples(get_non_token_capitalize_tuples([n for n in get_permutation_tuples(get_theme_lemmatized(c))]),top_k, online=online)
            [enhanced_themes.append(p) for p in per_lemma_capitalize if p[1] and p not in enhanced_themes]
            del per_lemma_capitalize

            per_nostop_capitalize = get_theme_tuples(get_capitalized_tuples(get_permutation_tuples(get_theme_no_stopwords(c),start=1)),top_k, online=online)
            [enhanced_themes.append(p) for p in per_nostop_capitalize if p[1] and p not in enhanced_themes]
            del per_nostop_capitalize

            per_lemma_nostop_capitalize = get_theme_tuples(get_non_token_capitalize_tuples([get_theme_lemmatized(s) for s in get_permutation_tuples(get_theme_no_stopwords(c),start=1)]),top_k, online=online)
            [enhanced_themes.append(p) for p in per_lemma_nostop_capitalize if p[1] and p not in enhanced_themes]

            per = get_theme_tuples(get_text_tuples(get_permutation_tuples(c)),top_k, online=online)
            [enhanced_themes.append(p) for p in per if p[1] and p not in enhanced_themes]
            del per

            per_lower = get_theme_tuples(get_lower_tuples(get_permutation_tuples(c)),top_k, online=online)
            [enhanced_themes.append(p) for p in per_lower if p[1] and p not in enhanced_themes]
            del per_lower

            per_capitalize = get_theme_tuples(get_capitalized_tuples(get_permutation_tuples(c)),top_k, online=online)
            [enhanced_themes.append(p) for p in per_capitalize if p[1] and p not in enhanced_themes]
            del per_capitalize
    
    if aggressive:
        predicates = []
        [predicates.append(get_wd_label(pred)) for pred in sum([p[1] for p in themes[0]],[]) if get_wd_label(pred) not in predicates]
        predicates_ids = [get_wd_ids_online(p, is_predicate=True, top_k=top_k) for p in predicates]
        predicated_themes = merge_lists(predicates, predicates_ids)
        predicated_themes = [pt for pt in predicated_themes if pt[1] != '']
        if predicates: enhanced_themes += predicated_themes
    
    #print("themes[0]",[t[0].text for t in themes[0]])
    #print("themes[0].lower()",[t[0].text.lower() for t in themes[0]])
    
    enhanced_themes_filtered = []
    for et in enhanced_themes:
        if not et[0] in [t[0].text for t in themes[0]]:
            #print("et not in themes",et)
            #print(len(themes[0]))
            #print([t[0].text.find(et[0]) for t in themes[0]].count(-1))
            if len([t for t in themes[0] if t[0].text.find(et[0]) == -1]) < len(themes[0]) or not et[1]:
                continue
            
            if et[0] in [e[0] for e in enhanced_themes_filtered]:
                index_et = [e[0] for e in enhanced_themes_filtered].index(et[0])
                if index_et != -1:
                    enhanced_themes_filtered[index_et] = (et[0], enhanced_themes_filtered[index_et][1]+et[1]) #.append((et[0],et[1][:top_k]))
                else: enhanced_themes_filtered.append((et[0],et[1][:top_k]))
            #elif et[0] not in :
            #    print("et unknown",et)
            else:
                enhanced_themes_filtered.append((et[0],et[1][:top_k]))
    
    return enhanced_themes_filtered

#q_test_3 = get_nlp("Which genre of album is harder.....faster?",autocorrect=True)
#q_test_3 = get_nlp("the unicorn and the raccoons love obama barack's tacos")
#q_test_3 = get_nlp("what was the cause of death of yves klein")
#q_test_3 = get_nlp("Who is the author that wrote the book Moby Dick")
#q_test_3_themes = get_themes(q_test_3, top_k=3)
#print(q_test_3_themes[0])
#print(get_enhanced_themes(q_test_3_themes, aggressive=False))

#print(get_enhanced_themes(q_test_3_themes, aggressive=True))


# In[23]:


def get_predicates_online(nlp_sentence, top_k=3, aggressive=False):
    PASSIVE_VERBS = ["be"]
    AGRESSIVE_FILTER = ["VERB","AUX","NOUN","ADJ"]
    if aggressive: predicates = [p for p in nlp_sentence if p.pos_ in AGRESSIVE_FILTER]
    else: predicates = [p for p in nlp_sentence if p.pos_ == "VERB" or p.pos_ == "AUX"]

    if len(predicates) == 1:
        if predicates[0].lemma_ in PASSIVE_VERBS:
            predicates += [p for p in nlp_sentence if p.pos_ in AGRESSIVE_FILTER if p not in predicates]
    
    predicates_filtered = []
    for p in predicates:
        if p.lemma_ in PASSIVE_VERBS: 
            p = get_nlp(p.lemma_)[0]
        if len(predicates_filtered) == 0:
            predicates_filtered.append(p)
        if p.text not in [p.text for p in predicates_filtered]:
            predicates_filtered.append(p)
    
    predicates_ids = []
    for i_p, p in enumerate(predicates_filtered):
        if p.lemma_ == "be":
            predicates_ids.append(get_wd_ids_online("is", is_predicate=True, top_k=top_k)[:1])
        else:
            p_id = get_wd_ids_online(p.text, is_predicate=True, top_k=top_k)
            if not p_id:
                p_id = get_wd_ids_online(p.lemma_, is_predicate=True, top_k=top_k)
                if not p_id:
                    similar_words = [w[0] for w in get_most_similar(p.lemma_, top_k=top_k)]
                    for sw in similar_words:
                        if not p_id:
                            p_id = get_wd_ids_online(sw, is_predicate=True, top_k=top_k)
            predicates_ids.append(p_id[:top_k])
    
    return merge_lists(predicates_filtered, predicates_ids)

#q_test = get_nlp("Who voiced the Unicorn in The Last Unicorn")
#q_test = get_nlp("Of what nationality is Ken McGoogan")
#q_test = get_nlp("Which have the nation of Martha Mattox")
#q_test = get_nlp("what city was alex golfis born in")
#q_test = get_nlp("who's born in city was alex golfis born in")
#q_test = get_nlp("what's the name fo the wife of my dads")
#start_time = time.time()
#q_test = get_nlp("Where did roger marquis die")
#print(get_predicates_online(q_test, top_k=2, aggressive=False))
#print("it was:",time.time()-start_time)
#q0_predicates_test_2 = get_predicates_online(q0_nlp_test_2, top_k=3, aggressive=True)


# In[24]:


def get_predicates(nlp_sentence, themes=False, top_k=0):
    PASSIVE_VERBS = ["be"]
    predicates = [p for p in nlp_sentence if p.pos_ == "VERB" or p.pos_ == "AUX"]
    #for i_p, p in enumerate(predicates):
    #    if p.text == "\'s":
    #        predicates[i_p] = get_nlp("is")[0]
    #    if p.text == "\'re":
    #        predicates[i_p] = get_nlp("are")[0]
            
    if themes:
        for t in themes[0]:
            for e in t[1]:
                if is_wd_predicate(e):
                    predicates.append(t[0])
    
    predicates_filtered = []
    for p in predicates:
        if p.lemma_ in PASSIVE_VERBS: 
            p = get_nlp(p.lemma_)[0]
        if len(predicates_filtered) == 0:
            predicates_filtered.append(p)
        if p.text not in [p.text for p in predicates_filtered]:
            predicates_filtered.append(p)
            
    predicates_ids = []
    for i_p, p in enumerate(predicates_filtered):
        if p.lemma_ in PASSIVE_VERBS: 
            predicates_ids.append(get_wd_ids(p.lemma_, is_predicate=True, top_k=top_k, limit=0)[:1])
        else:
            predicates_ids.append(get_wd_ids(p.text, is_predicate=True, top_k=top_k, limit=0)[:top_k])
                
    #predicates_ids = [ for p in predicates_filtered]
    return merge_lists(predicates_filtered, predicates_ids)

#q_test = get_nlp("Who voiced the Unicorn in The Last Unicorn")
#q_test = get_nlp("Of what nationality is Ken McGoogan")
#q_test = get_nlp("Where did roger marquis die")
#q_test = get_nlp("who's born in city was alex golfis born in")
#get_predicates(q_test)
#q_test_themes = get_themes(q_test)
#get_predicates(q_test, q_test_themes, top_k=3)
#q0_nlp_test_0 = get_nlp("Voiced")
#q0_predicates = get_predicates(q0_nlp, top_k=3)
#q0_predicates_test_2 = get_predicates(q0_nlp_test_2, top_k=3)
#print(q0_predicates)


# In[25]:


def extract_ids(to_extract):
    return [i for i in it.chain.from_iterable([id[1] for id in to_extract])]
#extract_ids([('name', ['id'])]) #q0_themes[0] #q0_focused_parts #q0_predicates
#print(extract_ids([("The Last Unicorn", ['Q16614390']),("Second Theme", ['Q12345'])]))
#extract_ids(q0_focused_parts)


# In[26]:


def get_similarity_by_words(nlp_word_from, nlp_word_to):
    if not nlp_word_from or not nlp_word_to:
        return 0
    elif not nlp_word_from.vector_norm or not nlp_word_to.vector_norm:
        return 0
    else:
        return nlp_word_from.similarity(nlp_word_to)

#print(get_similarity_by_words(get_nlp("character role"), get_nlp("voice actor")))


# In[27]:


def get_similarity_by_ids(word_id_from, word_id_to):
    nlp_word_from = get_nlp(get_wd_label(word_id_from))
    nlp_word_to = get_nlp(get_wd_label(word_id_to))
    return get_similarity_by_words(nlp_word_from, nlp_word_to)

#print(get_similarity_by_ids("P453", "P725"))


# In[28]:


def get_top_similar_statements(statements, from_token_id, similar_to_name, top_k=0, qualifier=False, statement_type="object", time_sentitive=False):
    highest_matching_similarity = -1
    top_statements = []
    nlp_name = get_nlp(similar_to_name)
    
    #print("get_top_similar_statements from_token_id",from_token_id)
    if get_wd_label(from_token_id):
        for statement in statements:
            if top_k>0:
                if qualifier:
                    for qualifier in statement['qualifiers']:
                        if time_sentitive and is_timestamp(qualifier[statement_type]['id']):
                            top_statements.append((1, statement))
                        else:
                            nlp_word_to = get_nlp(get_wd_label(qualifier[statement_type]['id']))
                            matching_similarity = get_similarity_by_words(nlp_name, nlp_word_to)
                            top_statements.append((matching_similarity, statement))
                else:
                    if time_sentitive and is_timestamp(statement[statement_type]['id']):
                        top_statements.append((1, statement))
                    else:
                        nlp_word_to = get_nlp(get_wd_label(statement[statement_type]['id']))
                        matching_similarity = get_similarity_by_words(nlp_name, nlp_word_to)
                        top_statements.append((matching_similarity, statement))
            else:    
                if qualifier:
                    if statement.get('qualifiers'):
                        for qualifier in statement['qualifiers']:
                            if time_sentitive and is_timestamp(qualifier[statement_type]['id']):
                                top_statements.append((1, statement))
                            else:
                                nlp_word_to = get_nlp(get_wd_label(qualifier[statement_type]['id']))
                                matching_similarity = get_similarity_by_words(nlp_name, nlp_word_to)
                                if highest_matching_similarity == -1 or matching_similarity >= highest_matching_similarity:
                                    highest_matching_similarity = matching_similarity
                                    best_statement = statement
                                    top_statements.append((highest_matching_similarity, best_statement))
                else:
                    if time_sentitive and is_timestamp(statement[statement_type]['id']):
                        top_statements.append((1, statement))
                    else:
                        nlp_word_to = get_nlp(get_wd_label(statement[statement_type]['id']))
                        matching_similarity = get_similarity_by_words(nlp_name, nlp_word_to)
                        if highest_matching_similarity == -1 or matching_similarity >= highest_matching_similarity:
                            highest_matching_similarity = matching_similarity
                            best_statement = statement
                            top_statements.append((highest_matching_similarity, best_statement))
    if top_k > 0:        
        return sorted(top_statements, key=lambda x: x[0], reverse=True)[:top_k]
    else:
        return sorted(top_statements, key=lambda x: x[0], reverse=True)
    
#statements = get_all_statements_of_entity('Q503992')
#top_similar_statements = get_top_similar_statements(statements, 'Q267721', 'western')
#print(top_similar_statements)
 


# In[29]:


def get_best_similar_statements_by_word_worker(in_mp_queue, out_mp_queue, top_k, qualifier, statement_type, time_sentitive):
    sentinel = None
    best_statements = []
    
    for token,similar_to_name in iter(in_mp_queue.get, sentinel):
    #    print("working on",token,similar_to_name)
        statements = get_all_statements_of_entity(token)
        if statements: best_statements += get_top_similar_statements(statements, token, similar_to_name, top_k=top_k, qualifier=qualifier, statement_type=statement_type, time_sentitive=time_sentitive)
    #    print("done with",token,similar_to_name)
    
    out_mp_queue.put(best_statements)
        


# In[30]:


def get_best_similar_statements_by_word(from_token_ids, similar_to_name, top_k=3, qualifier=False, statement_type="object", time_sentitive=False, cores=1):
    if not similar_to_name:
        return []
    
    best_statements = []
    if cores > 1:
        out_mp_queue = mp.Queue()
        in_mp_queue = mp.Queue()
        sentinel = None

        for token in from_token_ids:
            in_mp_queue.put((token,similar_to_name))

        procs = [mp.Process(target = get_best_similar_statements_by_word_worker, args = (in_mp_queue, out_mp_queue, top_k, qualifier, statement_type, time_sentitive)) for i in range(cores)]

        for proc in procs:
            proc.daemon = True
            proc.start()
        for proc in procs:    
            in_mp_queue.put(sentinel)
        for proc in procs:
            best_statements += out_mp_queue.get()
        for proc in procs:
            proc.join()
    else:
        for token in from_token_ids:
            statements = get_all_statements_of_entity(token)
            if statements: best_statements += get_top_similar_statements(statements, token, similar_to_name, top_k=top_k, qualifier=qualifier, statement_type=statement_type, time_sentitive=time_sentitive)
    
    #print("best_statements",best_statements)
    return sorted(best_statements, key=lambda x: x[0], reverse=True)

#best_similar_statements = get_best_similar_statements_by_word(extract_ids(q0_themes[0]), 'voiced', top_k=3, qualifier=True, statement_type="qualifier_object")
#print(best_similar_statements[0])

#init_clusters = cluster_extend_by_words(theme_ids, [p[0].text for p in q_predicates+predicates_enhanced], top_k=deep_k, time_sentitive=time_sensitive,cores=2)


# In[31]:


def get_statements_subjects_labels(statements):
    return [get_wd_label(t[1]['entity']['id']) for t in statements]
#print(get_statements_subjects_labels(best_similar_statements))


# In[32]:


def get_statements_predicates_labels(statements):
    return [get_wd_label(t[1]['predicate']['id']) for t in statements]
#print(get_statements_predicates_labels(best_similar_statements))


# In[33]:


def get_statements_objects_labels(statements):
    return [get_wd_label(t[1]['object']['id']) for t in statements]
#print(get_statements_objects_labels(best_similar_statements))


# In[34]:


def get_statements_qualifier_predicates_labels(statements):
    return [get_wd_label(t[1]['qualifiers'][0]['qualifier_predicate']['id']) for t in statements]
#print(get_statements_qualifier_predicates_labels(best_similar_statements))


# In[35]:


def get_statements_qualifier_objects_labels(statements):
    return [get_wd_label(t[1]['qualifiers'][0]['qualifier_object']['id']) for t in statements]
#print(get_statements_qualifier_objects_labels(best_similar_statements))


# In[36]:


def cluster_extend_by_words_worker(in_mp_queue, out_mp_queue, top_k, time_sentitive, cores):
    sentinel = None
    cluster = []
    
    for cluster_root_ids,name in iter(in_mp_queue.get, sentinel):
        cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=True, statement_type="qualifier_predicate", time_sentitive=time_sentitive,cores=1)
        cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=True, statement_type="qualifier_object", time_sentitive=time_sentitive,cores=1)
        cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=False, statement_type="predicate", time_sentitive=time_sentitive,cores=1)
        cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=False, statement_type="object", time_sentitive=time_sentitive,cores=1)
        
    out_mp_queue.put(cluster)
    


# In[37]:


def cluster_extend_by_words(cluster_root_ids, extending_words, top_k=3, time_sentitive=False,cores=mp.cpu_count()):
    if not cluster_root_ids or not extending_words:
        return []
    cluster = []
    
    if cores <= 0: cores = 1
    out_mp_queue = mp.Queue()
    in_mp_queue = mp.Queue()
    sentinel = None
    
    for name in extending_words:
        in_mp_queue.put((cluster_root_ids,name))
        
    procs = [mp.Process(target = cluster_extend_by_words_worker, args = (in_mp_queue, out_mp_queue, top_k, time_sentitive, cores)) for i in range(cores)]
    
    for proc in procs:
        proc.daemon = True
        proc.start()
    for proc in procs:    
        in_mp_queue.put(sentinel)
    for proc in procs:
        cluster += out_mp_queue.get()
    for proc in procs:
        proc.join()    
    
    return sorted(cluster, key=lambda x: x[0], reverse=True)
    
    #start_time = time.time()
    #for name in extending_words:
    #    #start_cluster_time = time.time()
    #    cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=True, statement_type="qualifier_predicate", time_sentitive=time_sentitive,cores=cores)
    #    cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=True, statement_type="qualifier_object", time_sentitive=time_sentitive,cores=cores)
    #    cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=False, statement_type="predicate", time_sentitive=time_sentitive,cores=cores)
    #    cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=False, statement_type="object", time_sentitive=time_sentitive,cores=cores)
    #    #end_time = time.time()
    #    #print("EXTENDING Cluster with:", name," ->\tRunning time is {}s".format(round(end_time-start_cluster_time,2)))
    ##end_time = time.time()
    ##print("EXTENDING Clusters ->\tRunning time is {}s".format(round(end_time-start_time,2)))
    
#test_cluster = cluster_extend_by_words(extract_ids(q0_themes[0]), ['voiced'], top_k=2)
#test_cluster_test_2 = cluster_extend_by_words(extract_ids(q0_themes_test_2[0]), ['birth'], top_k=2)
#print(test_cluster[0])

#start_time = time.time()
#init_clusters = cluster_extend_by_words(theme_ids, [p[0].text for p in q_predicates+predicates_enhanced], top_k=deep_k, time_sentitive=time_sensitive,cores=mp.cpu_count())
#print("timer",time.time()-start_time)


# In[38]:


# sorts by the similarity value of statements[0]
def sort_statements_by_similarity(statements):
    return [s for s in sorted(statements, key=lambda x: x[0], reverse=True)]

#test_sorted_statements = sort_statements_by_similarity(test_cluster)
#test_sorted_statements_test_2 = sort_statements_by_similarity(test_cluster_test_2)
#print(test_sorted_statements[0])


# In[39]:


# appends spo from qualifiers, removes qualifier tags, and removes similarity scores
def statements_flatter(statements):
    best_statements_to_graph = []
    for statement in statements:
        tmp_statement = copy(statement)
        if tmp_statement.get('qualifiers'):
            #print("statement", statement)
            for q in tmp_statement['qualifiers']:
                qualifier_statement = {'entity': {'id': tmp_statement['entity']['id']}}
                qualifier_statement['predicate'] = {'id': q['qualifier_predicate']['id']}
                qualifier_statement['object'] = {'id': q['qualifier_object']['id']}
                best_statements_to_graph.append(qualifier_statement)
            del(tmp_statement['qualifiers'])
        else:
            #print("tmp_statement", tmp_statement)
            if ('qualifiers' in tmp_statement): del(tmp_statement['qualifiers'])
        if tmp_statement not in best_statements_to_graph:
            #print("best_statements_to_graph", tmp_statement)
            best_statements_to_graph.append(tmp_statement)
        
    return best_statements_to_graph

#test_flatten_statements = statements_flatter([s[1] for s in test_sorted_statements])
#test_flatten_statements_test_2 = statements_flatter([s[1] for s in test_sorted_statements_test_2])
#print(test_flatten_statements[0])
#test_flatten_statements_test_2


# In[40]:


# remove duplicates from statements
def unduplicate_statements(statements):
    filtered_statements = []
    [filtered_statements.append(s) for s in statements if s not in [e for e in filtered_statements]]
    return filtered_statements

#test_unduplicate_statements = unduplicate_statements(test_flatten_statements)
#print(len(test_flatten_statements))
#print(len(test_unduplicate_statements))
#print(test_unduplicate_statements[0])


# In[41]:


def get_statements_by_id(statements, from_token_id, to_id, qualifier=False, statement_type="predicate"):
    id_statements = []
    if not statements:
        return id_statements
    if get_wd_label(from_token_id):
        for statement in statements:
            if qualifier:
                if statement.get('qualifiers'):
                    for s in statement['qualifiers']:
                        if to_id == s[statement_type]['id']:
                            id_statements.append(statement)
            else:
                if to_id == statement[statement_type]['id']:
                    id_statements.append(statement)
    
    return id_statements

#statements_test = get_all_statements_of_entity('Q176198')
#id_statements_test = get_statements_by_id(statements_test, 'Q176198', 'P725')
#print(id_statements_test[0])

#get_statements_by_id(root_statements, cluster_root_id, predicate_id, qualifier=False, statement_type="predicate")
#statements_test = get_all_statements_of_entity('Q176198')
#id_statements_test = get_statements_by_id(statements_test, 'Q176198', 'P725')
#id_statements_test[0]


# In[42]:


def cluster_extend_by_words(cluster_root_ids, extending_words, top_k=3, time_sentitive=False,cores=2):
    if not cluster_root_ids or not extending_words:
        return []
    cluster = []
    
    if cores <= 0: cores = 1
    out_mp_queue = mp.Queue()
    in_mp_queue = mp.Queue()
    sentinel = None
    
    for name in extending_words:
        in_mp_queue.put((cluster_root_ids,name))
        
    procs = [mp.Process(target = cluster_extend_by_words_worker, args = (in_mp_queue, out_mp_queue, top_k, time_sentitive, cores)) for i in range(cores)]
    
    for proc in procs:
        proc.daemon = True
        proc.start()
    for proc in procs:    
        in_mp_queue.put(sentinel)
    for proc in procs:
        cluster += out_mp_queue.get()
    for proc in procs:
        proc.join()    
    
    return sorted(cluster, key=lambda x: x[0], reverse=True)
    
    #start_time = time.time()
    #for name in extending_words:
    #    #start_cluster_time = time.time()
    #    cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=True, statement_type="qualifier_predicate", time_sentitive=time_sentitive,cores=cores)
    #    cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=True, statement_type="qualifier_object", time_sentitive=time_sentitive,cores=cores)
    #    cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=False, statement_type="predicate", time_sentitive=time_sentitive,cores=cores)
    #    cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=False, statement_type="object", time_sentitive=time_sentitive,cores=cores)
    #    #end_time = time.time()
    #    #print("EXTENDING Cluster with:", name," ->\tRunning time is {}s".format(round(end_time-start_cluster_time,2)))
    ##end_time = time.time()
    ##print("EXTENDING Clusters ->\tRunning time is {}s".format(round(end_time-start_time,2)))
    
#test_cluster = cluster_extend_by_words(extract_ids(q0_themes[0]), ['voiced'], top_k=2)
#test_cluster_test_2 = cluster_extend_by_words(extract_ids(q0_themes_test_2[0]), ['birth'], top_k=2)
#print(test_cluster[0])

#start_time = time.time()
#init_clusters = cluster_extend_by_words(theme_ids, [p[0].text for p in q_predicates+predicates_enhanced], top_k=deep_k, time_sentitive=time_sensitive,cores=mp.cpu_count())
#print("timer",time.time()-start_time)


# In[43]:


def cluster_extend_by_predicates_ids_worker(in_mp_queue, out_mp_queue):
    sentinel = None
    cluster = []
    
    for cluster_root_id, predicate_id in iter(in_mp_queue.get, sentinel):
        root_statements = get_all_statements_of_entity(cluster_root_id)
        if root_statements:
            cluster += get_statements_by_id(root_statements, cluster_root_id, predicate_id, qualifier=True, statement_type="qualifier_predicate")
            cluster += get_statements_by_id(root_statements, cluster_root_id, predicate_id, qualifier=False, statement_type="predicate")
    
    out_mp_queue.put(cluster)    


# In[44]:


# parameters
# cluster_root_ids: ['Qcode']
# predicates_ids: ['Pcode']
def cluster_extend_by_predicates_ids(cluster_root_ids, predicates_ids, cores=mp.cpu_count()):
    if not cluster_root_ids or not predicates_ids:
        return []
    
    cluster = []
    
    if cores <= 0: cores = 1
    out_mp_queue = mp.Queue()
    in_mp_queue = mp.Queue()
    sentinel = None
    
    for cluster_root_id, predicate_id in it.product(cluster_root_ids, predicates_ids):
        #print((cluster_root_id, predicates_id))
        in_mp_queue.put((cluster_root_id, predicate_id))
        
    procs = [mp.Process(target = cluster_extend_by_predicates_ids_worker, args = (in_mp_queue, out_mp_queue)) for i in range(cores)]
    
    for proc in procs:
        proc.daemon = True
        proc.start()
    for proc in procs:    
        in_mp_queue.put(sentinel)
    for proc in procs:
        cluster += out_mp_queue.get()
    for proc in procs:
        proc.join()    
    
    #for cluster_root_id in cluster_root_ids:
    #    root_statements = get_all_statements_of_entity(cluster_root_id)
    #    #print("root_statements", root_statements)
    #    for predicate_id in predicates_ids:
    #        cluster += get_statements_by_id(root_statements, cluster_root_id, predicate_id, qualifier=True, statement_type="qualifier_predicate")
    #        cluster += get_statements_by_id(root_statements, cluster_root_id, predicate_id, qualifier=False, statement_type="predicate")

    return cluster #sorted(cluster, key=lambda x: x[0], reverse=True)
    
#test_predicate_clusters = cluster_extend_by_predicates_ids(extract_ids(q0_themes[0]), extract_ids(q0_predicates))
#print(len(test_predicate_clusters))
#test_predicate_clusters[0]

#test_predicate_clusters_test_2 = cluster_extend_by_predicates_ids(extract_ids(q0_themes_test_2[0]), extract_ids(q0_predicates_test_2))
#print(len(test_predicate_clusters_test_2))
#print(test_predicate_clusters_test_2[-1])

#predicate_ids_clusters = cluster_extend_by_predicates_ids(theme_ids, predicates_ids+predicates_enhanced_ids)
#print(predicate_ids_clusters)


# In[45]:


def cluster_extractor_from_complements(complements):
    for c in complements:
        [print(t.pos_) for t in c]
    return complements

#print(cluster_extractor_from_complements(q0_themes[1]))


# In[46]:


#TODO: add cache
#TODO: Check if extending with predicate_ids is useful
# parameter
# question: nlp_string
#limits=plt.axis('off')
def build_graph(nlp, themes, themes_enhanced, predicates, deep_k=3, time_sensitive = False, cores=mp.cpu_count()):
    #print("time_sensitive",time_sensitive)
    #start_time = time.time()
    theme_ids = extract_ids(themes[0])
    theme_enhanced_ids = extract_ids(themes_enhanced)
    predicates_ids = extract_ids(predicates)
    predicates_enhanced_ids = [p for p in theme_enhanced_ids if is_wd_predicate(p)]
    predicates_enhanced = merge_lists([get_nlp(get_wd_label(p)) for p in predicates_enhanced_ids], predicates_enhanced_ids)
    
    #print(theme_ids)
    #print(theme_enhanced_ids)
    for i, tei in enumerate(theme_enhanced_ids):
        if tei in theme_ids:
            tmp = theme_enhanced_ids.pop(i)
    
    init_clusters = cluster_extend_by_words(theme_ids, [p[0].text for p in predicates+predicates_enhanced], top_k=deep_k, time_sentitive=time_sensitive, cores=cores)
    #print("init_clusters",len(init_clusters))
    init_clusters_enhanced = cluster_extend_by_words(theme_enhanced_ids, [p[0].text for p in predicates+predicates_enhanced], top_k=deep_k, time_sentitive=time_sensitive, cores=cores)
    #print("init_clusters_enhanced",len(init_clusters_enhanced))
    init_sorted_statements = sort_statements_by_similarity(init_clusters + init_clusters_enhanced)
    #print("init_sorted_statements",len(init_sorted_statements))
    init_flatten_statements = statements_flatter([s[1] for s in init_sorted_statements])
    #print("init_flatten_statements",len(init_flatten_statements))
    
    predicate_ids_clusters = cluster_extend_by_predicates_ids(theme_ids, predicates_ids+predicates_enhanced_ids, cores=cores)
    #print("predicate_ids_clusters",len(predicate_ids_clusters))
    predicate_ids_enhanced_clusters = cluster_extend_by_predicates_ids(theme_enhanced_ids, predicates_ids+predicates_enhanced_ids, cores=cores)
    #print("predicate_ids_enhanced_clusters",len(predicate_ids_enhanced_clusters))
    predicate_ids_flatten_statements = statements_flatter(predicate_ids_clusters+predicate_ids_enhanced_clusters)
    #print("predicate_ids_flatten_statements",len(predicate_ids_flatten_statements))
    
    clusters = init_flatten_statements+predicate_ids_flatten_statements
    filtered_statements = unduplicate_statements(clusters)
    #print(predicate_ids_enhanced_clusters)
    graph = make_statements_graph(filtered_statements, cores=cores)
    
    #print([get_wd_label(e) for e in g.nodes] )
    

    ##print("clusters:", len(clusters))
    ##print("filtered_statements:", len(filtered_statements))
    #end_time = time.time()
    #print("->\tRunning time is {}s".format(round(end_time-start_time,2)))
    
    return graph

#q0_test = questions[0]
#q0_test = "Which actor voiced the Unicorn in The Last Unicorn?"
#q0_test = "what was the cause of death of yves klein"
#q0_test = "Who is the wife of Barack Obama?"
#q0_test = "Who is the author of Le Petit Prince?"
#q0_nlp_test = get_nlp(q0_test)
#q0_themes_test = get_themes(q0_nlp_test, top_k=3)
#q0_themes_enhanced_test = get_enhanced_themes(q0_themes_test, top_k=3)
#q0_predicates_test = get_predicates_online(q0_nlp_test, top_k=3)
#q0_focused_parts_test = []
#graph, predicates_dict = build_graph(q0_nlp_test, q0_themes_test, q0_themes_enhanced_test, q0_predicates_test, deep_k=3)
#print(predicates_dict)
#plot_graph(graph, "file_name_graph", "Graph_title")


# In[47]:


def filter_graph_by_names(graph, filtering_names, entities=True, predicates=False):
    # remove meaningless subgraphs
    graph_copy = graph.copy()
    for g in [g for g in (graph.subgraph(c) for c in nx.connected_components(graph))]:
        is_meaningful = False
        for e in g:
            if get_wd_label(e) in filtering_names:
                is_meaningful = True
                break
        if not is_meaningful:
            for e in g:
                graph_copy.remove_node(e)
                
    return graph_copy


#q_theme_names = [q[0].text for q in q_themes[0]]
#q_theme_enhanced_names = [q[0] for q in q_themes_enhanced]
#filter_graph_by_names(graph, q_theme_names+q_theme_enhanced_names, entities=True, predicates=False)
#plot_graph(graph, "subgraph_test", "subgraph_test")


# In[48]:


#TODO
def filter_graph_by_ids(graph, filtering_ids, entities=True, predicates=False):
    # remove meaningless subgraphs
    #meaningful_ids = theme_ids+theme_enhanced_ids
    #print("meaningful_ids",meaningful_ids)
    filtered_graph = nx.Graph()
    for g in [g for g in (graph.subgraph(c) for c in nx.connected_components(graph))]:
        is_meaningful = False
        for e in g:
            #print("e",e)
            if e in filtering_ids:
                is_meaningful = True
                #print("break")
                break
        if is_meaningful:
            nx.compose(filtered_graph,g)
    #print("filtered_graph",filtered_graph)


    #plot_graph(filtered_graph, "subgraph_test", "subgraph_test")
    #plot_graph(graph, "subgraph_test", "subgraph_test")

#q_theme_ids = extract_ids(q_themes[0])
#q_theme_enhanced_ids = extract_ids(q_themes_enhanced)
#filter_graph_by_ids(graph, q_theme_ids+q_theme_enhanced_ids, entities=True, predicates=False)


# In[49]:


# check the graph for complements
# parameters
# name: string
def find_name_in_graph(graph, name):
    return [x for x,y in graph.nodes(data=True) if y['name'].lower() == name.lower()]

#[find_name_in_graph(c.text) for c in q0_themes[1]]
#print(find_name_in_graph(graph, "the unicorn"))


# In[50]:


# check the graph for complements
# parameters
# name: string
def find_id_in_graph(graph, id_to_find):
    return [x for x,y in graph.nodes(data=True) if x == id_to_find]

#[find_name_in_graph(c.text) for c in q0_themes[1]]
#print(find_name_in_graph(graph, "the unicorn"))


# In[51]:


# TODO: clean the complements by removing stopwords etc.
def find_theme_complement(graph, themes):
    return [i for i in it.chain.from_iterable(
        [id for id in [c for c in [find_name_in_graph(graph, t.text) for t in themes[1]] if c]])]

#print(find_theme_complement(graph, q0_themes_test))
#[i for i in it.chain.from_iterable([id for id in check_theme_complement(graph, q0_themes)])]


# In[52]:


def find_paths_in_graph(graph, node_start, node_end):
    return [p for p in nx.all_simple_paths(graph, source=node_start, target=node_end)]
        
#test_paths = find_paths_in_graph(graph, "Q16205566", "Q7774795")
#print(test_paths)


# In[53]:


def is_id_in_graph(graph, node_id):
    return graph.has_node(node_id)
#print(is_id_in_graph(graph, "Q24039104"))


# In[54]:


def is_name_in_graph(graph, node_name):
    return find_name_in_graph(graph, node_name) != []
#print(is_name_in_graph(graph, "the Unicorn"))


# In[55]:


def find_paths_for_themes(graph, themes):
    themes_ids = [t for t in  extract_ids(themes[0])]
    complements_ids = find_theme_complement(graph, themes)
    paths = []
    for t_id in themes_ids:
        if is_id_in_graph(graph, t_id):
            for c_id in complements_ids:
                if is_id_in_graph(graph, c_id):
                    path = find_paths_in_graph(graph, t_id, c_id)
                    if path:
                        paths.append(path)
    paths = [i for i in it.chain.from_iterable(
        [id for id in paths])]
    
    return paths
#print(find_paths_for_themes(graph, q0_themes_test))
#print(find_paths_for_themes(graph, q0_themes))


# In[56]:


def get_node_predicates_from_path(paths):
    predicates = []
    for p in paths:
        [predicates.append(i[:i.find("-")]) for i in p if is_wd_predicate(i[:i.find("-")]) and i[:i.find("-")] not in predicates]
    return predicates

#test_node_predicates = get_node_predicates_from_path(test_paths)
#print(test_node_predicates)


# In[57]:


def get_node_predicate_similarity_from_path(paths, predicates):
    path_predicates = get_node_predicates_from_path(paths)
    return sorted([(pp, get_similarity_by_ids(p2, pp)) for p in predicates for p2 in p[1] for pp in path_predicates], key=lambda x: x[-1], reverse=True)

#test_node_pedicate_similarities = get_node_predicate_similarity_from_path(test_paths, q0_predicates)
#print(test_node_pedicate_similarities)


# In[58]:


def get_focused_parts(nlp_sentence, themes, top_k=3):
    W_FILTERS = ["WDT", "WP", "WP$", "WRB"]
    V_FILTERS = ["VERB", "AUX"]
    
    dummy_doc = get_nlp("dummy doc")
    
    focused_parts = [t.head for t in nlp_sentence if t.tag_ in W_FILTERS] 
    for fp in focused_parts:
        if fp.children:
            for c in fp.children:
                if c.tag_ not in W_FILTERS and c.text not in [fp.text for fp in focused_parts]: 
                    focused_parts.append(c)
    
    #print("focused_parts",focused_parts)
    #print("themes[0]",themes[0])
    
    for t in themes[0]:
        for i_fp, fp in enumerate(focused_parts):
            #print("fp",fp, type(fp))
            for i_w, w in enumerate([w.lower_ for w in t[0]]):
                #print("w",w, type(w))
                if fp.lower_ == w:
                    #print("MATCHING")
                    if i_fp+1 < len(focused_parts):
                        #print("focused_parts[i_fp+1].lower_",focused_parts[i_fp+1].lower_)
                        #print("t[0][i_w-1].lower_",t[0][i_w-1].lower_)
                        if focused_parts[i_fp+1].lower_ == t[0][i_w-1].lower_:
                            #print(i_fp,fp, t[0][i_w-1], t[0])
                            #print("BEFORE focused_parts",focused_parts)
                            #print("t[0]",t[0])
                            
                            if type(t[0]) == type(dummy_doc):
                                focused_parts[i_fp] = t[0][:]
                            else:
                                focused_parts[i_fp] = t[0]
                            del focused_parts[i_fp+1]
                            #print("AFTER focused_parts",focused_parts)
                            
    
    #print()
    #for fp in focused_parts:
    #    print(type(fp))
    #    
    #        print(fp.as_doc())
        #if isinstance() == 'spacy.tokens.span.Span':
        #    print("in")
        #
    #focused_parts = [type(fp) for fp in focused_parts]
    #print("focused_parts",focused_parts)

    focused_parts_ids = [get_wd_ids(p.text, top_k=top_k) for p in focused_parts]
    #focused_parts_ids = [get_wd_ids(p.text, top_k=top_k, online=True) for p in focused_parts]
    #print("focused_parts_ids",focused_parts_ids)
    merged_list = merge_lists(focused_parts, focused_parts_ids)
    #print("merged_list",merged_list)
    
    dummy_span = dummy_doc[:]
    merged_list_filtered = []
    for ml in merged_list:
        if ml[1]:
            if type(ml[0]) == type(dummy_span):
                merged_list_filtered.append(ml)
            elif ml[0].pos_ not in V_FILTERS and not ml[0].is_stop:
                merged_list_filtered.append(ml)
                    
    return merged_list_filtered

#q_test_nlp = get_nlp("what's akbar tandjung's ethnicity")
#print(get_focused_parts(q0_nlp_test))

#q_test_nlp = get_nlp("Who voiced the Unicorn in The Last Unicorn?")
#print(get_focused_parts(q0_nlp_test))

#q_test_nlp = get_nlp("Who is the author that wrote the book Moby Dick")
#q_test_themes = get_themes(q_test_nlp, top_k=3)
#get_focused_parts(q_test_nlp,q_test_themes, top_k=3)

#q_test_nlp = get_nlp("Where was Shigeyasu Suzuki Place of Birth")
#q_test_nlp = get_nlp("Who is the author that wrote the book Moby Dick")
#q_test_nlp = get_nlp("Where was Shigeyasu Suzuki Place of Birth")
#q_test_themes = get_themes(q_test_nlp, top_k=3)
#get_focused_parts(q_test_nlp,q_test_themes, top_k=3)

#q_focused_parts: [(Unicorn, ['Q18356448', 'Q21070472', 'Q22043340', 'Q1565614', 'Q30060419']),
#(in, ['P642', 'Q29733109', 'P361', 'P131']),
#(the, ['Q1408543', 'Q2865743', 'Q29423', 'Q21121474']),
#(Unicorn, ['Q18356448', 'Q21070472', 'Q22043340', 'Q1565614', 'Q30060419']),
#(The, ['Q1067527', 'Q13423400', 'Q28457426', 'Q24406786', 'Q2430521', 'Q37199001']),
#(Last, ['Q16995904', 'Q20072822', 'Q24229340', 'Q20155285'])]

#[(author, ['P676', 'Q482980', 'Q3154968']),
# (book, ['Q571', 'Q4942925', 'Q997698']),
# (Dick, ['Q1471500', 'Q21510351', 'Q249606']),
# (Moby, ['Q1954726', 'Q6887412', 'Q14045'])]


# In[59]:


def is_in_list_by_similarity(word,list_of_words,similarity_threshold):
        nlp_word = get_nlp(word)
        for lw in list_of_words:
            if nlp_word.similarity(get_nlp(lw)) > similarity_threshold:
                return True
        return False

#is_in_list_by_similarity("Moby Dick", ["moby-dick","star wars"],0.9)


# In[60]:


def add_compound(nlp_list, themes):
    compounded = []
    #if not nlp_list[0]:
    #    return compounded
    try:
        for t in [e[0] for e in themes[0]] + themes[1]:
            for l in [n[0] for n in nlp_list]:
                if l.text.lower() in t.text.lower():
                    compounded.append(t.text)
        return compounded
    except:
        return compounded

# TODO: make the predicate search go further in the path list for the !i%2
def find_paths_keywords(graph, nlp, themes, themes_enhanced, predicates, focused_parts, keywords_len_limit=5, similarity_threshold=0.9):
    WH_FILTER = ["WDT", "WP", "WP$", "WRB"]
    VERB_FILTER = ["VERB", "AUX"]
    NOUN_FILTER = ["NOUN","PROPN"]
    POSITION_FILTER = ["ADP"]
    
    focused_parts_words = [t[0].text for t in focused_parts]
    focused_parts_ids = [j for i in [t[1] for t in focused_parts] for j in i]
    focused_parts_predicates_ids = [f for f in focused_parts_ids if is_wd_predicate(f)]
    focused_parts_words_ids = [f for f in focused_parts_ids if is_wd_entity(f)]
    focused_parts_words_ids_labeled = [get_wd_label(p) for p in focused_parts_words_ids]
    #print(focused_parts_words_2)

    question_anchors = [t for t in nlp if t.tag_ in WH_FILTER]
    themes_enhanced_list = [t[0] for t in themes_enhanced]
    focus_themes = [t[0].text for t in themes[0]]
    focus_path_by_tails = [[c for c in t.head.children if c.pos_ in NOUN_FILTER] for t in nlp if t.pos_ == "PRON"]
    focus_part_by_head = [t.head for t in question_anchors]
    predicates_nlp = [t for t in nlp if t.pos_ in VERB_FILTER]
    predicates_lemma = [t.lemma_ for t in predicates_nlp]
    predicates_attention = [t for t in nlp if t.head in predicates_nlp]
    predicates_attention_tails = [[c for c in t.children] for t in predicates_attention]
    in_attention_heads = [t.head.text for t in nlp if t.pos_ in POSITION_FILTER]
    in_attention_tails = add_compound([[c for c in t.children] for t in nlp if t.pos_ in POSITION_FILTER], themes)
    focus_themes_enhanced = [t[0] for t in themes_enhanced
                             if t[0].lower() in [a.lower() for a in in_attention_tails]
                             or t[0].lower() in [a.lower() for a in in_attention_heads]]
    
    theme_enhanced_ids = extract_ids(themes_enhanced)
    predicates_enhanced_ids = [(p) for p in theme_enhanced_ids if is_wd_predicate(p)]
    [predicates_enhanced_ids.append(p) for p in focused_parts_predicates_ids if p not in predicates_enhanced_ids]
    
    alterniative_words = {}
    for t in themes_enhanced:
        for e in predicates_enhanced_ids:
            if e in t[1]:
                alterniative_words[t[0]] = [get_nlp(get_wd_label(e)),[e]]
            else:
                alterniative_words[get_wd_label(e)] = [get_nlp(get_wd_label(e)),[e]]
    
    #print("focused_parts_predicates_ids",focused_parts_predicates_ids)
    #print("focused_parts_words_ids",focused_parts_words_ids)
    #print("alterniative_words",alterniative_words)
    #print("predicates_enhanced_ids",predicates_enhanced_ids)
    ##print("predicates_enhanced",predicates_enhanced)
    #print("question_anchors",question_anchors)
    #print("in_attention_heads",in_attention_heads)
    #print("in_attention_tails",in_attention_tails)
    #print("focus_themes",focus_themes)
    #print("themes_enhanced_list",themes_enhanced_list)
    #print("focus_themes_enhanced",focus_themes_enhanced)
    #print("focus_path_by_tails",focus_path_by_tails)
    #print("focus_part_by_head",focus_part_by_head)
    #print("predicates_nlp",predicates_nlp)
    #print("predicates_lemma",predicates_lemma)
    #print("predicates_attention",predicates_attention)
    #print("predicates_attention_tails",predicates_attention_tails)
    #
    #print("\n")
    paths_keywords = []
    [paths_keywords.append(e.lower()) for e in focused_parts_words + in_attention_heads + in_attention_tails + focus_themes + focus_themes_enhanced + focused_parts_words_ids_labeled if e.lower() not in paths_keywords]
    #print(paths_keywords)
    #paths_keywords = [p for p in it.permutations(paths_keywords)]
    #print(paths_keywords)
    
    paths_keywords = [p for p in paths_keywords if p and len(p.split(" ")) <= keywords_len_limit]
    
    paths_keywords_filtered = []
    
    #print("paths_keywords",paths_keywords)
    #for k in paths_keywords:
    #    print("current k",k)
    #    #print("paths_keywords_filtered before",paths_keywords_filtered)
    #    is_in_list_by_similarity(k, paths_keywords_filtered,similarity_threshold)
    
    [paths_keywords_filtered.append(k) for k in paths_keywords if not is_in_list_by_similarity(k, paths_keywords_filtered,similarity_threshold)]
    
    #is_in_list_by_similarity("Moby Dick", ["moby-dick","star wars"],0.9)
        
    
    return paths_keywords_filtered, alterniative_words, question_anchors
    
    #initial_paths = find_paths_for_themes(graph, themes)
    #predicate_id_similarities = get_node_predicate_similarity_from_path(initial_paths, predicates)
    #best_path = [p for p in initial_paths if predicate_id_similarities[0][0] == p[1][:p[1].find("-")]]
    #path_answer = get_wd_label(best_path[0][2]) if best_path else []
    
    #return (path_answer, best_path[0][2]) if path_answer else (False, False)

#paths_keywords_2 = find_paths_keywords(graph_2, q_nlp_2, q_themes_2, q_themes_enhanced_2, q_predicates_2, q_focused_parts_2)
#paths_keywords_2


# In[61]:


def get_all_simple_paths_worker(graph, in_mp_queue, out_mp_queue):
    found_paths = []
    sentinel = None
    for source, target in iter(in_mp_queue.get, sentinel):
        for path in nx.all_simple_paths(graph, source = source, target = target, cutoff = None):
            found_paths.append(path)
    out_mp_queue.put(found_paths)


# In[62]:


def get_keywords_nodes_worker(graph, threshold, in_mp_queue, out_mp_queue):
    keywords_nodes = []
    sentinel = None
    for name in iter(in_mp_queue.get, sentinel):
        #print("2 get_paths_keywords_nodes")
        nlp_lookup = get_nlp(name)
        keywords_nodes = [x for x,y in graph.nodes(data=True) if get_nlp(y['name']).similarity(nlp_lookup) >= threshold]
    
    out_mp_queue.put(keywords_nodes)


# In[63]:


def do_nothing(perm):
    return perm
    
def get_paths_keywords_nodes(graph, keywords,threshold=0.9,top_performance=50, cores=mp.cpu_count()):
    #print("1 get_paths_keywords_nodes")
    if cores <= 0: cores = 1
    sentinel = None
    
    out_mp_queue = mp.Queue()
    in_mp_queue = mp.Queue()
    
    for k in keywords:
        in_mp_queue.put(k)

    procs = [mp.Process(target = get_keywords_nodes_worker, args = (graph, threshold, in_mp_queue, out_mp_queue)) for i in range(cores)]
    
    keywords_nodes = []
    
    for proc in procs:
        proc.daemon = True
        proc.start()
    for proc in procs:    
        in_mp_queue.put(sentinel)
    for proc in procs:
        keywords_nodes.append(out_mp_queue.get())
    for proc in procs:
        proc.join()
        
    keywords_nodes = [k for k in keywords_nodes if k]
    #print("3 get_paths_keywords_nodes")
    
    #print("1 get_paths_keywords_nodes")
    #keywords_nodes = []
    #print("len(keywords)",len(keywords))
    #for k in keywords:
    #    nlp_lookup = get_nlp(k)
    #    keywords_nodes.append([x for x,y in graph.nodes(data=True) if get_nlp(y['name']).similarity(nlp_lookup) >= threshold])
    #    print("2 get_paths_keywords_nodes")
    #keywords_nodes = [k for k in keywords_nodes if k]
    #print("3 get_paths_keywords_nodes")
    
    #keywords_nodes [['Q17521117', 'Q17521118', 'Q557214', 'Q421946', 'Q11282976', 'Q4677712', 'Q33999'], ['Q7246', 'Q1307944', 'Q21070472', 'Q18356448', 'Q1863113', 'Q20983877', 'Q226755', 'Q22043340'], ['Q176198', 'Q967268', 'Q17553756', 'Q30060419', 'Q17985004', 'Q16614390', 'Q18647334', 'Q15628943'], ['Q176198', 'Q967268', 'Q17553756', 'Q30060419', 'Q17985004', 'Q16614390', 'Q18647334', 'Q15628943'], []]
    #keywords_nodes[0] ['Q17521117', 'Q17521118', 'Q557214', 'Q421946', 'Q11282976', 'Q4677712', 'Q33999']
    #keywords_nodes[1] ['Q7246', 'Q1307944', 'Q21070472', 'Q18356448', 'Q1863113', 'Q20983877', 'Q226755', 'Q22043340']
    
    keywords_nodes_per = []
    if keywords_nodes:
        #print("4 get_paths_keywords_nodes")
        if len(keywords_nodes) > 1:
            #print("5 get_paths_keywords_nodes")
            for kn_i, kn in enumerate(keywords_nodes):
                #print("6 get_paths_keywords_nodes")
                if kn_i + 1 < len(keywords_nodes):
                    if len(kn) * len(keywords_nodes[kn_i+1]) > top_performance:
                        if len(kn) <= int(sqrt(top_performance)):
                            keywords_nodes[kn_i+1] = keywords_nodes[kn_i+1][:int(top_performance/len(kn))]
                        elif len(kn) >= len(keywords_nodes[kn_i+1]):
                            kn = kn[:int(top_performance/len(keywords_nodes[kn_i+1]))]
                        else:
                            kn = kn[:int(sqrt(top_performance))]
                            keywords_nodes[kn_i+1] = keywords_nodes[kn_i+1][:int(sqrt(top_performance))]
                #print("7 get_paths_keywords_nodes")
            
            #print("8 get_paths_keywords_nodes")
            with mp.Pool() as pool:
                keywords_nodes_per = pool.map(do_nothing, it.permutations(keywords_nodes, 2))
            #print("9 get_paths_keywords_nodes")
            
            keywords_nodes_per = [p for p in keywords_nodes_per]
            #print(">1 len(keywords_nodes_per",len(keywords_nodes_per),keywords_nodes_per[0])
                            
        else:
            keywords_nodes_per = [(keywords_nodes+keywords_nodes)]
            #print("<1 len(keywords_nodes_per)",len(keywords_nodes_per),keywords_nodes_per[0])
            
 
    #print("keywords_nodes_per",keywords_nodes_per)
    #return 0
    paths_keyword_nodes = []
    

    targets = []
    sources = []
    for pkn in keywords_nodes_per:
        [sources.append(pkn0) for pkn0 in pkn[0] if pkn0 not in sources]
        [targets.append(pkn1) for pkn1 in pkn[1] if pkn1 not in targets]# and pkn1 not in pkn[0]]

    #print("len(targets)",len(targets))
    #print("len(sources)",len(sources))

    out_mp_queue = mp.Queue()
    in_mp_queue = mp.Queue()
    sentinel = None

    for source, target in it.product(sources, targets):
        in_mp_queue.put((source, target))

    procs = [mp.Process(target = get_all_simple_paths_worker, args = (graph, in_mp_queue, out_mp_queue)) for i in range(cores)]

    for proc in procs:
        proc.daemon = True
        proc.start()
    for proc in procs:    
        in_mp_queue.put(sentinel)
    for proc in procs:
        paths_keyword_nodes.extend(out_mp_queue.get())
    for proc in procs:
        proc.join()
    
    paths_keyword_nodes = [p for p in paths_keyword_nodes if p]
    #paths_keyword_nodes_filtered = []
    #[paths_keyword_nodes_filtered.append(p) for p in paths_keyword_nodes if p not in paths_keyword_nodes_filtered]
    #print("len(paths_keyword_nodes)",len(paths_keyword_nodes))
    
    return paths_keyword_nodes

def find_path_nodes_from_graph(nlp_question, graph, predicates_dict, keywords, threshold=0.9,special_pred_theshold=0.7, thres_inter=0.15, top_performance=50,min_paths=3000, cores=mp.cpu_count()):
    #print("current threshold", str(round(threshold, 1)))
    
    w_positions, w_names = w_converter(nlp_question)
    w_names_only = [wn[1] for wn in w_names]
    date_trigger = "date" in w_names_only
    location_trigger = "location" in w_names_only
    all_predicates = list(predicates_dict.keys())

    option_keywords = []
    
    if date_trigger:
        nlp_time = get_nlp("time")
        nlp_date = get_nlp("date")
        
        for p in all_predicates:
            #print("current p", p)
            p_label = get_wd_label(p)
            nlp_p = get_nlp(p_label)
            #print("nlp_p",nlp_p)
            p_date = nlp_p.similarity(nlp_date)
            p_time = nlp_p.similarity(nlp_time)
            
            #print("p_date",p_date)
            #print("p_time",p_time)
            
            if p_date > special_pred_theshold or p_time > special_pred_theshold:
                if p not in option_keywords:
                    #print("adding",p)
                    option_keywords.append(p_label)
                    
    if location_trigger:
        nlp_location = get_nlp("location")
        nlp_place = get_nlp("place")
        
        for p in all_predicates:
            #print("current p", p)
            p_label = get_wd_label(p)
            nlp_p = get_nlp(p_label)
            #print("nlp_p",nlp_p)
            p_location = nlp_p.similarity(nlp_location)
            p_place = nlp_p.similarity(nlp_place)
            
            #print(p_label, "p_location",p_location)
            #print(p_label, "p_place",p_place)
            
            if p_location > special_pred_theshold or p_place > special_pred_theshold:
                if p not in option_keywords:
                    #print("adding",p)
                    option_keywords.append(p_label)
    
    for k in keywords[0]:
        nlp_k = get_nlp(get_wd_label(k))
        for p in all_predicates:
            #print("current p", p)
            p_label = get_wd_label(p)
            nlp_p = get_nlp(p_label)
            #print("nlp_p",nlp_p)
            p_k_sim = nlp_p.similarity(nlp_k)

            if p_k_sim > special_pred_theshold:
                if p not in option_keywords:
                    option_keywords.append(p_label)
                    
    #print("keywords[1]",keywords[1])
    
    k1_predicates = keywords[1].values() #[[country of citizenship, ['P27']], [country of citizenship, ['P27']]]
    #print("k1_predicates",k1_predicates)
    k1_predicates = sum([[get_wd_label(p) for p in e[1]] for e in k1_predicates],[])
    #print("k1_predicates",k1_predicates)

                
    #print("option_keywords",option_keywords)
    
    all_keywords = []
    [all_keywords.append(k) for k in keywords[0] + option_keywords + k1_predicates if k and k not in all_keywords]
    
    #print("all_keywords",all_keywords)
    
    main_keyword_paths = get_paths_keywords_nodes(graph, all_keywords,threshold=threshold,top_performance=top_performance, cores=cores)
    alternative_keyword_paths = []
    
    #for k_1 in keywords[1]:
    #    for i, k_0 in enumerate(all_keywords):
    #        if k_1==k_0:
    #            tmp_keywords = all_keywords.copy()
    #            tmp_keywords[i] = keywords[1][k_1][0].text
    #            alternative_keyword_paths += get_paths_keywords_nodes(graph, all_keywords,threshold=threshold,top_performance=top_performance, cores=cores)
    
    keyword_paths = main_keyword_paths#+alternative_keyword_paths
    
    #print("BEFORE len(keyword_paths)",len(keyword_paths))
    keyword_paths_filtered=[]
    [keyword_paths_filtered.append(p) for p in keyword_paths if p not in keyword_paths_filtered]
    keyword_paths = keyword_paths_filtered
    #print("keyword_paths",len(keyword_paths))
    
    #print("len(keyword_paths)",len(keyword_paths))
    if len(keyword_paths) < min_paths:
        if threshold == 0: return keyword_paths
        threshold -= thres_inter
        if threshold < 0: threshold = 0
        keyword_paths = get_paths_keywords_nodes(graph, all_keywords,threshold=threshold,top_performance=top_performance, cores=cores)
        
        keyword_paths_filtered=[]
        [keyword_paths_filtered.append(p) for p in keyword_paths if p not in keyword_paths_filtered]
        keyword_paths = keyword_paths_filtered
        
    #keyword_paths_filtered = []
    
    #print("AFTER len(keyword_paths)",len(keyword_paths))
    #[keyword_paths_filtered.append(p) for p in keyword_paths if p not in keyword_paths_filtered]
    
    
    
    return keyword_paths


#path_nodes_2 = find_path_nodes_from_graph(nlp_question,graph_2, predicates_dict,paths_keywords_2, threshold=0.9, special_pred_theshold=0.7, thres_inter=0.15, top_performance=50, min_paths=3000)
#end_time = time.time()

#start_time = time.time()
#path_nodes = find_path_nodes_from_graph(nlp_question,graph, predicates_dict,paths_keywords, threshold=0.8, thres_inter=0.1, top_performance=graph.size(),min_paths=3000,cores=2)
#print("--> len(path_nodes):",len(path_nodes))
#print("Finding path nodes ->\tRunning time is {}s".format(round(time.time()-start_time,2)))


# In[64]:


def is_sublist(a, b):
    if not a: return True
    if not b: return False
    #if a == b: return False
    return b[:len(a)] == a or is_sublist(a, b[1:])

def paths_nodes_filter_is_sublist_worker(in_mp_queue, out_mp_queue, filtered_paths):
    found_paths = []
    sentinel = None
    
    #print("HI I AM A PROCESSOR", filtered_paths)
    for i, fp in iter(in_mp_queue.get, sentinel):
        for fp_2 in filtered_paths:
            #print("will process",i,fp,fp_2)
            if (is_sublist(fp, fp_2) and fp!=fp_2):
                #print("processed",i,fp,fp_2)
                out_mp_queue.put(i)
                break
    
    out_mp_queue.put(False)
    #print("I AM TERMINATED")


# In[65]:


#node_predicates_names_2 = get_node_predicates_from_path(path_nodes_2)


def paths_nodes_filter(path_nodes, graph, cores=mp.cpu_count(), with_sublists=True):
    
    filtered_paths = []
    
    for path in path_nodes:
        filtered_row = []
        for i,p in enumerate(path):
            if is_wd_predicate(p[:p.find("-")]):
                if i == 0:
                    #if p[:p.find("-")] == "P725":
                    #    print(p)
                    neighbor = [k for k in graph[p].keys() if k != path[i+1]]
                    if neighbor:
                        filtered_row.append(neighbor[0])
                        filtered_row.append(p[:p.find("-")])
                    else:
                        continue
                    #print(filtered_row)
                elif i > 0 and i < len(path)-1:
                    filtered_row.append(p[:p.find("-")])
                else:
                    neighbor = [k for k in graph[p].keys() if k != path[i-1]]
                    if neighbor:
                        filtered_row.append(p[:p.find("-")])
                        filtered_row.append(neighbor[0])
                    else:
                        continue
            else: filtered_row.append(p)
        
        #print("filtered_paths",filtered_paths)
        
        if len(filtered_row) > 1 and filtered_row not in filtered_paths: 
            filtered_paths.append(filtered_row)
    
    if with_sublists:
        if cores <= 0: cores = 1

        out_mp_queue = mp.Queue()
        in_mp_queue = mp.Queue()
        sentinel = None

        for i,fp in enumerate(filtered_paths):
            in_mp_queue.put((i, fp))

        procs = [mp.Process(target = paths_nodes_filter_is_sublist_worker, args = (in_mp_queue, out_mp_queue, filtered_paths)) for i in range(cores)]

        to_remove_idexes = []
        for proc in procs:
            proc.daemon = True
            proc.start()
        for proc in procs:    
            in_mp_queue.put(sentinel)
        for proc in procs:
            to_remove_idexes.append(out_mp_queue.get())
        for proc in procs:
            proc.join(1)
        
        #print("to_remove_idexes",to_remove_idexes)
        

        for tri in to_remove_idexes:
            if tri:
                filtered_paths[tri] = []

        unique_paths = [p for p in filtered_paths if p]

        unique_paths_with_reversed = []

        for up in unique_paths:
            reversed_up = list(reversed(up))
            if up not in unique_paths_with_reversed: 
                unique_paths_with_reversed.append(up)
            if reversed_up not in unique_paths_with_reversed: 
                unique_paths_with_reversed.append(reversed_up)

        #print("unique_paths",len(unique_paths))

        #for i, up in enumerate(unique_paths):
        #    for up_2 in unique_paths:
        #        if (list(reversed(up)) == up_2):
        #            unique_paths[i] = []
        #            break


        #cleaned_paths = []
        #unique_paths = [up for up in unique_paths if up]

        #for up in unique_paths:
        #    for i,e in enumerate(up):
        #        if not is_wd_predicate(e):
        #            for j,r in enumerate(list(reversed(up))): 
        #                if not is_wd_predicate(r):
        #                    cleaned_paths.append(up[i:-j])
        #            break

        #print("cleaned_paths",len(cleaned_paths))

        #cleaned_paths = [c for c in cleaned_paths if len(c) > 2]

        #unique_paths = cleaned_paths.copy()
        #for i,fp in enumerate(cleaned_paths):
        #    for fp_2 in cleaned_paths:
        #        if (is_sublist(fp, fp_2) and fp!=fp_2):
        #            unique_paths[i] = []
        #            break

        #unique_paths = [p for p in unique_paths if len(p) > 2]       

        #for i, up in enumerate(unique_paths):
        #    for up_2 in unique_paths:
        #        if (list(reversed(up)) == up_2):
        #            unique_paths[i] = []
        #            break

            #print(up)
        #[up for up in unique_paths if up and not is_wd_predicate(up[-1]) and not is_wd_predicate(up[0])]
        #print()
        #for up in unique_paths:
        #    print(up)
        #    break
        #    return []
    else:
        unique_paths_with_reversed = filtered_paths
    
    return [p for p in unique_paths_with_reversed if len(p) > 2] #False#[up for up in unique_paths if up and not is_wd_predicate(up[-1]) and not is_wd_predicate(up[0])]#False# [p for p in unique_paths if p]
                
#paths_nodes_filtered_2 = paths_nodes_filter(path_nodes_2, graph_2)
#print("unique_paths", len(paths_nodes_filtered_2))
#for p in paths_nodes_filtered_2:
#    print(p)


# In[66]:


def w_converter(nlp):
    w_positions = []
    w_names = []
    for i_q,q in enumerate(nlp):
        if q.lemma_ == "where": 
            w_positions.append((i_q))
            w_names.append((i_q,"location"))
        elif q.lemma_ == "when": 
            w_positions.append((i_q))
            w_names.append((i_q,"date"))
        elif q.lemma_ == "who": 
            w_positions.append((i_q))
            w_names.append((i_q,"person"))    
        elif q.lemma_ == "why": 
            w_positions.append(i_q)
            w_names.append((i_q,"cause"))
        elif q.lemma_ == "which": 
            w_positions.append(i_q)
            w_names.append((i_q,"which"))
        elif q.lemma_ == "what": 
            w_positions.append(i_q)
            w_names.append((i_q,"what"))
        elif i_q+1 < len(nlp) and q.lemma_ == "how" and (nlp[i_q+1].lemma_ == "much" or nlp[i_q+1].lemma_ == "many"): 
            w_positions.append(i_q)
            w_names.append((i_q,"quantity"))
    return w_positions, w_names


# In[67]:


def get_entity_similarity(word_id, entity_type, banned_labels=[], max_reward=2.0):
    
    LOCATION_FILTER = ["GPE", "FAC", "LOC","PERSON"]
    PERSON_FILTER = ["PERSON","NORP","ORG","PER"]
    DATE_FILTER = ["DATE","TIME"]
    CAUSE_FILTER = ["NORP","PRODUCT","EVENT","MISC"]
    WHICH_FILTER = PERSON_FILTER+DATE_FILTER+["GPE","LOC","PRODUCT","EVENT",
                    "WORK_OF_ART","LAW","LANGUAGE","MISC"]
    WHAT_FILTER = LOCATION_FILTER+DATE_FILTER+CAUSE_FILTER+PERSON_FILTER+["WORK_OF_ART","LAW","LANGUAGE"]
                    
    QUANTITY_FILTER = ["PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
    
    ALL_FILTER = LOCATION_FILTER + PERSON_FILTER + DATE_FILTER + CAUSE_FILTER + WHICH_FILTER + WHAT_FILTER + QUANTITY_FILTER
    
    similarities = []
    word_label = get_wd_label(word_id)
    
    is_banned_label = word_label.lower() in banned_labels
    if word_label == "" and not is_timestamp(word_id):
        return similarities

    word_ents = get_kb_ents(word_label)
    
    #print(word_id,word_label,entity_type,[e.label_ for e in word_ents])
    
    #if is_timestamp(word_id):
        #print("is_timestamp")
        #print("word_ents",word_ents)
    
    if is_timestamp(word_id) and entity_type == "date" and not is_banned_label:
        similarities.append(max_reward)
        #print("in the condition", word_id, entity_type, similarities)
    elif word_ents and not is_banned_label:
        for ent in word_ents:
            if (entity_type in ALL_FILTER and ent.label_ == entity_type):
                similarities.append(max_reward)
            
            elif ent.kb_id_ == word_id:
                if entity_type == "location" and ent.label_ in LOCATION_FILTER:
                    similarities.append(max_reward)
                elif entity_type == "person" and ent.label_ in PERSON_FILTER:
                    similarities.append(max_reward)
                elif entity_type == "date" and (ent.label_ in DATE_FILTER):
                    similarities.append(max_reward)
                elif entity_type == "cause" and ent.label_ in CAUSE_FILTER:
                    similarities.append(max_reward)
                elif entity_type == "which" and ent.label_ in WHICH_FILTER:
                    similarities.append(max_reward)
                elif entity_type == "what" and ent.label_ in WHAT_FILTER:
                    similarities.append(max_reward)
                elif entity_type == "quantity" and ent.label_ in QUANTITY_FILTER:
                    similarities.append(max_reward)
                else:                     
                    similarities.append(get_similarity_by_words(get_nlp(word_label),get_nlp(entity_type)))
            else: similarities.append(get_similarity_by_words(get_nlp(word_label),get_nlp(entity_type)))
    else:
        similarities.append(get_similarity_by_words(get_nlp(word_label),get_nlp(entity_type)))
    #print("get_entity_similarity:",word_label, entity_type, similarities)
    return similarities

#get_entity_similarity("place of birth", "location", [], max_reward=2)


# In[68]:


#    "whats the name of the organization that was founded by  frei otto",
#    "What is the name of the writer of The Secret Garden?",
#    "Where did roger marquis die",
#    "Which genre of album is harder.....faster?",
#    "Which actor voiced the Unicorn in The Last Unicorn?",
#    "Who voiced the Unicorn in The Last Unicorn?",
#    "Which is the nation of Martha Mattox",
#    "Who is the wife of Barack Obama?",
#    "of what nationality is ken mcgoogan",
#    "which stadium do the wests tigers play in",
#    "Who is the author that wrote the book Moby Dick",
#    "Which equestrian was is in dublin ?",
#    "how does engelbert zaschka identify	",
#    "Who influenced michael mcdowell?",
#    "what does  2674 pandarus orbit"
# "Who is the author that wrote the book Moby Dick"

#question="Which actor voiced the Unicorn in The Last Unicorn?"
#verbose=True
#aggressive=False
#looped=False
#deep_k=5
#deep_k_step=1
#graph_size_min=100
#graph_size_max=5000
#timer=True
#g_paths=False
#get_graph=False
#cores=mp.cpu_count()
#
#
#if verbose: start_time = time.time()
#if timer: timer_time = time.time()
#if verbose: print("Auto correcting question:",question)
#q_nlp = get_nlp(question, autocorrect=True)
#if verbose: print("-> q_nlp:",q_nlp)
#q_themes = get_themes(q_nlp, top_k=2, online=True)
#q_theme_names = [q[0].text for q in q_themes[0]]
#if verbose: print("-> q_themes:",q_themes)
#q_themes_enhanced = get_enhanced_themes(q_themes, top_k=1, aggressive=aggressive)
#q_theme_enhanced_names = [q[0] for q in q_themes_enhanced]
#if verbose: print("-> q_themes_enhanced:",q_themes_enhanced)
#if verbose: print("--> Calculating predicates... (could be long.. depends on uncached unpure predicates)")
#q_predicates_db = get_predicates(q_nlp, q_themes, top_k=0)
#q_predicates_online = get_predicates_online(q_nlp, top_k=2, aggressive=aggressive)
#q_predicates = []
#q_predicates_db_ids = [p[1] for p in q_predicates_db]
#q_predicates_db_names = [p[0] for p in q_predicates_db]
#q_predicates_online_ids = [p[1] for p in q_predicates_online]
#q_predicates_online_names = [p[0] for p in q_predicates_online]
#for i_n,n in enumerate(q_predicates_db_names):
#    pn_online_text = [n.text for n in q_predicates_online_names]
#    tmp_ids = q_predicates_db_ids[i_n]
#    if n.text in pn_online_text:
#        for p_o in q_predicates_online_ids[pn_online_text.index(n.text)]:
#            if p_o not in tmp_ids:
#                tmp_ids.append(p_o)
#    q_predicates.append((n,tmp_ids))
#
#for i_n_o,n_o in enumerate(q_predicates_online_names):
#    n_db_text = [n.text for n in q_predicates_db_names]
#    if n_o.text not in n_db_text:
#        q_predicates.append((n_o, q_predicates_online_ids[i_n_o]))
#
#if verbose: print("-> q_predicates:",q_predicates)
#if timer: 
#    print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
#    timer_time = time.time()
#q_focused_parts = get_focused_parts(q_nlp, q_themes, top_k=2)
#if verbose: print("-> q_focused_parts:",q_focused_parts)
#if verbose: print("-> Building the graph with k_deep",str(deep_k),"... (could be long)")
#
#if deep_k<=1:
#    deep_k = 1
#    graph, predicates_dict = build_graph(q_nlp, q_themes, q_themes_enhanced, q_predicates, deep_k=deep_k, cores=cores)
#    if timer: 
#        print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
#        timer_time = time.time()
#else:
#    for k in range(1, deep_k, deep_k_step):
#        graph, predicates_dict = build_graph(q_nlp, q_themes, q_themes_enhanced, q_predicates, deep_k=deep_k, cores=cores)
#        if timer: 
#            print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
#            timer_time = time.time()
#
#        if verbose: print("--> ",len(graph), "nodes and", graph.size(), "edges")
#        if verbose: print("--> Removing meaningless subgraphs")
#        graph = filter_graph_by_names(graph, q_theme_names+q_theme_enhanced_names, entities=True, predicates=False)
#        if verbose: print("--> New graph of:",len(graph), "nodes and", graph.size(), "edges")
#
#        if graph.size() > graph_size_max or len(graph) > graph_size_max:
#            deep_k -= deep_k_step
#            if verbose: print("---> Rebuilding the graph with k_deep",str(deep_k), "... Previously:",len(graph), "nodes or", graph.size(), "edges was above the limit of",graph_size_max)
#        else: break
#
### Auto-scaling the number of nodes in graph
##previous_deep_k = deep_k
##if graph.size() <= graph_size_min:
##    for k in range(deep_k, 10, deep_k_step):
##        if verbose: print("---> Rebuilding the graph with k_deep",str(k), "... Previously:",len(graph), "nodes or", graph.size(), "edges was below the limit of",graph_size_min)
##        graph, predicates_dict = build_graph(q_nlp, q_themes, q_themes_enhanced, q_predicates, deep_k=k, cores=cores)
##        if timer: 
##            print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
##            timer_time = time.time()
#
#if get_graph: 
#    plot_graph(graph, "file_name_graph", "Graph_title")
#
##if graph.size() > 510 or len(graph) > 510:
##    if verbose: print("Stopping the computing here, too computational.")
##    return False
#if verbose: print("-> predicates_dict:",predicates_dict)
#paths_keywords = find_paths_keywords(graph, q_nlp, q_themes, q_themes_enhanced, q_predicates, q_focused_parts)
#if verbose: print("-> paths_keywords:",paths_keywords)
#if timer: timer_time = time.time()
#if verbose: print("-> Computing possible paths... (could be long)")
#if graph.size() < 500:
#    top_perf = graph.size()
#else:
##    top_perf = 500
#path_nodes = find_path_nodes_from_graph(nlp_question, graph, predicates_dict,paths_keywords, threshold=0.8,special_pred_theshold=0.7, thres_inter=0.1, top_performance=graph.size(), min_paths=100, cores=cores)
##for p in path_nodes:
##    print("path_nodes",path_nodes)
#if verbose: print("--> len(path_nodes):",len(path_nodes))
#if timer: 
#    print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
#    timer_time = time.time()
#
#if len(path_nodes) < 20000:
#    if verbose: print("-> Filtering paths... (could be long)")
#    paths_nodes_filtered = paths_nodes_filter(path_nodes, graph)
#    if verbose: print("--> len(paths_nodes_filtered):",len(paths_nodes_filtered))
#    if timer: 
#        print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
#        timer_time = time.time()
#else: 
#    if verbose: print("--> Skipping paths filtering... (too much paths)")
#    paths_nodes_filtered = path_nodes
#    
#    
#if verbose: print("-> Computing hypothesises...")
#hypothesises = get_hypothesises(q_nlp, q_predicates, q_themes, paths_keywords, paths_nodes_filtered, threshold=0.5, max_reward=2.0) 
#if verbose: print("\n\n--> test_hypothesises:",hypothesises)


# In[69]:


def get_hypothesises(nlp, predicates_dict, predicates, themes, paths_keywords, filtered_paths, threshold=0.5, special_pred_theshold=0.7, max_reward=2.0):#, themes, themes_enhanced):
    
    w_positions, w_names = w_converter(nlp)
    w_names_only = [wn[1] for wn in w_names]
    #print("w_positions",w_positions)
    #print("w_names",w_names)
    #print("w_names_only",w_names_only)
    
    date_trigger = "date" in w_names_only
    print("date_trigger",date_trigger)
    
    BANNED_PREDICATATE_IDS = ["P31"]
    complementary_predicates = paths_keywords[0]+[p[0] for p in list(paths_keywords[1].values())]

    nlp_time = get_nlp("time")
    nlp_date = get_nlp("date")
    
    #locate positions   
    anchors_positions = []
    anchors_focuses = []
    #keywords_positions = []
    #predicates_positions = []
    
    theme_keywords = [t[0] for t in themes[0]]
    
    predicate_ids = sum([p[1] for p in predicates if p[1]],[])
    predicate_names = [get_nlp(p[0].text) for p in predicates]
    #print("predicate_ids",predicate_ids)
    #print("predicate_names",predicate_names)
    
    
    
    [anchors_positions.append(i) for i, w in enumerate(nlp) if w in paths_keywords[2]]
    #print("\nanchors_positions:",anchors_positions)
    
    #anchors_childrens
    for p in anchors_positions:
        children = [c for c in nlp[p].children]
        if children == []: 
            children = [c for c in nlp[p].head.children]
        else: 
            if nlp[p].head:
                children.append(nlp[p].head)
        
        anchors_focuses += ([c for c in children
               if c not in [nlp[a] for a in anchors_positions]
               and c.pos_ != "PUNCT"])
        
        if not anchors_focuses:
            anchors_focuses = [nlp[p].head]
        
        anchors_focuses += complementary_predicates
        #print("\nanchors_focuses",anchors_focuses)
    
    anchors_focuses_filtered = []
    
    for af in anchors_focuses:
        if isinstance(af, str):
            anchors_focuses_filtered.append(af)
        else:
            anchors_focuses_filtered.append(af.text)
        
    anchors_focuses = []
    [anchors_focuses.append(af) for af in anchors_focuses_filtered if af not in anchors_focuses and af]
    #print("\nanchors_focuses",anchors_focuses)
    
    #find anchor position in paths
    anchors_predicates = []
    
    main_predicate_ids = []
    main_predicate_names = []
    [main_predicate_ids.append(p) for p in predicate_ids+sum([p[1] for p in list(paths_keywords[1].values())],[]) if p not in main_predicate_ids]
    #print("paths_keywords[1]",paths_keywords[1])
    #print("main_predicate_ids",main_predicate_ids)
    
    #print("[p[0] for p in list(paths_keywords[1].values())]",[p[0].text for p in list(paths_keywords[1].values())])
    [main_predicate_names.append(p) for p in predicate_names+[get_nlp(p[0].text) for p in list(paths_keywords[1].values())] if p not in main_predicate_names]
    #print("paths_keywords[1]",paths_keywords[1])
    #print("main_predicate_names",main_predicate_names)
    #
    #return 0
    
    for p in filtered_paths:
        p_len = len(p)
        for i_e, e in enumerate(p):
            if is_wd_predicate(e):
                #print("predicate",e)
                if main_predicate_ids:
                    if e in main_predicate_ids and e not in BANNED_PREDICATATE_IDS:
                        if e not in [ap[0] for ap in anchors_predicates]:
                            if date_trigger:
                                time_similarity = get_similarity_by_words(get_nlp(get_wd_label(e)),nlp_time)
                                date_similarity = get_similarity_by_words(get_nlp(get_wd_label(e)),nlp_date)
                                #print("main_predicate_ids", e, "time_similarity",time_similarity)
                                #print("main_predicate_ids", e, "date_similarity",date_similarity)
                                if time_similarity > date_similarity:
                                    anchors_predicates.append((e, time_similarity))
                                else: anchors_predicates.append((e, date_similarity))
                            else:
                                anchors_predicates.append((e, max_reward))
                    elif e not in [ap[0] for ap in anchors_predicates]:
                        stat_count = 0
                        stat_current = 0
                        for pn in main_predicate_names:
                            stat_current += get_similarity_by_words(get_nlp(get_wd_label(e)),pn)
                            stat_count += 1
                        for pi in main_predicate_ids:
                            stat_current += get_similarity_by_words(get_nlp(get_wd_label(e)),get_nlp(get_wd_label(pi)))
                            stat_count += 1
                        if date_trigger:
                            time_similarity = get_similarity_by_words(get_nlp(get_wd_label(e)),nlp_time)
                            date_similarity = get_similarity_by_words(get_nlp(get_wd_label(e)),nlp_date)
                            #print("if main_pred -> date_trigger -> elif e not",e)
                            #print("time_similarity",time_similarity)
                            #print("date_similarity",date_similarity)
                            if time_similarity > special_pred_theshold or date_similarity > special_pred_theshold:
                                if stat_count > 1: 
                                    stat_count -= 1
                                else: stat_count += 1
                                if time_similarity > date_similarity:
                                    anchors_predicates.append((e, time_similarity))
                                else: anchors_predicates.append((e, date_similarity))
                        anchors_predicates.append((e, stat_current/stat_count))
                        
                elif e not in [ap[0] for ap in anchors_predicates]:
                    stat_count = 0
                    stat_current = 0
                    for af in anchors_focuses:
                        stat_current += get_similarity_by_words(get_nlp(get_wd_label(e)),get_nlp(af))
                        stat_count += 1
                    if date_trigger:
                        time_similarity = get_similarity_by_words(get_nlp(get_wd_label(e)),nlp_time)
                        date_similarity = get_similarity_by_words(get_nlp(get_wd_label(e)),nlp_date)
                        #print("if not main_pred -> date_trigger -> elif e not",e)
                        #print("time_similarity",time_similarity)
                        #print("date_similarity",date_similarity)
                        if time_similarity > special_pred_theshold or date_similarity > special_pred_theshold:
                            if stat_count > 1:
                                stat_count -= 1
                            else: stat_count += 1
                            if time_similarity > date_similarity:
                                anchors_predicates.append((e, time_similarity))
                            else: anchors_predicates.append((e, time_similarity))
                    anchors_predicates.append((e, stat_current/stat_count))
            
    
    #print("filtered_paths",filtered_paths)
    #for p in filtered_paths:
    #    for af in anchors_focuses:
    #        #print(af, p)
    #        for e in p:
    #            #print(af,get_wd_label(e))
    #            if is_wd_predicate(e):# and e not in [ap[0] for ap in anchors_predicates]:
    #                #print(af,get_wd_label(e))
    #                anchors_predicates.append([e, get_similarity_by_words(get_nlp(get_wd_label(e)),get_nlp(af))])
                
    #print("\nanchors_predicates",anchors_predicates)
    
    anchors_predicates_filtered = []
    [anchors_predicates_filtered.append(ap) for ap in anchors_predicates if ap not in anchors_predicates_filtered]
    
    #print("\anchors_predicates_filtered",anchors_predicates_filtered)
    
    #anchors_predicates = [a for a in sorted(anchors_predicates_filtered, key=lambda x: x[-1], reverse=True) if a[1] > threshold]
    
    for thres in [e/100 for e in reversed(range(10, int(threshold*100)+10, 10))]:
        #print("anchors_predicates current thres",thres)
        anchors_predicates = [a for a in sorted(anchors_predicates_filtered, key=lambda x: x[-1], reverse=True) if a[1] > thres]
        if anchors_predicates:
            break
            
    
    #print("len(anchors_predicates sorted)",len(anchors_predicates))
    #print("anchors_predicates sorted",anchors_predicates)
    
    #anchors_predicates_filtered = []
    #for ap in anchors_predicates:
    #    for af in anchors_focuses:
    #        anchors_predicates_filtered.append([ap[0],get_similarity_by_words(get_nlp(get_wd_label(ap[0])),get_nlp(af))])
    #
    #anchors_predicates_filtered = [a for a in sorted(anchors_predicates_filtered, key=lambda x: x[-1], reverse=True) if a[1] > 0]
    
    #for thres in [e/100 for e in reversed(range(10, int(threshold*100)+10, 10))]:
    #    print("anchors_predicates_filtered current thres",thres)
    #    if not anchors_predicates_filtered:
    #        anchors_predicates_filtered = anchors_predicates
    #        break
    #    anchors_predicates_filtered = [a for a in sorted(anchors_predicates_filtered, key=lambda x: x[-1], reverse=True) if a[1] > thres]
    #    if len(anchors_predicates) > 10:
    #        break
    
    #print("len(anchors_predicates_filtered)",len(anchors_predicates_filtered))
    #print("anchors_predicates_filtered",anchors_predicates_filtered)
    #
    #anchors_predicates=[]
    #[anchors_predicates.append(apf) for apf in anchors_predicates_filtered if apf not in anchors_predicates]
    #print("len(anchors_predicates)",len(anchors_predicates))
    #print("anchors_predicates",anchors_predicates)
    
    tuples_unique_ids = []
    tuples_unique_predicate_ids = []
    
    hypothesises_tuples = []
    for ap in anchors_predicates:
        #print("ap",ap)
        for fp in filtered_paths:
            #if "Q4985" in fp:
            #    print("Q4985 in fp",fp, ap)
            for i, e in enumerate(fp):
                #print(e)
                if e == ap[0] and i>1 and i<len(fp)-1:
                    #print(i, [fp[i-1], fp[i], fp[i+1]])
                    hypothesis_tuple = [fp[i-1], fp[i], fp[i+1]]
                    if hypothesis_tuple not in hypothesises_tuples:
                        hypothesises_tuples.append(hypothesis_tuple)
                        if hypothesis_tuple[0] not in tuples_unique_ids:
                            tuples_unique_ids.append(hypothesis_tuple[0])
                        if hypothesis_tuple[1] not in tuples_unique_predicate_ids:
                            tuples_unique_predicate_ids.append(hypothesis_tuple[1])
                        if hypothesis_tuple[2] not in tuples_unique_ids:
                            tuples_unique_ids.append(hypothesis_tuple[2])
                        #if "Q4985" in hypothesis_tuple:
                        #    print("Q4985 hypothesis_tuple",hypothesis_tuple, ap,fp)
                        
    #print("tuples_unique_ids",tuples_unique_ids)
    #print("tuples_unique_predicate_ids",tuples_unique_predicate_ids)
    
    hypothesises_unique_ids = [t for t in tuples_unique_ids if get_wd_label(t).lower() not in anchors_focuses]
    if len(hypothesises_unique_ids)>0 and len(tuples_unique_ids)>0:
        max_reward *= len(hypothesises_unique_ids)/len(tuples_unique_ids)
    
    #print("hypothesises_unique_ids",hypothesises_unique_ids)
    
    #print("hypothesises_tuples",hypothesises_tuples)
    #print("hypothesises_tuples",hypothesises_tuples)
    #print([a[0] for a in anchors_predicates])

    #keywords_ids = [i for j in [get_wd_ids(k) for k in anchors_focuses if get_wd_ids(k)] for i in j]
    #print("anchors_focuses",keywords_ids)
    #print(extract_ids(themes[0]))
    #print(extract_ids(themes_enhanced))
    #keywords_ids = []
    #[keywords_ids.append(i) for i in extract_ids(themes[0]) + extract_ids(themes_enhanced) if i not in keywords_ids]
    #print("keywords_ids",keywords_ids)
    
    #print("anchors_predicates",anchors_predicates)
    
    #print("-------START FILTERING-------")
    
    hypothesises = []
    hypothesises_all = []
    hypothesises_tuples_len = len(hypothesises_tuples)
    
    keywords_similarity_threshold = 0.9
    
    tmp_to_find="" # for debugging, set the ID of the element to track in the log as print
    for ht in hypothesises_tuples:
        if tmp_to_find in ht: print("ht",ht)
        if ht[1] in [a[0] for a in anchors_predicates]:
            for i_af, af in enumerate(anchors_focuses):
                hypo_sum = 0
                nlp_af = get_nlp(af)
                nlp_ht0 = get_nlp(get_wd_label(ht[0]))
                nlp_ht2 = get_nlp(get_wd_label(ht[2]))
                if not nlp_ht2:
                    break

                af_lemma = ' '.join([e.lower_ for e in nlp_af if e.pos_ != "DET"])
                ht0_lemma = ' '.join([e.lower_ for e in nlp_ht0 if e.pos_ != "DET"])
                ht2_lemma = ' '.join([e.lower_ for e in nlp_ht2 if e.pos_ != "DET"])
                
                #if get_wd_label(ht[0]).lower() not in anchors_focuses and get_wd_label(ht[2]).lower() not in anchors_focuses:
                #    for es in get_entity_similarity(ht[0], wn[1], anchors_focuses, max_reward=max_reward):
                #        hypo_sum += es
                
                if (
                    nlp_af.text.lower() != nlp_ht2.text.lower() 
                    and af_lemma != nlp_ht2[0].text.lower()
                    and nlp_af.text.lower() != ht2_lemma
                    and af_lemma != ht2_lemma
                   ):
                    
                    if date_trigger:
                        if is_timestamp(ht[0]):
                            for es in get_entity_similarity(ht[0], "date", anchors_focuses, max_reward=max_reward):
                                hypo_sum += es
                                #print("if date hypo_sum",ht[0], "date",ht[0], es, hypo_sum)
                        
                        else: hypo_sum += get_similarity_by_words(nlp_ht2, nlp_af)
                    else: hypo_sum += get_similarity_by_words(nlp_ht2, nlp_af)
                    
                    if i_af in w_positions:
                        for wn in w_names:
                            if i_af == wn[0]:
                                for es in get_entity_similarity(ht[0], wn[1], anchors_focuses, max_reward=max_reward):
                                    hypo_sum += es
                                    if tmp_to_find in ht: print("if i_af hypo_sum","ht[0], wn[1], es",ht[0], wn[1], es,hypo_sum)
                    
                    ht0_sum = 0
                    ht2_sum = 0
                    
                    if is_timestamp(ht[0]): ht0_label = ht[0]
                    else: ht0_label = get_wd_label(ht[0]).lower()
                    if is_timestamp(ht[2]): ht2_label = ht[2]
                    else: ht2_label = get_wd_label(ht[2]).lower()
                    
                    for tk in theme_keywords:
                        if tmp_to_find in ht: print("tk",tk)
                        if tmp_to_find in ht: print("ht0_label",ht0_label)
                        if tmp_to_find in ht: print("ht2_label",ht2_label)
                        
                        nlp_tk = get_nlp(tk.text.lower())
                        ht0_label_similarity = get_nlp(ht0_label).similarity(nlp_tk)
                        ht2_label_similarity = get_nlp(ht2_label).similarity(nlp_tk)
                        
                        if tmp_to_find in ht: print("ht0_label_similarity",ht0_label_similarity)
                        if tmp_to_find in ht: print("ht2_label_similarity",ht2_label_similarity)
#
                        if ht0_label_similarity > keywords_similarity_threshold and ht[1] in main_predicate_ids:
                            if tmp_to_find in ht: print("ht0_label",ht0_label)
                            for wn in w_names_only:
                                for es in get_entity_similarity(ht[2], wn, anchors_focuses, max_reward=max_reward*3):
                                    if tmp_to_find in ht: print("ht0_sum main_predicate_ids before",ht0_sum)
                                    ht0_sum += es
                                    if tmp_to_find in ht: print("theme_keywords ht0_sum ht[2], wn, es",ht[2], wn, es, ht0_sum)
                                    if tmp_to_find in ht: print("ht0_label",ht2_label,es, ht0_sum, ht)
                        elif ht0_label_similarity > keywords_similarity_threshold:
                            if tmp_to_find in ht: print("ht0_label",ht0_label)
                            for wn in w_names_only:
                                for es in get_entity_similarity(ht[2], wn, anchors_focuses, max_reward=max_reward*2):
                                    if tmp_to_find in ht: print("ht0_sum before",ht0_sum)
                                    ht0_sum += es
                                    if tmp_to_find in ht: print("theme_keywords not main_predicate_ids ht0_sum ht[2], wn, es",ht[2], wn, es, ht0_sum)
                                    if tmp_to_find in ht: print("ht0_label",ht2_label,es, ht0_sum, ht)
#
                        if ht2_label_similarity > keywords_similarity_threshold and ht[1] in main_predicate_ids:
                            if tmp_to_find in ht: print("ht2_label",ht2_label)
                            for wn in w_names_only:
                                for es in get_entity_similarity(ht[0], wn, anchors_focuses, max_reward=max_reward*3):
                                    if tmp_to_find in ht: print("ht2_sum before",ht0_sum)
                                    ht2_sum += es
                                    if tmp_to_find in ht: print("theme_keywords main_predicate_ids ht2_sum ht[0], wn, es",ht[0], wn, es, ht2_sum)
                                    if tmp_to_find in ht: print("ht2_label",ht0_label,es, ht2_sum, ht)
                        elif ht2_label_similarity > keywords_similarity_threshold:
                            if tmp_to_find in ht: print("ht2_label",ht2_label)
                            for wn in w_names_only:
                                for es in get_entity_similarity(ht[0], wn, anchors_focuses, max_reward=max_reward*2):
                                    if tmp_to_find in ht: print("ht2_sum before",ht0_sum)
                                    ht2_sum += es
                                    if tmp_to_find in ht: print("theme_keywords not main_predicate_ids ht2_sum ht[0], wn, es",ht[0], wn, es, ht2_sum)
                                    if tmp_to_find in ht: print("ht2_label",ht0_label,es, ht2_sum, ht)
                                                       
                    for ap in anchors_predicates:
                        if ap[0] == ht[1]:
                            for wn in w_names_only:
                                for es in get_entity_similarity(ht[0], wn, anchors_focuses, max_reward=max_reward*2):
                                    ht0_sum += es
                                    if tmp_to_find in ht: print("anchors_predicates w_names_only ht0_sum ht[0], wn, es",ht[0], wn, es, ht0_sum)
                                for es in get_entity_similarity(ht[2], wn, anchors_focuses, max_reward=max_reward*2):
                                    ht2_sum += es
                                    if tmp_to_find in ht: print("anchors_predicates w_names_only ht2_sum ht[2], wn, es",ht[2], wn, es, ht2_sum)
                                    
                            for tk in theme_keywords:
                                if tmp_to_find in ht: print("anchors_predicates tk",tk)
                                if tmp_to_find in ht: print("anchors_predicates ht0_label",ht0_label)
                                if tmp_to_find in ht: print("anchors_predicates ht2_label",ht2_label)

                                nlp_tk = get_nlp(tk.text.lower())
                                ht0_label_similarity = get_nlp(ht0_label).similarity(nlp_tk)
                                ht2_label_similarity = get_nlp(ht2_label).similarity(nlp_tk)

                                if tmp_to_find in ht: print("anchors_predicates ht0_label_similarity",ht0_label_similarity)
                                if tmp_to_find in ht: print("anchors_predicates ht2_label_similarity",ht2_label_similarity)

                                if ht0_label_similarity > keywords_similarity_threshold and ht[1] in main_predicate_ids:
                                    if tmp_to_find in ht: print("anchors_predicates ht0_label",ht0_label)
                                    for wn in w_names_only:
                                        for es in get_entity_similarity(ht[2], wn, anchors_focuses, max_reward=max_reward*2):
                                            if tmp_to_find in ht: print("anchors_predicates ht0_sum main_predicate_ids before",ht0_sum)
                                            ht0_sum += es
                                            if tmp_to_find in ht: print("anchors_predicates theme_keywords ht0_sum ht[2], wn, es",ht[2], wn, es, ht0_sum)
                                            if tmp_to_find in ht: print("anchors_predicates ht0_label",ht2_label,es, ht0_sum, ht)
                                elif ht0_label_similarity > keywords_similarity_threshold:
                                    if tmp_to_find in ht: print("anchors_predicates ht0_label",ht0_label)
                                    for wn in w_names_only:
                                        for es in get_entity_similarity(ht[2], wn, anchors_focuses, max_reward=max_reward):
                                            if tmp_to_find in ht: print("anchors_predicates ht0_sum before",ht0_sum)
                                            ht0_sum += es
                                            if tmp_to_find in ht: print("anchors_predicates theme_keywords not main_predicate_ids ht0_sum ht[2], wn, es",ht[2], wn, es, ht0_sum)
                                            if tmp_to_find in ht: print("anchors_predicates ht0_label",ht2_label,es, ht0_sum, ht)
                                                
                                if ht2_label_similarity > keywords_similarity_threshold and ht[1] in main_predicate_ids:
                                    if tmp_to_find in ht: print("anchors_predicates ht2_label",ht2_label)
                                    for wn in w_names_only:
                                        for es in get_entity_similarity(ht[0], wn, anchors_focuses, max_reward=max_reward*2):
                                            if tmp_to_find in ht: print("anchors_predicates ht2_sum before",ht0_sum)
                                            ht2_sum += es
                                            if tmp_to_find in ht: print("anchors_predicates theme_keywords main_predicate_ids ht2_sum ht[0], wn, es",ht[0], wn, es, ht2_sum)
                                            if tmp_to_find in ht: print("anchors_predicates ht2_label",ht0_label,es, ht2_sum, ht)
                                elif ht2_label_similarity > keywords_similarity_threshold:
                                    if tmp_to_find in ht: print("anchors_predicates ht2_label",ht2_label)
                                    for wn in w_names_only:
                                        for es in get_entity_similarity(ht[0], wn, anchors_focuses, max_reward=max_reward):
                                            if tmp_to_find in ht: print("anchors_predicates ht2_sum before",ht0_sum)
                                            ht2_sum += es
                                            if tmp_to_find in ht: print("anchors_predicates theme_keywords not main_predicate_ids ht2_sum ht[0], wn, es",ht[0], wn, es, ht2_sum)
                                            if tmp_to_find in ht: print("anchors_predicates ht2_label",ht0_label,es, ht2_sum, ht)
                                
                                    
                            
                            if date_trigger and is_timestamp(ht0_label) and ht2_label in anchors_focuses:
                                hypo_sum += ht0_sum
                                #print("is_timestamp(ht0_label) hypo_sum", hypo_sum)
                            elif date_trigger and is_timestamp(ht2_label) and ht0_label in anchors_focuses:
                                hypo_sum += ht2_sum
                                #print("is_timestamp(ht2_label) hypo_sum", hypo_sum)
                            elif ht2_label in anchors_focuses and ht0_label not in anchors_focuses:
                                hypo_sum += ht2_sum
                                if tmp_to_find in ht: print("ht2_label hypo_sum in anchors_focuses", hypo_sum)
                            elif ht0_label in anchors_focuses and ht2_label not in anchors_focuses:
                                hypo_sum += ht0_sum
                                if tmp_to_find in ht: print("ht0_label hypo_sum in anchors_focuses", hypo_sum)
                            else:
                                hypo_sum += ht0_sum
                                hypo_sum += ht2_sum
                                if tmp_to_find in ht: print("else in anchors_focuses hypo_sum", hypo_sum)
                                
                                
                            if tmp_to_find in ht: print("hypo_sum",hypo_sum)
                            if tmp_to_find in ht: print("ap[1]",ap[1])
                            hypo_sum *= ap[1]
                            
                            if tmp_to_find in ht: print("ht[0], ht[2], hypo_sum",ht[0], ht[2], hypo_sum)
                            
                            #if get_wd_label(ht[0]).lower() in anchors_focuses:
                            #    if not i_af in w_positions: 
                            #        hypo_sum += abs(ap[1])
                            #    else: hypo_sum -= abs(ap[1])
                            
                                
                                    
                            #if ht[0] == "Q202725": print("hypo_sum",hypo_sum)
                            
                                        

                            #else: hypo_sum = ap[1]
                            #hypo_sum *= abs(ap[1])
                                                        
                            
                            #break
                            #print("ap",ap, "ht",ht, "hypo_sum",hypo_sum)
                            #print(ht)
                            #break
                            #hypo_sum = abs(hypo_sum)
                            #hypo_sum += abs(ap[1])
                            #hypo_sum += abs(ap[1])
                            #hypo_sum += ap[1]
                            #hypo_sum += abs(hypo_sum)
                            #hypo_sum *= abs(ap[1])
                            
                            
                            #hypo_sum = abs(hypo_sum)
                            #hypo_sum /= ap[1]
                            #hypo_sum -= ap[1]
                            #hypo_sum += hypo_sum/ap[1]
                    
                    #print("ht[0]",ht[0])
                    #print("ht[2]",ht[2])
                    
                    if (date_trigger and is_timestamp(ht[0]) 
                        and not is_timestamp(ht[2]) 
                        and is_in_list_by_similarity(get_wd_label(ht[2]).lower(), anchors_focuses, keywords_similarity_threshold)):
                        #print("is_timestamp(ht[0]")
                        hypo = ht[0]
                    
                    elif (date_trigger and is_timestamp(ht[2]) 
                        and not is_timestamp(ht[0]) 
                        and is_in_list_by_similarity(get_wd_label(ht[0]).lower(), anchors_focuses, keywords_similarity_threshold)):
                        #print("is_timestamp(ht[2]")
                        hypo = ht[2]
                    
                    elif date_trigger and is_timestamp(ht[0]) and is_timestamp(ht[2]): break
                    
                    #is_in_list_by_similarity("Moby Dick", ["moby-dick","star wars"],0.9)
                    
                    #elif get_wd_label(ht[0]).lower() in anchors_focuses:
                    #    #print("get_wd_label(ht[0]).lower()",get_wd_label(ht[0]).lower())
                    #    if not get_wd_label(ht[2]).lower() in anchors_focuses:
                    #        hypo = ht[2]
                    #    if get_wd_label(ht[2]).lower() in anchors_focuses:
                    #        break
                    
                    elif is_in_list_by_similarity(get_wd_label(ht[0]).lower(), anchors_focuses, keywords_similarity_threshold):
                        if not is_in_list_by_similarity(get_wd_label(ht[2]).lower(), anchors_focuses, keywords_similarity_threshold):
                            hypo = ht[2]
                        else: break
                            
                    elif not is_in_list_by_similarity(get_wd_label(ht[0]).lower(), anchors_focuses, keywords_similarity_threshold):
                        if is_in_list_by_similarity(get_wd_label(ht[2]).lower(), anchors_focuses, keywords_similarity_threshold):
                            hypo = ht[0]
                        else:
                            hypothesises_all.append(ht[0])
                            if not hypothesises: hypothesises.append([ht[0], hypo_sum])
                            else: 
                                if ht[0] in [h[0] for h in hypothesises]:
                                    for i, h in enumerate(hypothesises):
                                        if ht[0] == h[0]: hypothesises[i] = [ht[0], hypo_sum+hypothesises[i][1]]
                                else: hypothesises.append([ht[0], hypo_sum])
                            
                            hypo = ht[2]
                    
                    #elif not get_wd_label(ht[0]).lower() in anchors_focuses:
                    #    if get_wd_label(ht[2]).lower() in anchors_focuses:
                    #        hypo = ht[0]
                    #    if not get_wd_label(ht[2]).lower() in anchors_focuses:
                    #        hypothesises_all.append(ht[0])
                    #        if not hypothesises: hypothesises.append([ht[0], hypo_sum])
                    #        else: 
                    #            if ht[0] in [h[0] for h in hypothesises]:
                    #                for i, h in enumerate(hypothesises):
                    #                    if ht[0] == h[0]: hypothesises[i] = [ht[0], hypo_sum+hypothesises[i][1]]
                    #            else: hypothesises.append([ht[0], hypo_sum])
                    #        
                    #        #if "Q4985" in ht: print("Q4985 ALONE hypo and sum:", ht[0], hypo_sum)
                    #        hypo = ht[2]

                    else:
                        #print("BREAK", ht)
                        break
                    
                    #print("hypothesises",hypothesises)
                    #if "Q4985" in ht:
                    #    print("Q4985 hypo and sum:", hypo, hypo_sum)
                    
                    hypothesises_all.append(hypo)
                    if not hypothesises: hypothesises.append([hypo, hypo_sum])
                    else: 
                        if hypo in [h[0] for h in hypothesises]:
                            for i, h in enumerate(hypothesises):
                                if hypo == h[0]: hypothesises[i] = [hypo, hypo_sum+hypothesises[i][1]]
                        else: hypothesises.append([hypo, hypo_sum])
    
    #print("len(hypothesises_all)",len(hypothesises_all))
    for i_h, h in enumerate(hypothesises):
        h_sum = hypothesises_all.count(h[0])
        #print("h_sum",h_sum)
        #print("BEFORE: hypothesises[i_h][1]",hypothesises[i_h][1])
        hypothesises[i_h][1] = hypothesises[i_h][1]/h_sum
        #print("AFTER: hypothesises[i_h][1]",hypothesises[i_h][1])
    
    #print("hypothesises_all",hypothesises_all)
    
    hypothesises = sorted(hypothesises, key=lambda x: x[-1], reverse=True)
    
    return hypothesises

#if verbose: print("-> Computing hypothesises...")
#hypothesises = get_hypothesises(q_nlp, q_predicates, q_themes, paths_keywords, paths_nodes_filtered, threshold=0.5, max_reward=2.0) 
#if verbose: print("\n\n--> hypothesises:",hypothesises)
    
    


# In[70]:


def get_unique_hypo_paths(hypothesis, other_hypothesis, hypo_paths):
    BANNED_IDS = ["P31"]
    filtered_hypo_paths = []
    
    other_hypothesis = other_hypothesis[:]+[hypothesis]
    
    for hp in hypo_paths:
        #print("hp",hp)
        path_is_used = False
        len_hp = len(hp)
        if len_hp >= 3:
            #if "P31" in hp: print("hp",hp)
            for e in hp:
                if e in other_hypothesis:
                    e_index = hp.index(e)
                    for step_index in range(1, len_hp, 2): #len_hp-e_index
                        #print("step_index",step_index)
                        if e_index-step_index-2 >= 0:
                            if hp[e_index-step_index] == hp[e_index-step_index-2] and hp[e_index-step_index] not in BANNED_IDS:
                                #print(hp.index(e), hp)
                                #print("IN")
                                part_1 = hp[:e_index-step_index-0]
                                part_2 = hp[e_index-step_index-1:]

                                #print("hp[:",e_index-step_index-0,"]",part_1)
                                #print("hp[",e_index-step_index-1,":]",part_2)
                                #hp[e_index:]

                                sub_part_1 = None
                                sub_part_2 = None

                                if hypothesis in part_1:
                                    sub_part_1 = get_unique_hypo_paths(hypothesis, other_hypothesis, [part_1,[]])
                                if hypothesis in part_2:
                                    sub_part_2 = get_unique_hypo_paths(hypothesis, other_hypothesis, [part_2,[]])

                                if sub_part_1 != None:
                                    if sub_part_1:
                                        [filtered_hypo_paths.append(sp) for sp in sub_part_1]
                                    else:
                                        for e in hp:
                                            if hp.count(e) > 1 and e not in BANNED_IDS: flag_too_much=True
                                        if not flag_too_much: filtered_hypo_paths.append(part_1)

                                if sub_part_2 != None:
                                    if sub_part_2:
                                        [filtered_hypo_paths.append(sp) for sp in sub_part_2]
                                    else:
                                        for e in hp:
                                            if hp.count(e) > 1 and e not in BANNED_IDS: flag_too_much=True
                                        if not flag_too_much: filtered_hypo_paths.append(part_2)
                                        
                                path_is_used=True

                        else: break

                    for step_index in range(1, len_hp, 2):
                        #print("step_index",step_index)
                        if e_index+step_index+2 < len_hp:
                            if hp[e_index+step_index] == hp[e_index+step_index+2] and hp[e_index+step_index] not in BANNED_IDS:
                                #print(hp.index(e), hp)
                                part_1 = hp[:e_index+step_index+2]
                                part_2 = hp[e_index+step_index+1:]

                                #print("hp[:",e_index+step_index+2,"]",part_1)
                                #print("hp[",e_index+step_index+1,":]",part_2)

                                #print("part_1",part_1)
                                #print("part_2",part_2)

                                sub_part_1 = None
                                sub_part_2 = None

                                if hypothesis in part_1:
                                    sub_part_1 = get_unique_hypo_paths(hypothesis, other_hypothesis, [part_1,[]])
                                if hypothesis in part_2:
                                    sub_part_2 = get_unique_hypo_paths(hypothesis, other_hypothesis, [part_2,[]])

                                if sub_part_1 != None:
                                    if sub_part_1:
                                        [filtered_hypo_paths.append(sp) for sp in sub_part_1]
                                    else:
                                        for e in hp:
                                            if hp.count(e) > 1 and e not in BANNED_IDS: flag_too_much=True
                                        if not flag_too_much: filtered_hypo_paths.append(part_1)

                                if sub_part_2 != None:
                                    if sub_part_2:
                                        [filtered_hypo_paths.append(sp) for sp in sub_part_2]
                                    else:
                                        flag_too_much = False
                                        for e in hp:
                                            if hp.count(e) > 1 and e not in BANNED_IDS: flag_too_much=True
                                        if not flag_too_much: filtered_hypo_paths.append(part_2)
                                        
                                path_is_used=True

                        else: break
            
            if path_is_used == False:
                flag_too_much = False
                for e in hp:
                    if hp.count(e) > 1 and e not in BANNED_IDS: flag_too_much=True
                if not flag_too_much: filtered_hypo_paths.append(hp)
        #else:
        #    filtered_hypo_paths.append(hp) 
    
    return filtered_hypo_paths


# In[71]:


def match_hypothesises_worker(in_mp_queue, out_mp_queue):
    golden_paths = []
    sentinel = None
    
    for mpu in iter(in_mp_queue.get, sentinel):
    
    #for mp in mp_similarities_untagged:
        #print("AFTER mpu",mpu)
        for i_e, e in enumerate(mpu[1]):
            if i_e <= 1 or i_e >= len(mpu[1])-2:
                continue
            if not is_wd_entity(e):
                continue

            mp_e_statements = get_all_statements_of_entity(e)
            
            mp_predicate_tagging_index = mpu[1][i_e+1].find("-")
            if mp_predicate_tagging_index != -1:
                mp_predicate = mpu[1][i_e+1][:mp_predicate_tagging_index]
            else:
                mp_predicate = mpu[1][i_e+1]
            extended_paths = get_statements_by_id(mp_e_statements, e, mp_predicate, qualifier=False, statement_type="predicate")
            extended_paths_qualifier = get_statements_by_id(mp_e_statements, e, mp_predicate, qualifier=True, statement_type="qualifier_predicate")
            
            ep_predicate_tagging_index_plus_1 = mpu[1][i_e+1].find("-")
            if ep_predicate_tagging_index_plus_1 != -1:
                ep_predicate_plus_1 = mpu[1][i_e+1][:ep_predicate_tagging_index_plus_1]
            else:
                ep_predicate_plus_1 = mpu[1][i_e+1]

            ep_predicate_tagging_index_minus_1 = mpu[1][i_e-1].find("-")
            if ep_predicate_tagging_index_minus_1 != -1:
                ep_predicate_minus_1 = mpu[1][i_e-1][:ep_predicate_tagging_index_minus_1]
            else:
                ep_predicate_minus_1 = mpu[1][i_e-1]
                    
            for ep in extended_paths_qualifier:
                if (ep['entity']['id'] == mpu[1][i_e] and 
                    ep['predicate']['id'] == ep_predicate_minus_1 and
                    ep['object']['id'] == mpu[1][i_e-2] and
                    ep['qualifiers']):
                    for q in ep['qualifiers']:
                        if(q['qualifier_predicate']["id"] == ep_predicate_plus_1 and
                          q['qualifier_object']["id"] == mpu[1][i_e+2]):
                            if mpu[1] not in golden_paths:
                                golden_paths.append(mpu[1])

                if (ep['entity']['id'] == mpu[1][i_e+2] and 
                    ep['predicate']['id'] == ep_predicate_plus_1 and
                    ep['object']['id'] == mpu[1][i_e] and
                    ep['qualifiers']):
                    for q in ep['qualifiers']:
                        if(q['qualifier_predicate']["id"] == ep_predicate_minus_1 and
                          q['qualifier_object']["id"] == mpu[1][i_e-2]):
                            if mpu[1] not in golden_paths:
                                golden_paths.append(mpu[1])

            for ep in extended_paths:
                if (ep['entity']['id'] == mpu[1][i_e] and 
                    ep['predicate']['id'] == ep_predicate_minus_1 and
                    ep['object']['id'] == mpu[1][i_e-2] and
                    ep['qualifiers']):
                    for q in ep['qualifiers']:
                        if(q['qualifier_predicate']["id"] == ep_predicate_plus_1 and
                          q['qualifier_object']["id"] == mpu[1][i_e+2]):
                            if mpu[1] not in golden_paths:
                                golden_paths.append(mpu[1])

                if (ep['entity']['id'] == mpu[1][i_e+2] and 
                    ep['predicate']['id'] == ep_predicate_plus_1 and
                    ep['object']['id'] == mpu[1][i_e] and
                    ep['qualifiers']):
                    for q in ep['qualifiers']:
                        if(q['qualifier_predicate']["id"] == ep_predicate_minus_1 and
                          q['qualifier_object']["id"] == mpu[1][i_e-2]):
                            if mpu[1] not in golden_paths:
                                golden_paths.append(mpu[1])  

    
    out_mp_queue.put(golden_paths)


# In[72]:


# TODO / REDO
# is_wd_entity not taking care of timestamps

def list_by_n(l, i):
    list_n = []
    for j in range(0, len(l)+1, 1):
        tmp = l[j-i:i+j-i]
        if tmp:
            list_n.append(tmp)
    return list_n

def match_hypothesises(graph, question, themes, predicates, hypothesises, paths, threshold=0.8, max_reward=2.0, winner_threshold_diff=7, time_sensitive=False, cores=mp.cpu_count()):
    BANNED_IDS = ["P31"]
    LOCATION_FILTER = ["GPE", "FAC", "LOC","PERSON"]
    
    filtered_paths = []
    
    #print("hypothesises",hypothesises)  
    if time_sensitive:
        hypothesises_time = [h for h in hypothesises if is_timestamp(h[0])]
        if hypothesises_time:
            hypothesises = hypothesises_time
    
    #if 'where' in [t.lower_ for t in question if t.tag_=="WRB"]:
    #    print("in where condition")
    #    hypothesises_location = []
    #    for h in hypothesises:
    #        word_label = get_wd_label(h[0])
    #        word_ents = get_kb_ents(word_label)
    #        entities = [e.label_ for e in word_ents]
    #        [hypothesises_location.append(e) for e in entities if e in LOCATION_FILTER]
    #        print(h[0],word_label,word_ents,entities)
    #    
    #    if hypothesises_location:
    #        hypothesises = hypothesises_location
    
    
    #print(word_id,word_label,entity_type,)
        
    #print("hypothesises",hypothesises)  
    for h in hypothesises:
        other_hypothesis = [e[0] for e in hypothesises if e != h]
        hypo_paths = [p for p in paths if h[0] in p]

        filtered_hypo_paths = []
        [filtered_hypo_paths.append(p) for p in get_unique_hypo_paths(h[0], other_hypothesis, hypo_paths) if p not in filtered_hypo_paths]

        for p in filtered_hypo_paths:
            for e in p:
                if p.count(e) > 1 and e not in BANNED_IDS: 
                    continue
            if p[-1] == h[0]:
                reversed_path = list(reversed(p))
                if reversed_path not in filtered_paths:
                    filtered_paths.append(reversed_path)
            else:
                if p not in filtered_paths:
                    filtered_paths.append(p)
                    
    #print("filtered_paths",filtered_paths)
    
    #print("1 hypothesises",hypothesises)
    
    # check if first hypothesis is clear winner
    winner_threshold_diff = 2*max_reward # this is coded in hard ! TBD
    first_is_winner = False
    if len(hypothesises) > 1:
        hypo_diff = hypothesises[0][1]-hypothesises[1][1]
        print("hypo_diff",hypo_diff)
        if hypo_diff > winner_threshold_diff:
            first_is_winner = True 
    print("first_is_winner",first_is_winner)
    
    if first_is_winner:
        sorted_golden_paths = []

        if not sorted_golden_paths:
            for p in filtered_paths:
                if len(p)>1 and p[0] == hypothesises[0][0]:
                    if p not in sorted_golden_paths:
                        sorted_golden_paths.append(p)
                if len(p)>1 and p[-1] == hypothesises[0][0]:
                    p = list(reversed(p))
                    if p not in sorted_golden_paths:
                        sorted_golden_paths.append(p)

        if not sorted_golden_paths:
            for p in filtered_paths:
                if len(p)>1 and hypothesises[0][0] in p:
                    if p not in sorted_golden_paths:
                        sorted_golden_paths.append(p)
    
    else:    
        meaningful_paths = []

        w_positions, w_names = w_converter(question)

        theme_ids = sum([t[1] for t in themes[0]],[])
        for p in filtered_paths:
            counter = 0

            for ti in theme_ids:
                if ti in p and p not in meaningful_paths:
                    counter += 1
            for pred in [p[1] for p in predicates]:
                for e in pred:
                    if e in p:
                        counter += 1
                    else:
                        counter = 0

            for i_wp, wp in enumerate(w_positions):
                if w_names[i_wp][1] and wp<len(p):
                    for es in get_entity_similarity(p[wp], w_names[i_wp][1], [], max_reward=max_reward):
                        counter += es
                        #print("p[wp], w_names[i_wp][1], es",p[wp], w_names[i_wp][1], es)

            for hypo in hypothesises:
                if hypo[0] in p:
                    counter += 1
                if hypo[0] == p[0]:
                    counter += 1
                if hypo[0] == p[-1]:
                    counter += 1

            if counter > 0: meaningful_paths.append((counter, p))

        meaningful_paths = sorted(meaningful_paths, key=lambda x: x[0], reverse=True)

        #print("meaningful_paths",meaningful_paths)
        #print("len(meaningful_paths):",len(meaningful_paths))
        #print("\n")

        #looped_paths = []
        #for hypo in hypothesises:
        #    for mp in meaningful_paths:
        #        if mp[1][0] == hypo[0] or mp[1][-1] == hypo[0]:
        #            if graph.has_node(mp[1][0]) and graph.has_node(mp[1][-1]):
        #                path_tmp = list(nx.all_simple_paths(graph, mp[1][0],mp[1][-1]))
        #                if len(path_tmp)>1:
        #                    for p in path_tmp:
        #                        if p not in [lp[1] for lp in looped_paths]:
        #                            looped_paths.append((mp[0],p))
        #            #else:
        #            #    if not graph.has_node(mp[1][0]):
        #            #        print("MISSING NODE:", mp[1][0], get_wd_label(mp[1][0]))
        #            #    if not graph.has_node(mp[1][-1]):
        #            #        print("MISSING NODE:", mp[1][-1], get_wd_label(mp[1][-1]))
        #            
        ##print("len(looped_paths)", len(looped_paths))
        #print("looped_paths",looped_paths)

        #looped_paths_untagged = []
        #for lp in looped_paths:
        #    row_tmp = []
        #    for w in lp[1]:
        #        if w.find("-") > 0:
        #            row_tmp.append(w[:w.find("-")])
        #        else:
        #            row_tmp.append(w)
        #    looped_paths_untagged.append((lp[0],row_tmp))
        #
    #
        #    
        #print("looped_paths_untagged",looped_paths_untagged)

        looped_paths_untagged = meaningful_paths

        mp_similarities_untagged = []
        #mp_similarities_tagged = []
        mp_similarities_untagged_hypo = []
        #mp_similarities_tagged_hypo = []

        question_enhanced = []
        for q in question:
            if q.lemma_ == "where": question_enhanced.append("location")
            elif q.lemma_ == "when": question_enhanced.append("date")
            elif q.lemma_ == "who": question_enhanced.append("person")    
            elif q.lemma_ == "why": question_enhanced.append("cause")
            else: question_enhanced.append(q.text)

        question_enhanced = nlp(" ".join([q for q in question_enhanced]))

        #print("question",question)
        #print("question_enhanced",question_enhanced)

        #print("[h[0] for h in hypothesises]",[h[0] for h in hypothesises])

        for i_lp, lp in enumerate(looped_paths_untagged):
            #print(lp)
            sentence = get_nlp(" ".join([get_wd_label(w) for w in lp[1]]))
            similarity = get_similarity_by_words(sentence, question)
            similarity_enhanced = get_similarity_by_words(sentence, question_enhanced)
            similarity_avg = (similarity+similarity_enhanced)/2*lp[0]
            #print(sentence,question,question_enhanced)
            #print("similarity", similarity)
            #print("question_enhanced", similarity_enhanced)
            #mp_similarities_untagged.append((similarity_enhanced,lp[1]))
            #mp_similarities_tagged.append((similarity_enhanced,looped_paths[i_lp][1]))

            if lp[1][0] in [h[0] for h in hypothesises]:
                #print("lp[1][0]",lp[1][0])
                mp_similarities_untagged_hypo.append((similarity_avg, lp[1]))
                #mp_similarities_tagged_hypo.append((similarity_avg, looped_paths[i_lp][1]))

            mp_similarities_untagged.append((similarity_avg, lp[1]))
            #mp_similarities_tagged.append((similarity_avg, looped_paths[i_lp][1]))

        #print("mp_similarities_untagged",len(mp_similarities_untagged))
        #print("mp_similarities_untagged_hypo",len(mp_similarities_untagged_hypo))
        #print("mp_similarities_untagged",mp_similarities_untagged)

        #mp_similarities_tagged = sorted(mp_similarities_tagged, key=lambda x: x[0], reverse=True)
        #mp_similarities_tagged = [mp for mp in mp_similarities_tagged if mpu[0] > threshold]

        mp_similarities_untagged = sorted(mp_similarities_untagged, key=lambda x: x[0], reverse=True)
        mp_similarities_untagged = [mpu for mpu in mp_similarities_untagged if mpu[0] > threshold]

        #print("mp_similarities_untagged",len(mp_similarities_untagged))
        #print("mp_similarities_untagged",mp_similarities_untagged)

        [mp_similarities_untagged.append(suh) for suh in mp_similarities_untagged_hypo if not suh in mp_similarities_untagged]
        #[mp_similarities_tagged.append(sth) for sth in mp_similarities_tagged_hypo if not sth in mp_similarities_tagged]

        #print("mp_similarities_untagged",len(mp_similarities_untagged))
        #print("mp_similarities_tagged",len(mp_similarities_tagged))

        #WH_FILTER = ["WDT", "WP", "WP$", "WRB"]
        #wh_position = [w.i for w in question if w.tag_ in WH_FILTER][0]
        #question_list = [w.lower_ for w in question if not w.is_punct]
        #question_list_filtered = [w.lower_ for w in question if not w.is_punct and w.tag_ not in WH_FILTER]



        if cores <= 0: cores = 1
        sentinel = None

        out_mp_queue = mp.Queue()
        in_mp_queue = mp.Queue()

        for mpu in mp_similarities_untagged:
            #print("BEFORE mpu",mpu)
            in_mp_queue.put(mpu)

        procs = [mp.Process(target = match_hypothesises_worker, args = (in_mp_queue, out_mp_queue)) for i in range(cores)]

        golden_paths = []

        for proc in procs:
            proc.daemon = True
            proc.start()
        for proc in procs:    
            in_mp_queue.put(sentinel)
        for proc in procs:
            golden_paths.extend(out_mp_queue.get())
        for proc in procs:
            proc.join()

        golden_paths = [gp for gp in golden_paths if gp]
        #print("golden_paths",golden_paths)

        #print("len(golden_paths)",len(golden_paths))
        sorted_golden_paths = []
        for gp in golden_paths:
            tmp_gp = []
            #if gp[0] in [h[0] for h in hypothesises]:
            for e in gp:
                if is_wd_entity(e):
                    tmp_gp.append(get_wd_label(e))
                else:
                    tmp_gp.append(get_wd_label(e[:e.find("-")]))
            nlp_gp = get_nlp(" ".join(tmp_gp))
            sorted_golden_paths.append((get_similarity_by_words(question,nlp_gp), gp))
    #
        sorted_golden_paths = sorted(sorted_golden_paths, key=lambda x: x[0], reverse=True)

        #print("sorted_golden_paths",sorted_golden_paths)
        #print("len(sorted_golden_paths) BEFORE",len(sorted_golden_paths))

        sorted_golden_paths = [sgp[1] for sgp in sorted_golden_paths]

        if not sorted_golden_paths: 
            for hypo in hypothesises:
                #print("hypo",hypo[0])
                for lp in [lp[1] for lp in meaningful_paths]:
                    if len(lp)>1 and lp[0] == hypo[0]:
                        if lp not in sorted_golden_paths:
                            sorted_golden_paths.append(lp)
                    if len(lp)>1 and lp[-1] == hypo[0]:
                        lp = list(reversed(lp))
                        if lp not in sorted_golden_paths:
                            sorted_golden_paths.append(lp)
                if len(sorted_golden_paths) >= 1: break


        #print("len(sorted_golden_paths) AFTER",len(sorted_golden_paths))

        if not sorted_golden_paths:
            for hypo in hypothesises:
                for p in filtered_paths:
                    if len(p)>1 and p[0] == hypo[0]:
                        #print(p)
                        if p not in sorted_golden_paths:
                            sorted_golden_paths.append(p)
                    if len(p)>1 and p[-1] == hypo[0]:
                        p = list(reversed(p))
                        if p not in sorted_golden_paths:
                            sorted_golden_paths.append(p)
                if len(sorted_golden_paths) >= 1: break

        #print("len(sorted_golden_paths) AFTER AFTER",len(sorted_golden_paths))

        if not sorted_golden_paths:
            for hypo in hypothesises:
                for p in filtered_paths:
                    if len(p)>1 and hypo[0] in p:
                        if p not in sorted_golden_paths:
                            sorted_golden_paths.append(p)
                if len(sorted_golden_paths) >= 1: break

        #print("len(sorted_golden_paths) AFTER AFTER AFTER",len(sorted_golden_paths))
    
    golden_paths_filtered = []
    for gp in sorted_golden_paths:
        tmp_path = []
        for i_e, e in enumerate(gp):
            if i_e < len(gp)-2 and not is_wd_entity(e):
                if e == gp[i_e+2]:
                    golden_paths_filtered.append(gp[:gp.index(e)+2])
                    break
                else:
                    tmp_path.append(e)
            else:
                tmp_path.append(e)
        
        if tmp_path:
            for i_e, e in enumerate(tmp_path):
                if is_wd_entity(e):
                    if tmp_path.count(e) > 1:
                        pass
                    else:
                        if tmp_path not in golden_paths_filtered:
                            golden_paths_filtered.append(tmp_path)
                            
    #print("len(golden_paths_filtered)",len(golden_paths_filtered))
    
    #print("golden_paths_filtered",golden_paths_filtered)
    
    golden_unique_paths = golden_paths_filtered.copy()
    for i_sgp, sgp in enumerate(golden_paths_filtered):
        for sgp_2 in golden_paths_filtered:
            if (is_sublist(sgp, sgp_2) and sgp!=sgp_2):
                golden_unique_paths[i_sgp] = []
                break
    
    golden_unique_paths = [gup for gup in golden_unique_paths if gup]
    hypothesises_names = [h[0] for h in hypothesises]
    
    #print("golden_unique_paths",golden_unique_paths)
    #print("before hypothesises_names",hypothesises_names)
    #print("golden_unique_paths[0][0]",golden_unique_paths[0][0])
    #print("hypothesises_names",hypothesises_names)
    
    #if is_valide_wd_id(hypothesises_names[0]):
    
    if not first_is_winner:
        if golden_unique_paths and hypothesises_names:
            if golden_unique_paths[0] and hypothesises_names[0]:
                if golden_unique_paths[0][0]:
                    if (not is_wd_entity(hypothesises_names[0])
                        and is_wd_entity(golden_unique_paths[0][0]) 
                        or hypothesises_names[0] != golden_unique_paths[0][0]):
                        if golden_unique_paths[0][0] in hypothesises_names:
                            #hypothesises_names.pop(hypothesises_names.index(golden_unique_paths[0][0]))
                            hypothesises_names.insert(0,golden_unique_paths[0][0])
        
        golden_unique_paths = [gup for gup in golden_unique_paths if gup[0] == hypothesises_names[0]]
    
    #print("after hypothesises_names",hypothesises_names)
            
    
            
    #elif hypothesises_names[0] != golden_unique_paths[0][0]
    
    golden_unique_paths = [hypothesises_names]+golden_unique_paths
    
    return golden_unique_paths



#if verbose: print("-> Matching hypothesises...")
#start_time = time.time()
#golden_paths = match_hypothesises(graph, q_nlp, q_themes, q_predicates, hypothesises, paths_nodes_filtered, threshold=0.8, max_reward=2.0)
#end_time = time.time()
#print("Golden paths ->\tRunning time is {}s".format(round(end_time-start_time,2)))
#print(golden_paths)


# In[75]:


## questions = ("what was the cause of death of yves klein",
#               "Who is the wife of Barack Obama?",
#               "Who is the president of the United States?",
#               "When was produced the first Matrix movie?",
#               "Who made the soundtrack of the The Last Unicorn movie?",
#               "Who is the author of Le Petit Prince?",
#               "Which actor voiced the Unicorn in The Last Unicorn?",
#               "how is called the rabbit in Alice in Wonderland?"
#              )

#def print_running_time(start_time, end_time=time.time()):
#    print("->\tRunning time is {}s".format(round(end_time-start_time,2)))

def answer_question(question, verbose=False, aggressive=True, looped=False, deep_k=2, deep_k_step=1, deep_k_max=20,graph_size_min=100, graph_size_max=350, timer=False, g_paths=True, show_graph=False, cores=mp.cpu_count(), banning_str=False, reload_cache=False):
    if verbose: start_time = time.time()
    if timer: timer_time = time.time()
    if verbose: print("User input:",question)
    if verbose: print("--> Auto correcting question in progress...")
    q_nlp = get_nlp(question, autocorrect=True, banning_str=banning_str)
    if verbose: print("-> Auto corrected q_nlp:",q_nlp)
    time_sensitive = False
    if 'when' in [t.lower_ for t in q_nlp if t.tag_=="WRB"]: time_sensitive = True
    q_themes = get_themes(q_nlp, question, top_k=2, online=True)
    q_theme_names = [q[0].text for q in q_themes[0]]
    if verbose: print("-> q_themes:",q_themes)
    q_themes_enhanced = get_enhanced_themes(q_themes, top_k=1, aggressive=True)
    q_theme_enhanced_names = [q[0] for q in q_themes_enhanced]
    if verbose: print("-> q_themes_enhanced:",q_themes_enhanced)
    if verbose: print("--> Calculating predicates... (could be long.. depends on uncached unpure predicates)")
    q_predicates_db = get_predicates(q_nlp, q_themes, top_k=0)
    q_predicates_online = get_predicates_online(q_nlp, top_k=2, aggressive=aggressive)
    q_predicates = []
    q_predicates_db_ids = [p[1] for p in q_predicates_db]
    q_predicates_db_names = [p[0] for p in q_predicates_db]
    q_predicates_online_ids = [p[1] for p in q_predicates_online]
    q_predicates_online_names = [p[0] for p in q_predicates_online]
    for i_n,n in enumerate(q_predicates_db_names):
        pn_online_text = [n.text for n in q_predicates_online_names]
        tmp_ids = q_predicates_db_ids[i_n]
        if n.text in pn_online_text:
            for p_o in q_predicates_online_ids[pn_online_text.index(n.text)]:
                if p_o not in tmp_ids:
                    tmp_ids.append(p_o)
        q_predicates.append((n,tmp_ids))
    
    for i_n_o,n_o in enumerate(q_predicates_online_names):
        n_db_text = [n.text for n in q_predicates_db_names]
        if n_o.text not in n_db_text:
            q_predicates.append((n_o, q_predicates_online_ids[i_n_o]))
    
    if verbose: print("-> q_predicates:",q_predicates)
    if timer: 
        print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
        timer_time = time.time()
    q_focused_parts = get_focused_parts(q_nlp, q_themes, top_k=2)
    if verbose: print("-> q_focused_parts:",q_focused_parts)
    if verbose: print("-> Building the graph with k_deep",str(deep_k),"... (could be long)")
    
    # Auto-scaling the graph size with deepness
    if deep_k > deep_k_max and looped:
        deep_k_max+=int(deep_k_max/2)
    
    previous_graph_size = 0
    previous_graph_len = 0
    if deep_k<1:
        deep_k = 1
        graph, predicates_dict = build_graph(q_nlp, q_themes, q_themes_enhanced, q_predicates, deep_k=deep_k, time_sensitive=time_sensitive, cores=cores)
        if verbose: print("--> ",len(graph), "nodes and", graph.size(), "edges")
        if verbose: print("--> Removing meaningless subgraphs")
        graph = filter_graph_by_names(graph, q_theme_names+q_theme_enhanced_names, entities=True, predicates=False)
        if verbose: print("--> New graph of:",len(graph), "nodes and", graph.size(), "edges")
        if timer: 
            print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
            timer_time = time.time()
    else:
        if deep_k > deep_k_max:
            graph, predicates_dict = build_graph(q_nlp, q_themes, q_themes_enhanced, q_predicates, deep_k=deep_k, time_sensitive=time_sensitive, cores=cores)
            if verbose: print("---> deep_k > deep_k_max, running graph as last trial with deep_k:",deep_k)
            if timer: 
                print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
                timer_time = time.time()
            if verbose: print("--> Removing meaningless subgraphs")
            graph = filter_graph_by_names(graph, q_theme_names+q_theme_enhanced_names, entities=True, predicates=False)
            if verbose: print("--> New graph of:",len(graph), "nodes and", graph.size(), "edges")
                
        else:
            for k in range(deep_k, deep_k_max, deep_k_step):
                graph, predicates_dict = build_graph(q_nlp, q_themes, q_themes_enhanced, q_predicates, deep_k=deep_k, time_sensitive=time_sensitive, cores=cores)
                if timer: 
                    print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
                    timer_time = time.time()

                if verbose: print("--> ",len(graph), "nodes and", graph.size(), "edges")
                if verbose: print("--> Removing meaningless subgraphs")
                graph = filter_graph_by_names(graph, q_theme_names+q_theme_enhanced_names, entities=True, predicates=False)
                if verbose: print("--> New graph of:",len(graph), "nodes and", graph.size(), "edges")

                if previous_graph_size == graph.size() and previous_graph_len == len(graph):
                    if verbose: print("---> Loop detected, returning the graph in the current state")
                    break
                else:
                    previous_graph_size = graph.size()
                    previous_graph_len = len(graph)

                if (graph.size() > graph_size_max or len(graph) > graph_size_max) and deep_k > 1:
                    deep_k -= deep_k_step
                    if verbose: print("---> Rebuilding the graph with k_deep",str(deep_k), "... Previously:",len(graph), "nodes or", graph.size(), "edges was above the limit of",graph_size_max)
                    graph, predicates_dict = build_graph(q_nlp, q_themes, q_themes_enhanced, q_predicates, deep_k=deep_k, time_sensitive=time_sensitive, cores=cores)
                    break
                elif graph.size() <= graph_size_min or len(graph) <= graph_size_min:
                    if graph.size() < graph_size_min/3 or len(graph) < graph_size_min/3:
                        deep_k += deep_k_step*3
                    elif graph.size() < graph_size_min/4*3 or len(graph) < graph_size_min/4*3:
                        deep_k += deep_k_step*2
                    else:
                        deep_k += deep_k_step
                    if verbose: print("---> Rebuilding the graph with k_deep",str(deep_k), "... Previously:",len(graph), "nodes or", graph.size(), "edges was below the limit of",graph_size_min)
                else: break

    if show_graph:
        if verbose: print("---> Ploting the graph")
        plot_graph(graph, "file_name_graph", "Graph_title")
    
    if verbose: print("-> predicates_dict:",predicates_dict)
    paths_keywords = find_paths_keywords(graph, q_nlp, q_themes, q_themes_enhanced, q_predicates, q_focused_parts)
    if verbose: print("-> paths_keywords:",paths_keywords)
    if timer: timer_time = time.time()
    
    if verbose: print("-> Computing possible paths... (could be long)")
    path_nodes = find_path_nodes_from_graph(q_nlp, graph, predicates_dict, paths_keywords, threshold=0.8,special_pred_theshold=0.7, thres_inter=0.1, top_performance=len(graph), min_paths=100, cores=cores)
    if verbose: print("--> len(path_nodes):",len(path_nodes))
    if timer: 
        print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
        timer_time = time.time()
    
    if len(path_nodes) < 20000:
        if verbose: print("-> Filtering paths... (could be long)")
        paths_nodes_filtered = paths_nodes_filter(path_nodes, graph)
        if verbose: print("--> len(paths_nodes_filtered):",len(paths_nodes_filtered))
        if timer: 
            print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
            timer_time = time.time()
    else: 
        if verbose: print("--> Skipping paths filtering... (too much paths)")
        paths_nodes_filtered = paths_nodes_filter(path_nodes, graph, with_sublists=False)

    if verbose: print("-> Computing hypothesises...")
    hypothesises = get_hypothesises(q_nlp, predicates_dict, q_predicates, q_themes, paths_keywords, paths_nodes_filtered, threshold=0.5, max_reward=2.0) 
    if verbose: print("--> hypothesises:",hypothesises)
    if timer: 
        print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
        timer_time = time.time()
    if g_paths:
        if hypothesises:
            if verbose: print("-> Computing golden paths...")
            golden_paths = match_hypothesises(graph, q_nlp, q_themes, q_predicates, hypothesises, paths_nodes_filtered, threshold=0.8, max_reward=2.0,winner_threshold_diff=4.0, time_sensitive=time_sensitive)
            if verbose: print("--> len(golden_paths):",len(golden_paths)-1)
            if timer: 
                print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
                timer_time = time.time()
        else:
            if not looped:
                if verbose: print("-> Looping on aggressive mode...")
                golden_paths = answer_question(question, verbose=verbose, aggressive=True, looped=True, deep_k=deep_k, deep_k_step=deep_k_step, deep_k_max=deep_k_max,graph_size_min=graph_size_min, graph_size_max=graph_size_max, timer=timer, g_paths=g_paths, show_graph=show_graph, cores=cores, banning_str=banning_str, reload_cache=reload_cache)
                 
            else: 
                if verbose: print("--> End of loop")
                golden_paths=[]
    
    save_cache_data()
    
    if g_paths:
        if golden_paths:
            cleared_golden_paths = [golden_paths[0].copy()]
            for p in golden_paths[1:]:
                tmp_labeling = []
                for e in p:
                    tmp_labeling.append(get_wd_label(e))
                if tmp_labeling not in cleared_golden_paths:
                    cleared_golden_paths.append(tmp_labeling)

            if verbose: print("--> len(cleared_golden_paths):",len(cleared_golden_paths)-1)
            if len(cleared_golden_paths) > 1:
                if verbose: print("---> First path:",cleared_golden_paths[1])
            if timer: timer_time = time.time()
            
    if verbose: print("->\tTotal Running time is {}s".format(round(time.time()-start_time,2)))
    
    if g_paths:
        if golden_paths:
            return cleared_golden_paths
        else: return False
    else:
        if hypothesises:
            return [[a[0] for a in hypothesises]] + [[hypothesises[0][0]]]
        else:
            return False


#answer = answer_question("what film is by the writer phil hay?", verbose=True, timer=True) #444.36s
#answer = answer_question("When was produced the first Matrix movie?", verbose=True, timer=True) #70.67s
#answer = answer_question("Which actor voiced the Unicorn in The Last Unicorn?", verbose=True, timer=True, g_paths=True, show_graph=True) #works 312.12s
#answer = answer_question("Who voiced the Unicorn in The Last Unicorn?", verbose=True, timer=True) #works 323.52s
#answer = answer_question("How many actors voiced the Unicorn in The Last Unicorn?", verbose=True, timer=True) #592.22s 
#answer = answer_question("Which is the nation of Martha Mattox", verbose=True, timer=True) #97.89s
#answer = answer_question("Who made the soundtrack of the The Last Unicorn movie?", verbose=True, timer=True)
#answer = answer_question("Who is the author of Le Petit Prince?", verbose=True, timer=True)

#answer = answer_question("When was produced the first Matrix movie?", verbose=True, timer=True)
#answer = answer_question("Who is the president of the United States?", verbose=True, timer=True) #node Q76 not in graph 324.88s
#answer = answer_question("Who is the wife of Barack Obama?", verbose=True, timer=True) #works 275.94s
#answer = answer_question("what was the cause of death of yves klein", verbose=True, timer=True) #309.06s
#answer = answer_question("what city was alex golfis born in", verbose=True, timer=True)
#answer = answer_question("which stadium do the wests tigers play in", verbose=True, timer=True) #462.47s
#answer = answer_question("lol", verbose=True, timer=True)
#answer = answer_question("what's akbar tandjung's ethnicity", verbose=True, timer=True)

#answer = answer_question("Which equestrian was is in dublin ?", verbose=True, timer=True)
#answer = answer_question("how does engelbert zaschka identify	", verbose=True, timer=True)
#answer = answer_question("Who influenced michael mcdowell?", verbose=True, timer=True)
#answer = answer_question("what does  2674 pandarus orbit", verbose=True, timer=True)
#answer = answer_question("what production company was involved in smokin' aces 2: assasins' ball", verbose=True, timer=True)
#answer = answer_question("who's a kung fu star from hong kong", verbose=True, timer=True)

#answer = answer_question("Which genre of album is harder.....faster?", verbose=True, timer=True)
#answer = answer_question("Which equestrian was born in dublin?", verbose=True, timer=True)
#answer = answer_question("Who is the author that wrote the book Moby Dick", verbose=True, timer=True, show_graph=True) #314.04s works
#answer = answer_question("Name a person who died from bleeding.", verbose=True, timer=True) # 117.35s
#answer = answer_question("What is the name of the person who created Saved by the Bell?", verbose=True, timer=True)
#answer = answer_question("of what nationality is ken mcgoogan", verbose=True, timer=True) #works 51.39s
#
#answer = answer_question("What is a tv action show?", verbose=True, timer=True, g_paths=False)
##answer = answer_question("who published neo contra", verbose=True, timer=True, g_paths=False)
#answer = answer_question("When was the publication date of the movie Grease?", verbose=True, timer=True)
#answer = answer_question("When did the movie Grease come out?", verbose=True, timer=True, show_graph=True)
#answer = answer_question("whats the name of the organization that was founded by  frei otto", verbose=True, timer=True, g_paths=False)
#answer = answer_question("where was johannes messenius born", verbose=True, timer=True, g_paths=False)
#answer = answer_question("What is a type of gameplay available to gamers playing custom robo v2", verbose=True, timer=True, g_paths=False)
#answer = answer_question("Which genre of album is Harder ... Faster?", verbose=True, timer=True, show_graph=True)
#answer = answer_question("Who is the author that wrote the book Moby Dick", verbose=True, timer=True, show_graph=True)
#answer = answer_question("how does engelbert zaschka identify", verbose=True, timer=True, show_graph=True)

#answer = answer_question("where was shigeyasu suzuki's place of birth", verbose=True, timer=True, show_graph=True)
#answer = answer_question("What is the name of the writer of The Secret Garden?", verbose=True, timer=True, show_graph=True)
#answer = answer_question("Who was an influential figure for miško Šuvaković", verbose=True, timer=True, show_graph=True)
#
#answer = answer_question("When did the movie Grease come out?", verbose=True, timer=True, show_graph=True)

#answer = answer_question("of what nationality is ken mcgoogan", verbose=True, timer=True) #works 51.39s
#answer = answer_question("Where did roger marquis die", verbose=True, timer=True, show_graph=True) # works 64.56s

#answer = answer_question("How many people were in The Beatles?", verbose=True, timer=True, show_graph=True)
#if answer:
#    print("Answer:",get_wd_label(answer[0][0]), "("+str(answer[0][0])+")")
#    #print("Paths:",[[get_wd_label(e) for e in row] for row in answer[1:]])


#graph = answer_question("When did the movie Grease come out?", verbose=True, timer=True, g_paths=False, get_graph=True)

#graph = answer_question("When was the publication date of the movie Grease?", verbose=True, timer=True, g_paths=False)

#graph = answer_question("Which actor voiced the Unicorn in The Last Unicorn?", verbose=True, timer=True, g_paths=False, get_graph=True)

#graph = answer_question("whats the name of the organization that was founded by  frei otto", verbose=True, timer=True, g_paths=False, get_graph=True)


# In[ ]:


#questions = [
#    "When was the publication date of the movie Grease?",
#    "When was produced the first Matrix movie?",
#    
#    "Which is the nation of Martha Mattox",
#    "Where did roger marquis die",
#    "Who is the author that wrote the book Moby Dick",
#    "Who is the wife of Barack Obama?",
#    "of what nationality is ken mcgoogan",
#    
#    "What is the name of the writer of The Secret Garden?",
#    "whats the name of the organization that was founded by  frei otto",
#    
#    "Which genre of album is harder.....faster?",
#    "Which genre of album is Harder ... Faster?",
#    "Which actor voiced the Unicorn in The Last Unicorn?",
#    "Who voiced the Unicorn in The Last Unicorn?",
#    
#    "When did the movie Grease come out?",
#    
#    "which stadium do the wests tigers play in",
#    
#    "Which equestrian was is in dublin ?",
#    "how does engelbert zaschka identify	",
#    "Who influenced michael mcdowell?",
#    "what does  2674 pandarus orbit"
#            ]
#
#for i_q, question in enumerate(questions):
#    if i_q >= 0:
#        answer = answer_question(question, verbose=True, timer=True, show_graph=True)
#        if answer:
#            print("Answer:",get_wd_label(answer[0][0]), "("+str(answer[0][0])+")\n")
#    


# In[ ]:


#answer


# In[ ]:




#to_translate = ['Q13133', 'P26', 'Q76', 'P31', 'Q5', 'P31', 'Q24039104', 'P21', 'Q6581072', 'P1552', 'Q188830', 'P26', 'Q18531596']
#to_translate = ['Q202725', 'P725', 'Q176198', 'P453', 'Q30060419', 'P31', 'Q30167264', 'P1889', 'Q7246', 'P138', 'Q18356448']
#
#masked = []
#for tt in to_translate:
#    masked.append(get_wd_label(tt))
#    masked.append("[MASK]")
#print("marked",masked)
#
#print("->",[get_wd_label(e) for e in to_translate])
#print("-->"," ".join([get_wd_label(e) for e in to_translate]))
#
#print("-->"," [MASK] ".join([get_wd_label(e) for e in to_translate]))
#print("-->","[" +" , [MASK] , ".join([get_wd_label(e) for e in to_translate])+"]")
#
#FILTER_ELEMENTS = ['P31']
#filtered_by_elements = ["[MASK]" if x in FILTER_ELEMENTS else get_wd_label(x) for x in to_translate]
#
#print("-->"," [MASK] ".join([e for e in filtered_by_elements]))


# In[ ]:


#graph_2.has_node("Q13133")


# In[ ]:


#get_similarity_by_words(get_nlp("mia farrow"), get_nlp("farrow mia")) #1.000000077374981
#get_similarity_by_words(get_nlp("actor voiced"), get_nlp("voice actor")) #0.8541489425987572 
#get_similarity_by_words(get_nlp("actor voiced the unicorn in the last unicorn"), 
#                        get_nlp("the unicorn last unicorn actor voiced")) #0.9573255410217848
#get_similarity_by_words(get_nlp("voice actor"),get_nlp("instance of")) #0.30931508860569823
#get_similarity_by_words(get_nlp("voice actor"),get_nlp("present in work")) #0.34966764303274056
#get_similarity_by_words(get_nlp("voice actor"),get_nlp("subject has role")) #0.5026860362728758

#get_similarity_by_words(get_nlp("voice actor"),get_nlp("protagonist")) #0.4688377364169893


# In[ ]:


#subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
#print(len(subgraphs))
#[len(s.nodes) for s in subgraphs]
#len(subgraphs[0].nodes)


# In[ ]:


#for path in nx.all_simple_paths(graph, source="Q176198", target="Q202725"):
#    print(path)


# In[ ]:


#nx.shortest_path(graph, source="Q176198", target="Q202725")


# In[ ]:


#nlp_lookup_test = get_nlp("klein yves")
#[y['name'] for x,y in graph.nodes(data=True) if get_nlp(y['name']).similarity(nlp_lookup_test) >= 0.9]


# In[ ]:


#list(nx.dfs_labeled_edges(graph, source=get_themes(q0_nlp, top_k=3)[0][0][1][0], depth_limit=4))[0]


# In[ ]:


#plot_graph(graph_2, "test_file_name_graph", "Graph_title")


# In[ ]:


#nlp_lookup_test = get_nlp("klein yves")
#[y['name'] for x,y in graph.nodes(data=True) if get_nlp(y['name']).similarity(nlp_lookup_test) >= 0.9]
#[y['name'] for x,y in graph_2.nodes(data=True) if y['type'] == 'predicate']


# In[ ]:


#get_nlp(get_wd_label("Q13133")).similarity(get_nlp("person"))


# In[ ]:


#get_similarity_by_words(get_nlp("PERSON"),get_nlp("person"))


# In[ ]:


#get_similarity_by_words(get_nlp("GPE"),get_nlp("location"))


# In[ ]:


#get_most_similar("Michelle Obama", top_k=5)


# In[ ]:


#doc_ents_tmp = get_kb_ents("Michelle Obama")
#for ent in doc_ents_tmp:
#    print(" ".join(["ent", ent.text, ent.label_, ent.kb_id_]))
#
#doc_ents_tmp = get_kb_ents("New York")
#for ent in doc_ents_tmp:
#    print(" ".join(["ent", ent.text, ent.label_, ent.kb_id_]))
#    
#doc_ents_tmp = get_kb_ents("Electrocution")
#for ent in doc_ents_tmp:
#    print(" ".join(["ent", ent.text, ent.label_, ent.kb_id_]))


# In[ ]:





# In[ ]:


#nlp.get_vector("Q42")


# In[ ]:





# In[ ]:




