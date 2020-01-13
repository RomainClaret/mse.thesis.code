#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('autosave', '180')


# In[2]:


#import convex as cx
import requests
import time
import itertools
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

os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"
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
with open( "settings-tmqa-1.json", "r") as settings_data:
    settings = json.load(settings_data)
    use_cache = settings['use_cache']
    save_cache = settings['save_cache']
    cache_path = settings['cache_path']
#cache_path


# In[7]:


def save_cache_data():
    if save_cache:
        with open(os.path.join(cache_path,'statements_dict.json'), 'wb') as outfile:
            outfile.write(json.dumps(statements_dict, separators=(',',':')).encode('utf8'))
        with open(os.path.join(cache_path,'wd_labels_dict.json'), 'wb') as outfile:
            outfile.write(json.dumps(wd_labels_dict, separators=(',',':')).encode('utf8'))
        with open(os.path.join(cache_path,'wd_word_ids_dict.json'), 'wb') as outfile:
            outfile.write(json.dumps(wd_word_ids_dict, separators=(',',':')).encode('utf8'))
        with open(os.path.join(cache_path,'wd_predicate_ids_dict.json'), 'wb') as outfile:
            outfile.write(json.dumps(wd_predicate_ids_dict, separators=(',',':')).encode('utf8'))


# In[8]:


# Load statements cache
use_cache = False
if use_cache:
    path_statements_dict = "statements_dict.json"
    path_wd_labels_dict = 'wd_labels_dict.json'
    path_wd_word_ids_dict = 'wd_word_ids_dict.json'
    path_wd_predicate_ids_dict = 'wd_predicate_ids_dict.json'
else:
    path_statements_dict = "statements_dict_empty.json"
    path_wd_labels_dict = 'wd_labels_dict_empty.json'
    path_wd_word_ids_dict = 'wd_word_ids_dict_empty.json'
    path_wd_predicate_ids_dict = 'wd_predicate_ids_dict_empty.json'

with open(os.path.join(cache_path,path_statements_dict), "rb") as data:
    statements_dict = json.load(data)
with open(os.path.join(cache_path,path_wd_labels_dict), "rb") as data:
    wd_labels_dict = json.load(data)
with open(os.path.join(cache_path,path_wd_word_ids_dict), "rb") as data:
    wd_word_ids_dict = json.load(data)
with open(os.path.join(cache_path,path_wd_predicate_ids_dict), "rb") as data:
    wd_predicate_ids_dict = json.load(data)

#print("len(statements_dict)",len(statements_dict))
#print("len(wd_labels_dict)",len(wd_labels_dict))
#print("len(wd_word_ids_dict)",len(wd_word_ids_dict))
#print("len(wd_predicate_ids_dict)",len(wd_predicate_ids_dict))


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


def get_nlp(sentence, autocorrect=False):
    #print("sentence",sentence)
    nlp_sentence = nlp(sentence)
    nlp_sentence_list = list(nlp_sentence)
    meaningful_punct = []
    
    for i_t, t in enumerate(nlp_sentence_list):
        if t.lemma_ == "year":
            nlp_sentence_list[i_t] = "date"
        elif t.text == "\'s":
            if t.pos_ == "VERB" or t.pos_ == "AUX":
                nlp_sentence_list[i_t] = "is"
            else: nlp_sentence_list[i_t] = ""
        elif t.text == "\'re":
            nlp_sentence_list[i_t] = "are"
        elif t.pos_ == "PUNCT":
            if t.text.count(".") > 2:
                meaningful_punct.append((i_t,"..."))
                nlp_sentence_list[i_t] = "..."
            else:
                nlp_sentence_list[i_t] = ""
        else: nlp_sentence_list[i_t] = nlp_sentence_list[i_t].text
    
    nlp_sentence_list = [w for w in nlp_sentence_list if w]
    #nlp_sentence = " ".join(nlp_sentence_list)
    
    if autocorrect:
        nlp_sentence = " ".join(nlp_sentence_list)
        nlp_sentence = (nlp_sentence.replace("’", "\'").replace("€", "euro").replace("ç", "c")
                    .replace("à", "a").replace("é","e").replace("ä","a").replace("ö","o")
                   .replace("ü","u").replace("è","e").replace("¨","").replace("ê","e")
                   .replace("â","a").replace("ô","o").replace("î","i").replace("û","u")
                    .replace("_"," ").replace("°","degree").replace("§","section"))
        nlp_sentence = corrector.correct(nlp_sentence)
        nlp_sentence = nlp_sentence[0]["sequence"]
    
        nlp_sentence = nlp(nlp_sentence)
        nlp_sentence_list = list(nlp_sentence)

        for i_t, t in enumerate(nlp_sentence_list):
            if t.pos_ == "PUNCT":
                if i_t in [mp[0] for mp in meaningful_punct]:
                    for mp in meaningful_punct:
                        if i_t == mp[0]:
                            nlp_sentence_list[mp[0]] = mp[1]
                else: nlp_sentence_list[i_t] = ''

            else:
                nlp_sentence_list[i_t] = nlp_sentence_list[i_t].text

        for mp in meaningful_punct:
            if mp[0] < len(nlp_sentence_list):
                if nlp_sentence_list[mp[0]] != mp[1]:
                    nlp_sentence_list.insert(mp[0], mp[1])
        
    return nlp(" ".join(nlp_sentence_list))


#get_nlp("Which genre of album is harder.....faster?", autocorrect=True)
#get_nlp("Which genre of album is harder ... faster", autocorrect=True)
#get_nlp("Which home is an example of italianate architecture?", autocorrect=True)
#get_nlp("Your mom's father, were nice in the Years.!?\'\":`’^!$£€\(\)ç*+%&/\\\{\};,àéäöüè¨êâôîû~-_<>°§...@.....", autocorrect=True)
#get_nlp("of what nationality is ken mcgoogan", autocorrect=True)
#get_nlp("you're fun", autocorrect=True)
#get_nlp("where's the fun", autocorrect=True)


# In[11]:


#questions = [ 
#    "Which actor voiced the Unicorn in The Last Unicorn?",
#    "And Alan Arkin was behind...?",
#    "And Alan Arkin be behind...? Why How when which was happy make fun",
#    "Who is the composer of the soundtrack?",
#    "So who performed the songs?",
#    "Genre of this band's music?",
#    "By the way, who was the director?"
#            ]
#
#q_test = str("Which actor voiced the Unicorn in The Last Unicorn? "+
#    "And Alan Arkin was behind...? "+
#    "And Alan Arkin be behind...? Why How when which was happy make fun. "+
#    "Who is the composer of the soundtrack? "+
#    "So who performed songs? "+
#    "Genre of this band's music? "+
#    "By the way, who was the director? ")
#
#q_test_2 = "Who is the wife of Barack Obama?"


# In[12]:


#q0_nlp = get_nlp(questions[0])
#q0_nlp_test = get_nlp(q_test)
#q0_nlp_test_2 = get_nlp(q_test_2)
#print(q0_nlp)


# In[13]:


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


# In[14]:


# TODO redo the functions and optimize

def is_entity_or_literal(wd_object):
    if is_wd_entity(wd_object.strip()):
        return True
    pattern = re.compile('^[A-Za-z0-9]*$')
    if len(wd_object) == 32 and pattern.match(wd_object.strip()):
        return False
    return True

# return if the given string is a literal or a date
def is_literal_or_date (answer): 
    return not('www.wikidata.org' in answer)

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
    if statements_dict.get(entity_id) != None:
        #print("saved statement")
        return statements_dict[entity_id]
    entity = "http://www.wikidata.org/entity/"+entity_id
    statements = []
    # entity as subject
    triples_sub, cardinality_sub = hdt_wd.search_triples(entity, "", "")
    # entity as object
    triples_obj, cardinality_obj = hdt_wd.search_triples("", "", entity)
    if cardinality_sub + cardinality_obj > 5000:
        statements_dict[entity_id] = []
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
    statements_dict[entity_id] = statements
    return statements

#print(len(get_all_statements_of_entity("Q16614390")))
#save_cache_data()


# In[15]:


def get_wd_ids_online(name, is_predicate=False, top_k=3):
    name = name.split('(')[0]
    
    if is_predicate and wd_predicate_ids_dict.get(name) != None:
        #print("saved predicate online")
        return wd_predicate_ids_dict[name]
    elif not is_predicate and wd_word_ids_dict.get(name) != None:
        #print("saved word online")
        return wd_word_ids_dict[name]

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
        if is_predicate: wd_predicate_ids_dict[name] = ""
        else: wd_word_ids_dict[name] = ""
        return ""
    if not len(results):
        if is_predicate: wd_predicate_ids_dict[name] = ""
        else: wd_word_ids_dict[name] = ""
        return ""
    res = []
    for result in results:
        res.append(result['id'])
    
    if is_predicate: wd_predicate_ids_dict[name] = res
    else: wd_word_ids_dict[name] = res
    
    return res[:top_k]
#print(get_wd_ids_online("be", is_predicate=True, top_k=1))


# In[16]:


# very computational
def get_most_similar(word, topn=5):
    word = nlp.vocab[str(word)]
    queries = [w for w in word.vocab if w.is_lower == word.is_lower and w.prob >= -15]
    by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)
    return [(w.lower_,w.similarity(word)) for w in by_similarity[:topn+1] if w.lower_ != word.lower_]

#print(get_most_similar("voiced", topn=3))


# In[17]:


def get_wd_ids(word, is_predicate=False, top_k=3, limit=10):
    if is_predicate and wd_predicate_ids_dict.get(word) != None:
        #print("saved predicate local")
        return wd_predicate_ids_dict[word]
    elif not is_predicate and wd_word_ids_dict.get(word) != None:
        #print("saved word local")
        return wd_word_ids_dict[word]
    
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
    if is_predicate: wd_predicate_ids_dict[word] = results
    else: wd_word_ids_dict[word] = results
    
    return results if limit<=0 else results[:limit]
     
    
#get_wd_ids("The Last Unicorn", top_k=0, limit=10)
#print(get_wd_ids("wife", is_predicate=False , top_k=0, limit=0))
#print(get_wd_ids("voiced", is_predicate=False , top_k=0, limit=0))


# In[18]:


def get_wd_label(from_id):
    #print("from_id",from_id)
    if is_valide_wd_id(from_id):
        if wd_labels_dict.get(from_id) != None:
            #print("saved label local")
            return wd_labels_dict[from_id]
        
        language = "en"
        id_url = "http://www.wikidata.org/entity/"+from_id
        t_name, card_name = hdt_wd.search_triples(id_url, "http://schema.org/name", "")
        name = [t[2].split('\"@en')[0].replace("\"", "") for t in t_name if "@"+language in t[2]]
        result = name[0] if name else ''
        wd_labels_dict[from_id] = result #caching
        return result
        
    else:
        return ''
    
#print(get_wd_label("P725"))
#get_wd_label("Q20789322")


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')

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


# In[20]:


# TODO: handle dates and other literals, idea to start: entity != Q and objects != Q

def make_statements_graph(statements, indexing_predicates=True):
    BANNED_WD_IDS = [
        "Q4167410","Q66087861","Q65932995","Q21281405","Q17442446","Q41770487","Q29548341",
        "Q29547399","Q25670"
    ]
    BANNED_WD_PRED_IDS = [
        "P1687","P7087","P1889","P646", "P227", "P1256", "P1257", "P1258", "P1260", "P301",
        "P18","P1266","P487","P1970","P2529", "P4390", "P4342", "P4213", "P487", "P2624",
        "P4953", "P2241", "P345", "P703", "P2163", "P18", "P436", "P227", "P646", "P2581",
        "P1006", "P244", "P214", "P1051", "P1296", "P461", "P2959", "P1657", "P3834","P243",
        "P3306","P6932","P356","P1630","P3303","P1921","P1793","P1628","P1184","P1662","P2704",
        "P4793","P1921","P2302"
    ]
    
    graph = nx.Graph()
    turn=0
    predicate_nodes = {}

    for statement in statements:
        #statement['entity']['id'] in BANNED_WD_IDS 
        #statement['object']['id'] in BANNED_WD_IDS
        #statement['predicate']['id'] in BANNED_WD_PRED_IDS
        if (statement['entity']['id'][0] != "Q"
            or statement['entity']['id'] in BANNED_WD_IDS
            or statement['predicate']['id'][0] != "P"
            or statement['predicate']['id'] in BANNED_WD_PRED_IDS
            or statement['object']['id'][0] != "Q"
            or statement['object']['id'] in BANNED_WD_IDS):
            continue
        
        #print(statement)
        if not statement['entity']['id'] in graph:
            graph.add_node(statement['entity']['id'], name=get_wd_label(statement['entity']['id']), type='entity', turn=turn)
        if not statement['object']['id'] in graph:
            graph.add_node(statement['object']['id'], name=get_wd_label(statement['object']['id']), type='entity', turn=turn)

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
    
    return graph, predicate_nodes

#test_graph = make_statements_graph(test_unduplicate_statements, indexing_predicates=False)
#print(test_graph[1])
#plot_graph(test_graph[0],"test")


# In[21]:


def merge_lists(list_1, list_2):
    if len(list_1) == len(list_2):
        return [(list_1[i], list_2[i]) for i in range(0, len(list_1))]
    else:
        return "Error: lists are not the same lenght"

#print(merge_lists(["author"],['P50']))


# In[22]:


def get_themes(nlp_question, top_k=3):
    # PART1: finding themes as the user typed it
    filter_list = ["PART", "PRON", "NUM"]
    nlp_list_src = list(nlp_question)
    nlp_list = []
    for w in nlp_question:
        if w.pos_ not in filter_list:
            nlp_list.append(w)
    nlp_question = get_nlp(" ".join([e.text for e in nlp_list]))
    
    themes = [(ent, [ent.kb_id_]) for ent in get_kb_ents(nlp_question.text) if ent.kb_id_ != "NIL"]
    theme_complements = []
    
    noun_chunks = [chunk for chunk in nlp_question.noun_chunks]
    theme_ids = [get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks][:top_k]

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
        if w.pos_ == "DET":
            nlp_list_no_det.append(w.text)
            
    nlp_question_cap = get_nlp(" ".join([e for e in nlp_list_cap]))
    nlp_question_low = get_nlp(" ".join([e for e in nlp_list_low]))
    nlp_question_lemma = get_nlp(" ".join([e for e in nlp_list_lemma]))
    nlp_question_no_det = get_nlp(" ".join([e for e in nlp_list_no_det]))

    themes += [(ent, [ent.kb_id_]) for ent in get_kb_ents(nlp_question_cap.text) if ent.kb_id_ != "NIL" and (ent, [ent.kb_id_]) not in themes]
    themes += [(ent, [ent.kb_id_]) for ent in get_kb_ents(nlp_question_low.text) if ent.kb_id_ != "NIL" and (ent, [ent.kb_id_]) not in themes]
    themes += [(ent, [ent.kb_id_]) for ent in get_kb_ents(nlp_question_lemma.text) if ent.kb_id_ != "NIL" and (ent, [ent.kb_id_]) not in themes]
    themes += [(ent, [ent.kb_id_]) for ent in get_kb_ents(nlp_question_no_det.text) if ent.kb_id_ != "NIL" and (ent, [ent.kb_id_]) not in themes]
    
    #print(themes)
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
    
    theme_ids = [get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks][:top_k]

    for i, chunk in enumerate(theme_ids):
        if chunk: themes.append((noun_chunks[i], chunk))
        else: theme_complements.append(noun_chunks[i])
    
    noun_chunks = [chunk for chunk in nlp_question_low.noun_chunks]
    theme_ids = [get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks][:top_k]

    for i, chunk in enumerate(theme_ids):
        if chunk: themes.append((noun_chunks[i], chunk))
        else: theme_complements.append(noun_chunks[i])
            
    noun_chunks = [chunk for chunk in nlp_question_lemma.noun_chunks]
    theme_ids = [get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks][:top_k]

    for i, chunk in enumerate(theme_ids):
        if chunk: themes.append((noun_chunks[i], chunk))
        else: theme_complements.append(noun_chunks[i])
            
    noun_chunks = [chunk for chunk in nlp_question_no_det.noun_chunks]
    theme_ids = [get_wd_ids(chunk.text, top_k=top_k) for chunk in noun_chunks][:top_k]

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
            
    themes_filtered_undupped = []
    for tf in themes_filtered:
        tmp_ids = []
        for tfid in tf[1]:
            if tfid not in tmp_ids:
                tmp_ids.append(tfid)
        themes_filtered_undupped.append((tf[0],tmp_ids))
        
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

#q_test_test = get_nlp("Who voiced the Unicorn in The Last Unicorn")
#q_test_test = get_nlp("What is the name of the person who created Saved by the Bell?")
#get_themes(q_test_test, top_k=3)

#q_themes: ([(Unicorn, ['Q7246', 'Q7246', 'Q7246']), 
#(The Last Unicorn, ['Q30060419', 'Q16614390', 'Q176198', 'Q15628943', 'Q30060419', 'Q16614390', 'Q176198', 'Q15628943']), 
#(The Unicorn, ['Q17985004', 'Q18647334', 'Q17553756', 'Q65070436'])], 
#[the Unicorn, the unicorn, the last unicorn, the Unicorn, the last Unicorn])


# In[ ]:





# In[23]:


BANNED_WORDS = ["..."]

def get_theme_tuples(theme_list, top_k=3):
    return [(t, get_wd_ids(t, top_k=top_k)) for t in theme_list if t not in BANNED_WORDS]

def get_theme_no_stopwords(theme_list):
    return [s for s in theme_list if not s.is_stop]

def get_theme_lemmatized(theme_list):
    return [s.lemma_ for s in theme_list]

def get_permutation_tuples(theme_list, start=2):
    permutations = []
    for i in range(start, len(theme_list)+1):
        permutations += itertools.permutations(theme_list,i)
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

def get_enhanced_themes(themes, top_k=3, aggressive=False):
    if aggressive: top_k+=1
    enhanced_themes = []
    # permute, capitalize, lowering of the words in the complements
    for c in themes[1]:
        per_lemma = get_theme_tuples(get_non_token_tuples([n for n in get_permutation_tuples(get_theme_lemmatized(c))]),top_k)
        per_nostop = get_theme_tuples(get_text_tuples(get_permutation_tuples(get_theme_no_stopwords(c),start=1)),top_k)
        per_lemma_nostop = get_theme_tuples(get_non_token_tuples([get_theme_lemmatized(s) for s in get_permutation_tuples(get_theme_no_stopwords(c),start=1)]),top_k)

        per_lemma_lower = get_theme_tuples(get_non_token_lower_tuples([n for n in get_permutation_tuples(get_theme_lemmatized(c))]),top_k)
        per_nostop_lower = get_theme_tuples(get_lower_tuples(get_permutation_tuples(get_theme_no_stopwords(c),start=1)),top_k)
        per_lemma_nostop_lower = get_theme_tuples(get_non_token_lower_tuples([get_theme_lemmatized(s) for s in get_permutation_tuples(get_theme_no_stopwords(c),start=1)]),top_k)

        per_lemma_capitalize = get_theme_tuples(get_non_token_capitalize_tuples([n for n in get_permutation_tuples(get_theme_lemmatized(c))]),top_k)
        per_nostop_capitalize = get_theme_tuples(get_capitalized_tuples(get_permutation_tuples(get_theme_no_stopwords(c),start=1)),top_k)
        per_lemma_nostop_capitalize = get_theme_tuples(get_non_token_capitalize_tuples([get_theme_lemmatized(s) for s in get_permutation_tuples(get_theme_no_stopwords(c),start=1)]),top_k)

        per = get_theme_tuples(get_text_tuples(get_permutation_tuples(c)),top_k)
        per_lower = get_theme_tuples(get_lower_tuples(get_permutation_tuples(c)),top_k)
        per_capitalize = get_theme_tuples(get_capitalized_tuples(get_permutation_tuples(c)),top_k)

        for p in (per + per_lower + per_capitalize +
                 per_lemma + per_lemma_lower + per_lemma_capitalize +
                 per_nostop + per_nostop_lower + per_nostop_capitalize +
                 per_lemma_nostop + per_lemma_nostop_lower + per_lemma_nostop_capitalize):
            if p[1] and p not in enhanced_themes: enhanced_themes.append(p)
    
    if aggressive:
        predicates = []
        [predicates.append(get_wd_label(pred)) for pred in sum([p[1] for p in themes[0]],[]) if get_wd_label(pred) not in predicates]
        predicates_ids = [get_wd_ids_online(p, is_predicate=True, top_k=top_k) for p in predicates][:top_k]
        predicated_themes = merge_lists(predicates, predicates_ids)
        predicated_themes = [pt for pt in predicated_themes if pt[1] != '']
        if predicates: enhanced_themes += predicated_themes
            
    enhanced_themes_filtered = []
    for et in enhanced_themes:
        #print(et[0],[t[0].text for t in themes[0]],et[0] in [t[0].text for t in themes[0]])
        if not et[0] in [t[0].text for t in themes[0]]:
            enhanced_themes_filtered.append(et)
    
    return enhanced_themes_filtered

#q_test_3 = get_nlp("Which genre of album is harder.....faster?",autocorrect=True)
#q_test_3 = get_nlp("the unicorn and the raccoons love obama barack's tacos")
#q_test_3 = get_nlp("what was the cause of death of yves klein")
#q_test_3_themes = get_themes(q_test_3, top_k=3)
#print(q_test_3_themes[0])
#print(get_enhanced_themes(q_test_3_themes, aggressive=False))


# In[24]:


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
                    similar_words = [w[0] for w in get_most_similar(p.lemma_, topn=top_k)]
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


# In[25]:


#get_nlp("was").similarity(get_nlp("instance of"))
#get_wd_ids_online("do", is_predicate=True, top_k=3)
#get_nlp("do")[0].lemma_


# In[26]:


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


# In[27]:


def extract_ids(to_extract):
    return [i for i in itertools.chain.from_iterable([id[1] for id in to_extract])]
#extract_ids([('name', ['id'])]) #q0_themes[0] #q0_focused_parts #q0_predicates
#print(extract_ids([("The Last Unicorn", ['Q16614390']),("Second Theme", ['Q12345'])]))
#extract_ids(q0_focused_parts)


# In[28]:


def get_similarity_by_words(nlp_word_from, nlp_word_to):
    if not nlp_word_from or not nlp_word_to:
        return 0
    elif not nlp_word_from.vector_norm or not nlp_word_to.vector_norm:
        return 0
    else:
        return nlp_word_from.similarity(nlp_word_to)

#print(get_similarity_by_words(get_nlp("character role"), get_nlp("voice actor")))


# In[29]:


def get_similarity_by_ids(word_id_from, word_id_to):
    nlp_word_from = get_nlp(get_wd_label(word_id_from))
    nlp_word_to = get_nlp(get_wd_label(word_id_to))
    return get_similarity_by_words(nlp_word_from, nlp_word_to)

#print(get_similarity_by_ids("P453", "P725"))


# In[30]:


def get_top_similar_statements(statements, from_token_id, similar_to_name, top_k=3, qualifier=False, statement_type="object"):
    highest_matching_similarity = -1
    top_statements = []
    nlp_name = get_nlp(similar_to_name)
    
    if get_wd_label(from_token_id):
        for statement in statements:
            if qualifier:
                if statement.get('qualifiers'):
                    for qualifier in statement['qualifiers']:
                        nlp_word_to = get_nlp(get_wd_label(qualifier[statement_type]['id']))
                        matching_similarity = get_similarity_by_words(nlp_name, nlp_word_to)
                        if highest_matching_similarity == -1 or matching_similarity > highest_matching_similarity:
                            highest_matching_similarity = matching_similarity
                            best_statement = statement
                            top_statements.append((highest_matching_similarity, best_statement))
            else:
                nlp_word_to = get_nlp(get_wd_label(statement[statement_type]['id']))
                matching_similarity = get_similarity_by_words(nlp_name, nlp_word_to)
                if highest_matching_similarity == -1 or matching_similarity > highest_matching_similarity:
                    highest_matching_similarity = matching_similarity
                    best_statement = statement
                    top_statements.append((highest_matching_similarity, best_statement))
            
    return sorted(top_statements, key=lambda x: x[0], reverse=True)[:top_k]

#statements = get_all_statements_of_entity('Q176198')
#top_similar_statements = get_top_similar_statements(statements, 'Q176198', 'voiced')
#print(top_similar_statements[0])


# In[31]:


def get_best_similar_statements_by_word(from_token_ids, similar_to_name, top_k=3, qualifier=False, statement_type="object"):
    best_statements = []
    for token in from_token_ids:
        statements = get_all_statements_of_entity(token)
        if statements: best_statements += get_top_similar_statements(statements, token, similar_to_name, top_k=top_k, qualifier=qualifier, statement_type=statement_type)

    return sorted(best_statements, key=lambda x: x[0], reverse=True)[:top_k]

#best_similar_statements = get_best_similar_statements_by_word(extract_ids(q0_themes[0]), 'voiced', top_k=3, qualifier=True, statement_type="qualifier_object")
#print(best_similar_statements[0])


# In[32]:


def get_statements_subjects_labels(statements):
    return [get_wd_label(t[1]['entity']['id']) for t in statements]
#print(get_statements_subjects_labels(best_similar_statements))


# In[33]:


def get_statements_predicates_labels(statements):
    return [get_wd_label(t[1]['predicate']['id']) for t in statements]
#print(get_statements_predicates_labels(best_similar_statements))


# In[34]:


def get_statements_objects_labels(statements):
    return [get_wd_label(t[1]['object']['id']) for t in statements]
#print(get_statements_objects_labels(best_similar_statements))


# In[35]:


def get_statements_qualifier_predicates_labels(statements):
    return [get_wd_label(t[1]['qualifiers'][0]['qualifier_predicate']['id']) for t in statements]
#print(get_statements_qualifier_predicates_labels(best_similar_statements))


# In[36]:


def get_statements_qualifier_objects_labels(statements):
    return [get_wd_label(t[1]['qualifiers'][0]['qualifier_object']['id']) for t in statements]
#print(get_statements_qualifier_objects_labels(best_similar_statements))


# In[37]:


def cluster_extend_by_words(cluster_root_ids, extending_words, top_k=3):
    cluster = []
    #start_time = time.time()
    
    for name in extending_words:
        #start_cluster_time = time.time()
        cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=True, statement_type="qualifier_predicate")
        cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=True, statement_type="qualifier_object")
        cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=False, statement_type="predicate")
        cluster += get_best_similar_statements_by_word(cluster_root_ids, name, top_k=top_k, qualifier=False, statement_type="object")
        #end_time = time.time()
        #print("EXTENDING Cluster with:", name," ->\tRunning time is {}s".format(round(end_time-start_cluster_time,2)))
    #end_time = time.time()
    #print("EXTENDING Clusters ->\tRunning time is {}s".format(round(end_time-start_time,2)))
    return cluster
    
#test_cluster = cluster_extend_by_words(extract_ids(q0_themes[0]), ['voiced'], top_k=2)
#test_cluster_test_2 = cluster_extend_by_words(extract_ids(q0_themes_test_2[0]), ['birth'], top_k=2)
#print(test_cluster[0])


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


#test_graph = make_statements_graph(test_unduplicate_statements)
#print(test_graph[1])
#plot_graph(test_graph[0], "file_name_graph", "Graph_title")


# In[42]:


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


# In[43]:


# parameters
# cluster_root_ids: ['Qcode']
# predicates_ids: ['Pcode']
def cluster_extend_by_predicates_ids(cluster_root_ids, predicates_ids):
    cluster = []
    
    for cluster_root_id in cluster_root_ids:
        root_statements = get_all_statements_of_entity(cluster_root_id)
        #print("root_statements", root_statements)
        for predicate_id in predicates_ids:
            cluster += get_statements_by_id(root_statements, cluster_root_id, predicate_id, qualifier=True, statement_type="qualifier_predicate")
            cluster += get_statements_by_id(root_statements, cluster_root_id, predicate_id, qualifier=False, statement_type="predicate")

    return cluster
    
#test_predicate_clusters = cluster_extend_by_predicates_ids(extract_ids(q0_themes[0]), extract_ids(q0_predicates))
#print(len(test_predicate_clusters))
#test_predicate_clusters[0]

#test_predicate_clusters_test_2 = cluster_extend_by_predicates_ids(extract_ids(q0_themes_test_2[0]), extract_ids(q0_predicates_test_2))
#print(len(test_predicate_clusters_test_2))
#print(test_predicate_clusters_test_2[-1])


# In[44]:


def cluster_extractor_from_complements(complements):
    for c in complements:
        [print(t.pos_) for t in c]
    return complements

#print(cluster_extractor_from_complements(q0_themes[1]))


# In[45]:


#TODO: add cache
#TODO: Check if extending with predicate_ids is useful
# parameter
# question: nlp_string
#limits=plt.axis('off')
def build_graph(nlp, themes, themes_enhanced, predicates, deep_k=10):
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
    
    init_clusters = cluster_extend_by_words(theme_ids, [p[0].text for p in predicates+predicates_enhanced], top_k=deep_k)
    #print("init_clusters",len(init_clusters))
    init_clusters_enhanced = cluster_extend_by_words(theme_enhanced_ids, [p[0].text for p in predicates+predicates_enhanced], top_k=deep_k)
    #print("init_clusters_enhanced",len(init_clusters_enhanced))
    init_sorted_statements = sort_statements_by_similarity(init_clusters + init_clusters_enhanced)
    #print("init_sorted_statements",len(init_sorted_statements))
    init_flatten_statements = statements_flatter([s[1] for s in init_sorted_statements])
    #print("init_flatten_statements",len(init_flatten_statements))
    
    predicate_ids_clusters = cluster_extend_by_predicates_ids(theme_ids, predicates_ids+predicates_enhanced_ids)
    #print("predicate_ids_clusters",len(predicate_ids_clusters))
    predicate_ids_enhanced_clusters = cluster_extend_by_predicates_ids(theme_enhanced_ids, predicates_ids+predicates_enhanced_ids)
    #print("predicate_ids_enhanced_clusters",len(predicate_ids_enhanced_clusters))
    predicate_ids_flatten_statements = statements_flatter(predicate_ids_clusters+predicate_ids_enhanced_clusters)
    #print("predicate_ids_flatten_statements",len(predicate_ids_flatten_statements))
    
    clusters = init_flatten_statements+predicate_ids_flatten_statements
    filtered_statements = unduplicate_statements(clusters)
    #print(predicate_ids_enhanced_clusters)
    graph = make_statements_graph(filtered_statements)

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


# In[46]:


# check the graph for complements
# parameters
# name: string
def find_name_in_graph(graph, name):
    return [x for x,y in graph.nodes(data=True) if y['name'].lower() == name.lower()]

#[find_name_in_graph(c.text) for c in q0_themes[1]]
#print(find_name_in_graph(graph, "the unicorn"))


# In[47]:


# TODO: clean the complements by removing stopwords etc.
def find_theme_complement(graph, themes):
    return [i for i in itertools.chain.from_iterable(
        [id for id in [c for c in [find_name_in_graph(graph, t.text) for t in themes[1]] if c]])]

#print(find_theme_complement(graph, q0_themes_test))
#[i for i in itertools.chain.from_iterable([id for id in check_theme_complement(graph, q0_themes)])]


# In[48]:


def find_paths_in_graph(graph, node_start, node_end):
    return [p for p in nx.all_simple_paths(graph, source=node_start, target=node_end)]
        
#test_paths = find_paths_in_graph(graph, "Q16205566", "Q7774795")
#print(test_paths)


# In[49]:


def is_id_in_graph(graph, node_id):
    return graph.has_node(node_id)
#print(is_id_in_graph(graph, "Q24039104"))


# In[50]:


def is_name_in_graph(graph, node_name):
    return find_name_in_graph(graph, node_name) != []
#print(is_name_in_graph(graph, "the Unicorn"))


# In[51]:


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
    paths = [i for i in itertools.chain.from_iterable(
        [id for id in paths])]
    
    return paths
#print(find_paths_for_themes(graph, q0_themes_test))
#print(find_paths_for_themes(graph, q0_themes))


# In[52]:


def get_node_predicates_from_path(paths):
    predicates = []
    for p in paths:
        [predicates.append(i[:i.find("-")]) for i in p if is_wd_predicate(i[:i.find("-")]) and i[:i.find("-")] not in predicates]
    return predicates

#test_node_predicates = get_node_predicates_from_path(test_paths)
#print(test_node_predicates)


# In[53]:


def get_node_predicate_similarity_from_path(paths, predicates):
    path_predicates = get_node_predicates_from_path(paths)
    return sorted([(pp, get_similarity_by_ids(p2, pp)) for p in predicates for p2 in p[1] for pp in path_predicates], key=lambda x: x[-1], reverse=True)

#test_node_pedicate_similarities = get_node_predicate_similarity_from_path(test_paths, q0_predicates)
#print(test_node_pedicate_similarities)


# In[54]:


def get_focused_parts(nlp_sentence, themes, top_k=3):
    W_FILTERS = ["WDT", "WP", "WP$", "WRB"]
    V_FILTERS = ["VERB", "AUX"]

    focused_parts = [t.head for t in nlp_sentence if t.tag_ in W_FILTERS] 
    for fp in focused_parts:
        if fp.children:
            for c in fp.children:
                if c.tag_ not in W_FILTERS and c.text not in [fp.text for fp in focused_parts]: 
                    focused_parts.append(c)
    
    #print("focused_parts",focused_parts)
    #print("themes[0]",themes[0])
    
    focused_parts_len = len(focused_parts)
    for t in themes[0]:
        for i_fp, fp in enumerate(focused_parts):
            for i_w, w in enumerate([w.lower_ for w in t[0]]):
                if fp.lower_ == w:
                    if i_fp+1 < focused_parts_len-1:
                        if focused_parts[i_fp+1].lower_ == t[0][i_w-1].lower_:
                            #print(i_fp,fp, t[0][i_w-1], t[0])
                            #print("BEFORE focused_parts",focused_parts)
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
    #print("focused_parts_ids",focused_parts_ids)
    merged_list = merge_lists(focused_parts, focused_parts_ids)
    #print("merged_list",merged_list)
    
    dummy_span = get_nlp("dummy span")[:]
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

#q_focused_parts: [(Unicorn, ['Q18356448', 'Q21070472', 'Q22043340', 'Q1565614', 'Q30060419']),
#(in, ['P642', 'Q29733109', 'P361', 'P131']),
#(the, ['Q1408543', 'Q2865743', 'Q29423', 'Q21121474']),
#(Unicorn, ['Q18356448', 'Q21070472', 'Q22043340', 'Q1565614', 'Q30060419']),
#(The, ['Q1067527', 'Q13423400', 'Q28457426', 'Q24406786', 'Q2430521', 'Q37199001']),
#(Last, ['Q16995904', 'Q20072822', 'Q24229340', 'Q20155285'])]


# In[55]:


#questions_2 = ("what was the cause of death of yves klein",
#               "Who is the wife of Barack Obama?",
#               "Who is the president of the United States?",
#               "When was produced the first Matrix movie?",
#               "Who made the soundtrack of the The Last Unicorn movie?",
#               "Who is the author of Le Petit Prince?",
#               "Which actor voiced the Unicorn in The Last Unicorn?",
#               "how is called the rabbit in Alice in Wonderland?",
#               "what city was alex golfis born in",
#               "which stadium do the wests tigers play in",
#               "Which nation is Martha Mattox from"
#              )
#
#question_2 = questions_2[6] #"what city was alex golfis born in"#
#question_2 = "what's akbar tandjung's ethnicity"
#q_nlp_2 = get_nlp(question_2)
#q_themes_2 = get_themes(q_nlp_2, top_k=3)
#q_themes_enhanced_2 = get_enhanced_themes(q_themes_2, top_k=3)
#q_predicates_2 = get_predicates(q_nlp_2, top_k=3)
#if q_predicates_2:
#        if not q_predicates_2[0][1]: q_predicates_2 = get_predicates_online(q_nlp_2, top_k=3)
#q_focused_parts_2 = get_focused_parts(q_nlp_2)
#print("q_nlp:", q_nlp_2)
#print("e\t\te.pos_\te.tag_\te.dep_\te.head\te.children")
#for e in q_nlp_2:
#    print(e.text,"\t\t", e.pos_,"\t", e.tag_,"\t", e.dep_,"\t", e.head, "\t", [child for child in e.children])
#
#print("\nq_themes:", q_themes_2)
#print("q_themes_enhanced:",q_themes_enhanced_2)
#print("q_predicates:", q_predicates_2)
#print("q_focused_parts:", q_focused_parts_2)
#
#graph_2, predicates_dict_2 = build_graph(q_nlp_2, q_themes_2, q_themes_enhanced_2, q_predicates_2, deep_k=40)
#print(len(graph_2), "nodes and", graph_2.size(), "edges")
#print(predicates_dict_2)
##plot_graph(graph_2, "main_graph", "Main_graph_title")
##answers_2 = find_anwser_from_graph_2(graph, q0_nlp, q0_themes, q_themes_enhanced_2, q_predicates_2, q_focused_parts_2)


# In[56]:


#plot_graph(graph_2, "test_file_name_graph", "Graph_title")


# In[57]:


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
def find_paths_keywords(graph, nlp, themes, themes_enhanced, predicates, focused_parts):
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
    #paths_keywords = [p for p in itertools.permutations(paths_keywords)]
    #print(paths_keywords)
    return paths_keywords, alterniative_words, question_anchors
    
    #initial_paths = find_paths_for_themes(graph, themes)
    #predicate_id_similarities = get_node_predicate_similarity_from_path(initial_paths, predicates)
    #best_path = [p for p in initial_paths if predicate_id_similarities[0][0] == p[1][:p[1].find("-")]]
    #path_answer = get_wd_label(best_path[0][2]) if best_path else []
    
    #return (path_answer, best_path[0][2]) if path_answer else (False, False)

#paths_keywords_2 = find_paths_keywords(graph_2, q_nlp_2, q_themes_2, q_themes_enhanced_2, q_predicates_2, q_focused_parts_2)
#paths_keywords_2


# In[58]:


def get_paths_keywords_nodes(graph, keywords,threshold=0.9,top_performance=50):
    keywords_nodes = []
    for k in keywords:
        nlp_lookup = get_nlp(k)
        keywords_nodes.append([x for x,y in graph.nodes(data=True)
               if get_nlp(y['name']).similarity(nlp_lookup) >= threshold])
    keywords_nodes = [k for k in keywords_nodes if k]
    
    #keywords_nodes [['Q17521117', 'Q17521118', 'Q557214', 'Q421946', 'Q11282976', 'Q4677712', 'Q33999'], ['Q7246', 'Q1307944', 'Q21070472', 'Q18356448', 'Q1863113', 'Q20983877', 'Q226755', 'Q22043340'], ['Q176198', 'Q967268', 'Q17553756', 'Q30060419', 'Q17985004', 'Q16614390', 'Q18647334', 'Q15628943'], ['Q176198', 'Q967268', 'Q17553756', 'Q30060419', 'Q17985004', 'Q16614390', 'Q18647334', 'Q15628943'], []]
    #keywords_nodes[0] ['Q17521117', 'Q17521118', 'Q557214', 'Q421946', 'Q11282976', 'Q4677712', 'Q33999']
    #keywords_nodes[1] ['Q7246', 'Q1307944', 'Q21070472', 'Q18356448', 'Q1863113', 'Q20983877', 'Q226755', 'Q22043340']
    
    keywords_nodes_per = []
    if keywords_nodes:
        if len(keywords_nodes) > 1:
            for kn_i, kn in enumerate(keywords_nodes):
                if kn_i + 1 < len(keywords_nodes):
                    if len(kn) * len(keywords_nodes[kn_i+1]) > top_performance:
                        if len(kn) <= int(sqrt(top_performance)):
                            keywords_nodes[kn_i+1] = keywords_nodes[kn_i+1][:int(top_performance/len(kn))]
                        elif len(kn) >= len(keywords_nodes[kn_i+1]):
                            kn = kn[:int(top_performance/len(keywords_nodes[kn_i+1]))]
                        else:
                            kn = kn[:int(sqrt(top_performance))]
                            keywords_nodes[kn_i+1] = keywords_nodes[kn_i+1][:int(sqrt(top_performance))]
            
            keywords_nodes_per = [p for p in itertools.permutations(keywords_nodes, 2)]
                            
        else:
            keywords_nodes_per = [(keywords_nodes+keywords_nodes)]
    
    paths_keyword_nodes = []
    for pkn in keywords_nodes_per:
        for pkn1 in pkn[0]:
            for pkn2 in pkn[1]:
                [paths_keyword_nodes.append(p) for p in nx.all_simple_paths(graph, source=pkn1, target=pkn2) if p not in paths_keyword_nodes]
    
    return paths_keyword_nodes

def find_path_nodes_from_graph(graph, keywords, threshold=0.9, thres_inter=0.15, top_performance=50,min_paths=3000):
    #print("current threshold", str(round(threshold, 1)))
    main_keyword_paths = get_paths_keywords_nodes(graph, keywords[0],threshold=threshold,top_performance=top_performance)
    alternative_keyword_paths = []
    
    for k_1 in keywords[1]:
        for i, k_0 in enumerate(keywords[0]):
            if k_1==k_0:
                tmp_keywords = keywords[0].copy()
                tmp_keywords[i] = keywords[1][k_1][0].text
                alternative_keyword_paths += get_paths_keywords_nodes(graph, tmp_keywords, threshold=threshold,top_performance=top_performance)
    
    keyword_paths = main_keyword_paths+alternative_keyword_paths
    
    #print("len(keyword_paths)",len(keyword_paths))
    if len(keyword_paths) < min_paths:
        if threshold == 0: return keyword_paths
        threshold -= thres_inter
        if threshold < 0: threshold = 0
        keyword_paths = find_path_nodes_from_graph(graph, keywords, threshold, thres_inter,top_performance,min_paths)
    
    return keyword_paths

#start_time = time.time()
#path_nodes_2 = find_path_nodes_from_graph(graph_2, paths_keywords_2, threshold=0.9, thres_inter=0.15, top_performance=50, min_paths=3000)
#end_time = time.time()
#print("Finding path nodes ->\tRunning time is {}s".format(round(end_time-start_time,2))) 
#print(path_nodes_2)


# In[59]:


#node_predicates_names_2 = get_node_predicates_from_path(path_nodes_2)

def is_sublist(a, b):
    if not a: return True
    if not b: return False
    #if a == b: return False
    return b[:len(a)] == a or is_sublist(a, b[1:])

def paths_nodes_filter(path_nodes, graph):
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
    
    unique_paths = filtered_paths.copy()
    for i,fp in enumerate(filtered_paths):
        for fp_2 in filtered_paths:
            if (is_sublist(fp, fp_2) and fp!=fp_2):
                unique_paths[i] = []
                break
    
    unique_paths = [p for p in unique_paths if p]
    
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
    
    return [p for p in unique_paths_with_reversed if len(p) > 2] #False#[up for up in unique_paths if up and not is_wd_predicate(up[-1]) and not is_wd_predicate(up[0])]#False# [p for p in unique_paths if p]
                
#paths_nodes_filtered_2 = paths_nodes_filter(path_nodes_2, graph_2)
#print("unique_paths", len(paths_nodes_filtered_2))
#for p in paths_nodes_filtered_2:
#    print(p)


# In[60]:


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


# In[61]:


def get_entity_similarity(word_id, entity_type, max_reward=2.0):
    LOCATION_FILTER = ["GPE", "FAC", "LOC"]
    PERSON_FILTER = ["PERSON","NORP","ORG","PER"]
    DATE_FILTER = ["DATE","TIME"]
    CAUSE_FILTER = ["NORP","PRODUCT","EVENT","MISC"]
    WHICH_FILTER = PERSON_FILTER+DATE_FILTER+["GPE","LOC","PRODUCT","EVENT",
                    "WORK_OF_ART","LAW","LANGUAGE","MISC"]
    WHAT_FILTER = LOCATION_FILTER+DATE_FILTER+CAUSE_FILTER+["NORP","ORG",
                     "PER","WORK_OF_ART","LAW","LANGUAGE"]
                    
    QUANTITY_FILTER = ["PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
    
    ALL_FILTER = LOCATION_FILTER + PERSON_FILTER + DATE_FILTER + CAUSE_FILTER + WHICH_FILTER + WHAT_FILTER + QUANTITY_FILTER
    
    similarities = []
    word_label = get_wd_label(word_id)
    if word_label == "":
        return similarities
    
    word_ents = get_kb_ents(word_label)
    if word_ents:
        for ent in word_ents:
            if (entity_type in ALL_FILTER and ent.label_ == entity_type):
                similarities.append(max_reward)
            
            elif ent.kb_id_ == word_id:
                if entity_type == "location" and ent.label_ in LOCATION_FILTER:
                    similarities.append(max_reward)
                elif entity_type == "person" and ent.label_ in PERSON_FILTER:
                    similarities.append(max_reward)
                elif entity_type == "date" and ent.label_ in DATE_FILTER:
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


# In[ ]:





# In[62]:


#test_question = "Of what nationality is Ken McGoogan"#"Which is the nation of Martha Mattox"#"Who voiced the Unicorn in The Last Unicorn?"
#
#test_verbose = True
#test_deep_k = 30
#test_deep_k_step = 20
#
#test_q_nlp = get_nlp(test_question)
#if test_verbose: print("-> test_q_nlp:",test_q_nlp)
#test_q_themes = get_themes(test_q_nlp, top_k=3)
#if test_verbose: print("-> test_q_themes:",test_q_themes)
#test_q_themes_enhanced = get_enhanced_themes(test_q_themes, top_k=1, aggressive=False)
#if test_verbose: print("-> test_q_themes_enhanced:",test_q_themes_enhanced)
#test_q_predicates = get_predicates(test_q_nlp, test_q_themes, top_k=0)
#if test_q_predicates:
#    test_has_predicates = False
#    for test_qp in test_q_predicates:
#        if test_qp[1]:
#            test_has_predicates = True
#    if not test_has_predicates: test_q_predicates = get_predicates_online(test_q_nlp, top_k=2, aggressive=False)
#else:
#    test_q_predicates = get_predicates_online(test_q_nlp, top_k=2, aggressive=True)    
#if test_verbose: print("-> test_q_predicates:",test_q_predicates)
#test_q_focused_parts = get_focused_parts(test_q_nlp, test_q_themes, top_k=3)
#if test_verbose: print("-> test_q_focused_parts:",test_q_focused_parts)
#if test_verbose: print("-> Building the graph with test_k_deep",str(test_deep_k),"... (could be long)")
#if test_deep_k<=10:
#    test_deep_k = 10
#    test_graph, test_predicates_dict = build_graph(test_q_nlp, test_q_themes, test_q_themes_enhanced, test_q_predicates, deep_k=test_deep_k)
#else:
#    for k in range(10, test_deep_k, test_deep_k_step):
#        test_graph, test_predicates_dict = build_graph(test_q_nlp, test_q_themes, test_q_themes_enhanced, test_q_predicates, deep_k=test_deep_k)
#        if test_graph.size() > 1000 or len(test_graph) > 1000 or test_deep_k<=10:
#            break
#        elif test_graph.size() > 500 or len(test_graph) > 500:
#            test_deep_k -= test_deep_k_step
#            if test_verbose: print("---> Rebuilding the graph with k_deep",str(test_deep_k), "... Previously:",len(test_graph), "nodes or", test_graph.size(), "edges was above the limit...")
#        else: break
#if test_verbose: print("--> ",len(test_graph), "nodes and", test_graph.size(), "edges")
#if test_graph.size() > 510 or len(test_graph) > 510:
#    if test_verbose: print("Stopping the computing here, too computational.")
#if test_verbose: print("-> test_predicates_dict:",test_predicates_dict)
#test_paths_keywords = find_paths_keywords(test_graph, test_q_nlp, test_q_themes, test_q_themes_enhanced, test_q_predicates, test_q_focused_parts)
#if test_verbose: print("-> test_paths_keywords:",test_paths_keywords)
#if test_verbose: print("-> Computing possible paths... (could be long)")
#test_path_nodes = find_path_nodes_from_graph(test_graph, test_paths_keywords, threshold=0.8, thres_inter=0.1, top_performance=test_graph.size(),min_paths=3000)
#if test_verbose: print("--> len(path_nodes):",len(test_path_nodes))
#if len(test_path_nodes) < 20000:
#    if test_verbose: print("-> Filtering paths... (could be long)")
#    test_paths_nodes_filtered = paths_nodes_filter(test_path_nodes, test_graph)
#    if test_verbose: 
#        print("--> len(paths_nodes_filtered):",len(test_paths_nodes_filtered))
#else: 
#    if test_verbose: print("--> Skipping paths filtering... (too much paths)")
#    test_paths_nodes_filtered = test_path_nodes


# In[63]:


def get_hypothesises(nlp, predicates, themes, paths_keywords, filtered_paths, threshold=0.5, max_reward=2.0):#, themes, themes_enhanced):
    
    complementary_predicates = paths_keywords[0]+[p[0] for p in list(paths_keywords[1].values())]
    #print("complementary_predicates",complementary_predicates)
    
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
    
    w_positions, w_names = w_converter(nlp)
    w_names_only = [wn[1] for wn in w_names]
    #print("w_positions",w_positions)
    #print("w_names",w_names)
    #print("w_names_only",w_names_only)
    
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
    
    #return 0
    for p in filtered_paths:
        p_len = len(p)
        for i_e, e in enumerate(p):
            if is_wd_predicate(e):
                if main_predicate_ids:
                    if e in main_predicate_ids:
                        if e not in [ap[0] for ap in anchors_predicates]:
                            anchors_predicates.append((e, 1.0))
                    elif e not in [ap[0] for ap in anchors_predicates]:
                        stat_count = 0
                        stat_current = 0
                        for pn in main_predicate_names:
                            stat_current += get_similarity_by_words(get_nlp(get_wd_label(e)),pn)
                            stat_count += 1
                        for pi in main_predicate_ids:
                            stat_current += get_similarity_by_words(get_nlp(get_wd_label(e)),get_nlp(get_wd_label(pi)))
                            stat_count += 1
                        anchors_predicates.append((e, stat_current/stat_count))
                        
                elif e not in [ap[0] for ap in anchors_predicates]:
                    stat_count = 0
                    stat_current = 0
                    for af in anchors_focuses:
                        stat_current += get_similarity_by_words(get_nlp(get_wd_label(e)),get_nlp(af))
                        stat_count += 1
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
    
    hypothesises_tuples = []
    for ap in anchors_predicates:
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
                        #if "Q4985" in hypothesis_tuple:
                        #    print("Q4985 hypothesis_tuple",hypothesis_tuple, ap,fp)
                        
    #print("hypothesises_tuples",hypothesises_tuples)
    #print("hypothesises_tuples",hypothesises_tuples)
    #print([a[0] for a in anchors_predicates])
    keywords_names = [af for af in anchors_focuses]
    #print("keywords_names",keywords_names)
    #keywords_ids = [i for j in [get_wd_ids(k) for k in keywords_names if get_wd_ids(k)] for i in j]
    #print("keywords_names",keywords_ids)
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
    for ht in hypothesises_tuples:
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
                #    for es in get_entity_similarity(ht[0], wn[1], max_reward=max_reward):
                #        hypo_sum += es
                
                if (nlp_af.text.lower() != nlp_ht2.text.lower() 
                    and af_lemma != nlp_ht2[0].text.lower()
                    and nlp_af.text.lower() != ht2_lemma
                    and af_lemma != ht2_lemma
                   ):
                    hypo_sum += get_similarity_by_words(nlp_ht2, nlp_af)
                    
                    if i_af in w_positions:
                        for wn in w_names:
                            if i_af == wn[0]:
                                for es in get_entity_similarity(ht[0], wn[1], max_reward=max_reward):
                                    #if ht[0] == "Q4985": print("before Q4985:",i_af, es,ht[0], wn[1], hypo_sum)
                                    hypo_sum += es
                                    #if ht[0] == "Q4985": print("after Q4985:",i_af, es,ht[0], wn[1], hypo_sum)
                                                       
                    for ap in anchors_predicates:
                        if ap[0] == ht[1]:
                            ht0_label = get_wd_label(ht[0]).lower()
                            ht2_label = get_wd_label(ht[2]).lower()
                            
                            ht0_sum = 0
                            ht2_sum = 0
                            
                            for wn in w_names_only:
                                for es in get_entity_similarity(ht[0], wn, max_reward=max_reward):
                                    ht0_sum += es
                                for es in get_entity_similarity(ht[2], wn, max_reward=max_reward):
                                    ht2_sum += es
                                    
                            for tk in theme_keywords:
                                if ht0_label == tk.text.lower() and ht[1] in main_predicate_ids:
                                    #print("ht0_label",ht0_label)
                                    for wn in w_names_only:
                                        for es in get_entity_similarity(ht[2], wn, max_reward=max_reward*2):
                                            #print("ht0_sum before",ht0_sum)
                                            ht0_sum += es
                                            #print("ht0_label",ht2_label,es, ht0_sum, ht)
                                if ht2_label == tk.text.lower() and ht[1] in main_predicate_ids:
                                    for wn in w_names_only:
                                        for es in get_entity_similarity(ht[0], wn, max_reward=max_reward*2):
                                            #print("ht2_sum before",ht0_sum)
                                            ht2_sum += es
                                            #print("ht2_label",ht0_label,es, ht2_sum, ht)
                                    
                            
                            if ht2_label in keywords_names and ht0_label not in keywords_names:
                                hypo_sum += ht2_sum
                            elif ht0_label in keywords_names and ht2_label not in keywords_names:
                                hypo_sum += ht0_sum
                            else:
                                hypo_sum += ht0_sum
                                hypo_sum += ht2_sum
                                
                            
                            hypo_sum *= ap[1]
                            
                            #if get_wd_label(ht[0]).lower() in keywords_names:
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
                    
                    if get_wd_label(ht[0]).lower() in anchors_focuses: 
                        if not get_wd_label(ht[2]).lower() in anchors_focuses:
                            hypo = ht[2]
                        if get_wd_label(ht[2]).lower() in anchors_focuses:
                            break
                    elif not get_wd_label(ht[0]).lower() in anchors_focuses:
                        if get_wd_label(ht[2]).lower() in anchors_focuses:
                            hypo = ht[0]
                        if not get_wd_label(ht[2]).lower() in anchors_focuses:
                            hypothesises_all.append(ht[0])
                            if not hypothesises: hypothesises.append([ht[0], hypo_sum])
                            else: 
                                if ht[0] in [h[0] for h in hypothesises]:
                                    for i, h in enumerate(hypothesises):
                                        if ht[0] == h[0]: hypothesises[i] = [ht[0], hypo_sum+hypothesises[i][1]]
                                else: hypothesises.append([ht[0], hypo_sum])
                            
                            #if "Q4985" in ht: print("Q4985 ALONE hypo and sum:", ht[0], hypo_sum)
                            hypo = ht[2]
                    else:
                        #print("BREAK", ht)
                        break
                    
                    #if "Q4985" in ht:
                    #    print("Q4985 hypo and sum:", hypo, hypo_sum)
                    
                    hypothesises_all.append(hypo)
                    if not hypothesises: hypothesises.append([hypo, hypo_sum])
                    else: 
                        if hypo in [h[0] for h in hypothesises]:
                            for i, h in enumerate(hypothesises):
                                if hypo == h[0]: hypothesises[i] = [hypo, hypo_sum+hypothesises[i][1]]
                        else: hypothesises.append([hypo, hypo_sum])
                            
    for i_h, h in enumerate(hypothesises):
        h_sum = hypothesises_all.count(h[0])
        #print("h_sum",h_sum)
        #print("BEFORE: hypothesises[i_h][1]",hypothesises[i_h][1])
        hypothesises[i_h][1] = hypothesises[i_h][1]/h_sum
        #print("AFTER: hypothesises[i_h][1]",hypothesises[i_h][1])
    
    #print("hypothesises_all",hypothesises_all)    
    return sorted(hypothesises, key=lambda x: x[-1], reverse=True)
    
#hypothesises_2 = get_hypothesises(q_nlp_2, paths_keywords_2, paths_nodes_filtered_2)#, q_themes_2, q_themes_enhanced_2)
#print("q_nlp_2", q_nlp_2)
#print("paths_keywords_2", paths_keywords_2)
#print("paths_nodes_filtered", paths_nodes_filtered)
#print("hypothesises_2", hypothesises_2)
#print(hypothesises_2)
#print([(get_wd_label(h[0]),h[1]) for i,h in enumerate(hypothesises_2)])

#if test_verbose: print("-> Computing hypothesises...")
#test_hypothesises = get_hypothesises(test_q_nlp, test_q_predicates, test_q_themes, test_paths_keywords, test_paths_nodes_filtered, threshold=0.5, max_reward=2.0)
#if test_verbose: print("\n\n--> test_hypothesises:",test_hypothesises)


# In[64]:


def list_by_n(l, i):
    list_n = []
    for j in range(0, len(l)+1, 1):
        tmp = l[j-i:i+j-i]
        if tmp:
            list_n.append(tmp)
    return list_n

def match_hypothesises(graph, question, themes, predicates, hypothesises, paths, threshold=0.8, max_reward=2.0):
    meaningful_paths = []
    
    w_positions, w_names = w_converter(question)
    
    theme_ids = sum([t[1] for t in themes[0]],[])
    for p in paths:
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
                for es in get_entity_similarity(p[wp], w_names[i_wp][1], max_reward=max_reward):
                    counter += es
                
        for hypo in hypothesises:
            if hypo[0] in p:
                counter += 1
            if hypo[0] == p[0]:
                counter += 1
            if hypo[0] == p[-1]:
                counter += 1

        if counter > 0: meaningful_paths.append((counter, p))

    meaningful_paths = sorted(meaningful_paths, key=lambda x: x[0], reverse=True)
    #print("len(meaningful_paths):",len(meaningful_paths))
    #print("\n")
          
    looped_paths = []
    for hypo in hypothesises:
        for mp in meaningful_paths:
            if mp[1][0] == hypo[0] or mp[1][-1] == hypo[0]:
                if graph.has_node(mp[1][0]) and graph.has_node(mp[1][-1]):
                    path_tmp = list(nx.all_simple_paths(graph, mp[1][0],mp[1][-1]))
                    if len(path_tmp)>1:
                        for p in path_tmp:
                            if p not in [lp[1] for lp in looped_paths]:
                                looped_paths.append((mp[0],p))
                #else:
                #    if not graph.has_node(mp[1][0]):
                #        print("MISSING NODE:", mp[1][0], get_wd_label(mp[1][0]))
                #    if not graph.has_node(mp[1][-1]):
                #        print("MISSING NODE:", mp[1][-1], get_wd_label(mp[1][-1]))
                
    #print("len(looped_paths)", len(looped_paths))
    
    looped_paths_untagged = []
    for lp in looped_paths:
        row_tmp = []
        for w in lp[1]:
            if w.find("-") > 0:
                row_tmp.append(w[:w.find("-")])
            else:
                row_tmp.append(w)
        looped_paths_untagged.append((lp[0],row_tmp))
        
    #print("looped_paths_untagged",looped_paths_untagged)
    
    mp_similarities_untagged = []
    mp_similarities_tagged = []
    mp_similarities_untagged_hypo = []
    mp_similarities_tagged_hypo = []
    
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
            mp_similarities_tagged_hypo.append((similarity_avg, looped_paths[i_lp][1]))
        
        mp_similarities_untagged.append((similarity_avg, lp[1]))
        mp_similarities_tagged.append((similarity_avg, looped_paths[i_lp][1]))
        
    #print("mp_similarities_untagged",len(mp_similarities_untagged))
    #print("mp_similarities_untagged_hypo",len(mp_similarities_untagged_hypo))
    #print("mp_similarities_untagged",mp_similarities_untagged)
    
    mp_similarities_tagged = sorted(mp_similarities_tagged, key=lambda x: x[0], reverse=True)
    mp_similarities_tagged = [mp for mp in mp_similarities_tagged if mp[0] > threshold]
    
    mp_similarities_untagged = sorted(mp_similarities_untagged, key=lambda x: x[0], reverse=True)
    mp_similarities_untagged = [mp for mp in mp_similarities_untagged if mp[0] > threshold]
    
    #print("mp_similarities_untagged",len(mp_similarities_untagged))
            
    [mp_similarities_untagged.append(suh) for suh in mp_similarities_untagged_hypo if not suh in mp_similarities_untagged]
    [mp_similarities_tagged.append(sth) for sth in mp_similarities_tagged_hypo if not sth in mp_similarities_tagged]
    
    #print("mp_similarities_untagged",len(mp_similarities_untagged))
    #print("mp_similarities_tagged",len(mp_similarities_tagged))
    
    #WH_FILTER = ["WDT", "WP", "WP$", "WRB"]
    #wh_position = [w.i for w in question if w.tag_ in WH_FILTER][0]
    #question_list = [w.lower_ for w in question if not w.is_punct]
    #question_list_filtered = [w.lower_ for w in question if not w.is_punct and w.tag_ not in WH_FILTER]
    
    #golden_paths = []
    #for mp in mp_similarities_tagged:
    #    #print("mp[1]",mp[1])
    #    for i_e, e in enumerate(mp[1]):
    #        if i_e <= 1 or i_e >= len(mp[1])-2:
    #            continue
    #        if not is_wd_entity(e):
    #            continue
#
    #        mp_e_statements = get_all_statements_of_entity(e)
    #        extended_paths = get_statements_by_id(mp_e_statements, e, mp[1][i_e+1][:mp[1][i_e+1].find("-")], qualifier=False, statement_type="predicate")
    #        extended_paths_qualifier = get_statements_by_id(mp_e_statements, e, mp[1][i_e+1][:mp[1][i_e+1].find("-")], qualifier=True, statement_type="qualifier_predicate")
#
    #        for ep in extended_paths_qualifier:
    #            if (ep['entity']['id'] == mp[1][i_e] and 
    #                ep['predicate']['id'] == mp[1][i_e-1][:mp[1][i_e-1].find("-")] and
    #                ep['object']['id'] == mp[1][i_e-2] and
    #                ep['qualifiers']):
    #                for q in ep['qualifiers']:
    #                    if(q['qualifier_predicate']["id"] == mp[1][i_e+1][:mp[1][i_e+1].find("-")] and
    #                      q['qualifier_object']["id"] == mp[1][i_e+2]):
    #                        if mp[1] not in golden_paths:
    #                            golden_paths.append(mp[1])
#
    #            if (ep['entity']['id'] == mp[1][i_e+2] and 
    #                ep['predicate']['id'] == mp[1][i_e+1][:mp[1][i_e+1].find("-")] and
    #                ep['object']['id'] == mp[1][i_e] and
    #                ep['qualifiers']):
    #                for q in ep['qualifiers']:
    #                    if(q['qualifier_predicate']["id"] == mp[1][i_e-1][:mp[1][i_e-1].find("-")] and
    #                      q['qualifier_object']["id"] == mp[1][i_e-2]):
    #                        if mp[1] not in golden_paths:
    #                            golden_paths.append(mp[1])
#
    #        for ep in extended_paths:
    #            if (ep['entity']['id'] == mp[1][i_e] and 
    #                ep['predicate']['id'] == mp[1][i_e-1][:mp[1][i_e-1].find("-")] and
    #                ep['object']['id'] == mp[1][i_e-2] and
    #                ep['qualifiers']):
    #                for q in ep['qualifiers']:
    #                    if(q['qualifier_predicate']["id"] == mp[1][i_e+1][:mp[1][i_e+1].find("-")] and
    #                      q['qualifier_object']["id"] == mp[1][i_e+2]):
    #                        if mp[1] not in golden_paths:
    #                            golden_paths.append(mp[1])
#
    #            if (ep['entity']['id'] == mp[1][i_e+2] and 
    #                ep['predicate']['id'] == mp[1][i_e+1][:mp[1][i_e+1].find("-")] and
    #                ep['object']['id'] == mp[1][i_e] and
    #                ep['qualifiers']):
    #                for q in ep['qualifiers']:
    #                    if(q['qualifier_predicate']["id"] == mp[1][i_e-1][:mp[1][i_e-1].find("-")] and
    #                      q['qualifier_object']["id"] == mp[1][i_e-2]):
    #                        if mp[1] not in golden_paths:
    #                            golden_paths.append(mp[1])    
    #
    #print("len(golden_paths)",len(golden_paths))
    sorted_golden_paths = []
    #for gp in golden_paths:
    #    tmp_gp = []
    #    #if gp[0] in [h[0] for h in hypothesises]:
    #    for e in gp:
    #        if is_wd_entity(e):
    #            tmp_gp.append(get_wd_label(e))
    #        else:
    #            tmp_gp.append(get_wd_label(e[:e.find("-")]))
    #    nlp_gp = get_nlp(" ".join(tmp_gp))
    #    sorted_golden_paths.append((get_similarity_by_words(question,nlp_gp), gp))
#
    #sorted_golden_paths = sorted(sorted_golden_paths, key=lambda x: x[0], reverse=True)
    #print("len(sorted_golden_paths) BEFORE",len(sorted_golden_paths))
    
    sorted_golden_paths = [sgp[1] for sgp in sorted_golden_paths]
    
    if not sorted_golden_paths: 
        for lp in [lp[1] for lp in looped_paths]:
            if lp[0] == hypothesises[0][0]:
                if lp not in sorted_golden_paths:
                    sorted_golden_paths.append(lp)
            if lp[-1] == hypothesises[0][0]:
                lp = list(reversed(lp))
                if lp not in sorted_golden_paths:
                    sorted_golden_paths.append(lp)
    
    #print("len(sorted_golden_paths) AFTER",len(sorted_golden_paths))
        
    if not sorted_golden_paths: 
        for p in paths:
            #print(p)
            if p[0] == hypothesises[0][0]:
                #print(p)
                if p not in sorted_golden_paths:
                    sorted_golden_paths.append(p)
            if p[-1] == hypothesises[0][0]:
                p = list(reversed(p))
                if p not in sorted_golden_paths:
                    sorted_golden_paths.append(p)
                    
    #print("len(sorted_golden_paths) AFTER AFTER",len(sorted_golden_paths))
    
    if not sorted_golden_paths:
        for p in paths:
            if hypothesises[0][0] in p:
                if p not in sorted_golden_paths:
                    sorted_golden_paths.append(p)
                    
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
    
    golden_unique_paths = golden_paths_filtered.copy()
    for i_sgp, sgp in enumerate(golden_paths_filtered):
        for sgp_2 in golden_paths_filtered:
            if (is_sublist(sgp, sgp_2) and sgp!=sgp_2):
                golden_unique_paths[i_sgp] = []
                break
    
    golden_unique_paths = [gup for gup in golden_unique_paths if gup]
    hypothesises_names = [h[0] for h in hypothesises]
    
    #print("golden_paths_filtered",golden_paths_filtered)
    #print("before hypothesises_names",hypothesises_names)
    #print("golden_unique_paths[0][0]",golden_unique_paths[0][0])
    #print("hypothesises_names",hypothesises_names)
    
    #if is_valide_wd_id(hypothesises_names[0]):
    
    if golden_unique_paths and hypothesises_names:
        if golden_unique_paths[0] and hypothesises_names[0]:
            if golden_unique_paths[0][0]:
                if (not is_wd_entity(hypothesises_names[0])
                    and is_wd_entity(golden_unique_paths[0][0]) 
                    or hypothesises_names[0] != golden_unique_paths[0][0]):
                    if golden_unique_paths[0][0] in hypothesises_names:
                        hypothesises_names.pop(hypothesises_names.index(golden_unique_paths[0][0]))
                        hypothesises_names.insert(0,golden_unique_paths[0][0])
    
    #print("after hypothesises_names",hypothesises_names)
            
    golden_unique_paths = [gup for gup in golden_unique_paths if gup[0] == hypothesises_names[0]]
            
    #elif hypothesises_names[0] != golden_unique_paths[0][0]
    
    golden_unique_paths = [hypothesises_names]+golden_unique_paths
    
    return golden_unique_paths


#start_time = time.time()
#golden_paths_2 = match_hypothesises(graph_2, q_nlp_2, q_themes_2, q_predicates_2, hypothesises_2, paths_nodes_filtered_2)
#end_time = time.time()
#print("Golden paths ->\tRunning time is {}s".format(round(end_time-start_time,2)))
#print(golden_paths_2)


# In[86]:


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

def answer_question(question, verbose=False, aggressive=False, looped=False, deep_k=30, deep_k_step=20, timer=False, g_paths=False):
    if verbose: start_time = time.time()
    if timer: timer_time = time.time()
    if verbose: print("Auto correcting question:",question)
    q_nlp = get_nlp(question, autocorrect=True)
    if verbose: print("-> q_nlp:",q_nlp)
    q_themes = get_themes(q_nlp, top_k=2)
    if verbose: print("-> q_themes:",q_themes)
    q_themes_enhanced = get_enhanced_themes(q_themes, top_k=1, aggressive=aggressive)
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
    if deep_k<=10:
        deep_k = 10
        graph, predicates_dict = build_graph(q_nlp, q_themes, q_themes_enhanced, q_predicates, deep_k=deep_k)
        if timer: 
            print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
            timer_time = time.time()
    else:
        for k in range(10, deep_k, deep_k_step):
            graph, predicates_dict = build_graph(q_nlp, q_themes, q_themes_enhanced, q_predicates, deep_k=deep_k)
            if timer: 
                print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
                timer_time = time.time()
            if graph.size() > 1000 or len(graph) > 1000 or deep_k<=10:
                break
            elif graph.size() > 500 or len(graph) > 500:
                deep_k -= deep_k_step
                if verbose: print("---> Rebuilding the graph with k_deep",str(deep_k), "... Previously:",len(graph), "nodes or", graph.size(), "edges was above the limit...")
            else: break
    if verbose: print("--> ",len(graph), "nodes and", graph.size(), "edges")
    #if graph.size() > 510 or len(graph) > 510:
    #    if verbose: print("Stopping the computing here, too computational.")
    #    return False
    if verbose: print("-> predicates_dict:",predicates_dict)
    paths_keywords = find_paths_keywords(graph, q_nlp, q_themes, q_themes_enhanced, q_predicates, q_focused_parts)
    if verbose: print("-> paths_keywords:",paths_keywords)
    if timer: timer_time = time.time()
    if verbose: print("-> Computing possible paths... (could be long)")
    path_nodes = find_path_nodes_from_graph(graph, paths_keywords, threshold=0.8, thres_inter=0.1, top_performance=graph.size(),min_paths=3000)
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
        paths_nodes_filtered = path_nodes
    if verbose: print("-> Computing hypothesises...")
    hypothesises = get_hypothesises(q_nlp, q_predicates, q_themes, paths_keywords, paths_nodes_filtered, threshold=0.5, max_reward=2.0) 
    if verbose: print("--> hypothesises:",hypothesises)
    if timer: 
        print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
        timer_time = time.time()
    if g_paths:
        if hypothesises:
            if verbose: print("-> Computing golden paths...")
            golden_paths = match_hypothesises(graph, q_nlp, q_themes, q_predicates, hypothesises, paths_nodes_filtered, threshold=0.8, max_reward=2.0)
            if verbose: print("--> len(golden_paths):",len(golden_paths)-1)
            if timer: 
                print("->\tRunning time is {}s".format(round(time.time()-timer_time,2)))
                timer_time = time.time()
        else:
            if not looped:
                if verbose: print("-> Looping on aggressive mode...")
                golden_paths = answer_question(question, verbose=verbose, aggressive=True, looped=True, deep_k=deep_k)
            else: 
                if verbose: print("--> End of loop")
                golden_paths=[]

    save_cache_data()
    
    if g_paths:
        if golden_paths:
            cleared_golden_paths = [golden_paths[0].copy()]
            for p in golden_paths[1:]:
                tmp_translation = []
                for e in p:
                    if is_wd_entity(e) or is_wd_predicate(e):
                        tmp_translation.append(e)
                    else:
                        if e[0] == "P": tmp_translation.append(e[:e.find("-")])
                        else: tmp_translation.append(e)
                if tmp_translation not in cleared_golden_paths:
                    cleared_golden_paths.append(tmp_translation)

            if verbose: print("--> len(cleared_golden_paths):",len(cleared_golden_paths)-1)
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
#answer = answer_question("Which actor voiced the Unicorn in The Last Unicorn?", verbose=True, timer=True) #works 312.12s
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
#answer = answer_question("Where did roger marquis die", verbose=True, timer=True) # works 64.56s
#answer = answer_question("Which genre of album is harder.....faster?", verbose=True, timer=True)
#answer = answer_question("Which equestrian was born in dublin?", verbose=True, timer=True)
#answer = answer_question("Who is the author that wrote the book Moby Dick", verbose=True, timer=True) #314.04s works
#answer = answer_question("Name a person who died from bleeding.", verbose=True, timer=True) # 117.35s
#answer = answer_question("What is the name of the person who created Saved by the Bell?", verbose=True, timer=True)
#answer = answer_question("of what nationality is ken mcgoogan", verbose=True, timer=True) #works 51.39s

#answer = answer_question("Which actor voiced the Unicorn in The Last Unicorn?", verbose=True, timer=True, g_paths=False)
#if answer:
#    print("Answer:",get_wd_label(answer[0][0]), "("+str(answer[0][0])+")")
#    print("Paths:",[[get_wd_label(e) for e in row] for row in answer[1:]])


# In[88]:


#questions = [
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
#            ]
#
#for i_q, question in enumerate(questions):
#    if i_q >= 12:
#        answer = answer_question(question, verbose=True, timer=True, g_paths=False)
#        if answer:
#            print("Answer:",get_wd_label(answer[0][0]), "("+str(answer[0][0])+")\n")
#    


# In[67]:


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


# In[68]:


#graph_2.has_node("Q13133")


# In[69]:


#get_similarity_by_words(get_nlp("mia farrow"), get_nlp("farrow mia")) #1.000000077374981
#get_similarity_by_words(get_nlp("actor voiced"), get_nlp("voice actor")) #0.8541489425987572 
#get_similarity_by_words(get_nlp("actor voiced the unicorn in the last unicorn"), 
#                        get_nlp("the unicorn last unicorn actor voiced")) #0.9573255410217848
#get_similarity_by_words(get_nlp("voice actor"),get_nlp("instance of")) #0.30931508860569823
#get_similarity_by_words(get_nlp("voice actor"),get_nlp("present in work")) #0.34966764303274056
#get_similarity_by_words(get_nlp("voice actor"),get_nlp("subject has role")) #0.5026860362728758

#get_similarity_by_words(get_nlp("voice actor"),get_nlp("protagonist")) #0.4688377364169893


# In[70]:


#subgraphs = [graph.subgraph(c) for c in nx.connected_components(graph)]
#print(len(subgraphs))
#[len(s.nodes) for s in subgraphs]
#len(subgraphs[0].nodes)


# In[71]:


#for path in nx.all_simple_paths(graph, source="Q176198", target="Q202725"):
#    print(path)


# In[72]:


#nx.shortest_path(graph, source="Q176198", target="Q202725")


# In[73]:


#nlp_lookup_test = get_nlp("klein yves")
#[y['name'] for x,y in graph.nodes(data=True) if get_nlp(y['name']).similarity(nlp_lookup_test) >= 0.9]


# In[74]:


#list(nx.dfs_labeled_edges(graph, source=get_themes(q0_nlp, top_k=3)[0][0][1][0], depth_limit=4))[0]


# In[75]:


#plot_graph(graph_2, "test_file_name_graph", "Graph_title")


# In[76]:


#nlp_lookup_test = get_nlp("klein yves")
#[y['name'] for x,y in graph.nodes(data=True) if get_nlp(y['name']).similarity(nlp_lookup_test) >= 0.9]
#[y['name'] for x,y in graph_2.nodes(data=True) if y['type'] == 'predicate']


# In[77]:


#get_nlp(get_wd_label("Q13133")).similarity(get_nlp("person"))


# In[78]:


#get_similarity_by_words(get_nlp("PERSON"),get_nlp("person"))


# In[79]:


#get_similarity_by_words(get_nlp("GPE"),get_nlp("location"))


# In[80]:


#get_most_similar("Michelle Obama", topn=5)


# In[81]:


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





# In[82]:


#nlp.get_vector("Q42")


# In[ ]:





# In[ ]:




