import corenlp

text = "Chris wrote a simple sentence that he parsed with Stanford CoreNLP."

# We assume that you've downloaded Stanford CoreNLP and defined an environment
# variable $CORENLP_HOME that points to the unzipped directory.
# The code below will launch StanfordCoreNLPServer in the background
# and communicate with the server to annotate the sentence.
with corenlp.CoreNLPClient(annotators="tokenize ssplit pos lemma ner depparse".split()) as client:
  ann = client.annotate(text)

# You can access annotations using ann.
sentence = ann.sentence[0]

# The corenlp.to_text function is a helper function that
# reconstructs a sentence from tokens.
assert corenlp.to_text(sentence) == text

# You can access any property within a sentence.
print(sentence.text)

# Likewise for tokens
token = sentence.token[0]
print(token.lemma)

# Use tokensregex patterns to find who wrote a sentence.
pattern = '([ner: PERSON]+) /wrote/ /an?/ []{0,3} /sentence|article/'
matches = client.tokensregex(text, pattern)
# sentences contains a list with matches for each sentence.
assert len(matches["sentences"]) == 1
# length tells you whether or not there are any matches in this
assert matches["sentences"][0]["length"] == 1
# You can access matches like most regex groups.
matches["sentences"][1]["0"]["text"] == "Chris wrote a simple sentence"
matches["sentences"][1]["0"]["1"]["text"] == "Chris"

# Use semgrex patterns to directly find who wrote what.
pattern = '{word:wrote} >nsubj {}=subject >dobj {}=object'
matches = client.semgrex(text, pattern)
# sentences contains a list with matches for each sentence.
assert len(matches["sentences"]) == 1
# length tells you whether or not there are any matches in this
assert matches["sentences"][0]["length"] == 1
# You can access matches like most regex groups.
matches["sentences"][1]["0"]["text"] == "wrote"
matches["sentences"][1]["0"]["$subject"]["text"] == "Chris"
matches["sentences"][1]["0"]["$object"]["text"] == "sentence"