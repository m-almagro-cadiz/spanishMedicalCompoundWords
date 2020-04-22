import re, tables
from unidecode import unidecode
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

pathCieSnFrequencies = 'data/cie_sn_frequencies.csv'
pathTags = 'data/tags.csv'
pathLemmas = 'data/extendedLemmas.csv'
pathPrefixes = 'data/medicalPrefixes.csv'
pathOtherPrefixes = 'data/otherMedicalPrefixes.csv'
pathSuffixes = 'data/medicalSuffixes.csv'
pathConceptNetEmbeddings = 'wordEmbeddings.h5'

with open(pathCieSnFrequencies, encoding='utf8') as f: freq = Counter({line.strip().split('	')[0]:int(line.strip().split('	')[1]) for line in re.split('\n', f.read()) if line})

with open(pathTags, 'r', encoding='utf8') as f: wordTag = {line.strip().split('\t')[0]:{element.split(' ')[0]:element.split(' ')[1] for element in line.strip().split('\t')[1::]} for line in re.split('\n', f.read())}

with open(pathLemmas, 'r', encoding='utf8') as f: lemmaList = [line.strip().split('\t') for line in re.split('\n', f.read())]

with open(pathPrefixes, encoding='utf8') as f: medicalPrefixes = {prefix.split('\t')[0].lower().replace('-', '').strip():prefix.split('\t')[1].strip() if len(prefix.split('\t')) > 1 else '' for prefix in re.split('\n', f.read()) if prefix}

with open(pathOtherPrefixes, encoding='utf8') as f: otherMedicalPrefixes = {prefix.split('\t')[0]:prefix.split('\t')[1] if len(prefix.split('\t')) > 1 else '' for prefix in re.split('\n', f.read()) if prefix}

with open(pathSuffixes, encoding='utf8') as f: medicalSuffixes = {prefix.split('\t')[0].lower().replace('-', '').strip():prefix.split('\t')[1].strip() if len(prefix.split('\t')) > 1 else '' for prefix in re.split('\n', f.read()) if prefix}

nb = pd.read_hdf(pathConceptNetEmbeddings)

def getAbsFreq(word):
   lemmas_ = [word]
   if word in splittedLemmas:
      lemmas_ = splittedLemmas[word]
   elif word not in freq:
      word_ = unidecode(word).lower()
      if word_ in splittedLemmas:
         lemmas_ = splittedLemmas[word_]
   f = 0
   for lemma_ in lemmas_:
      if lemma_ in freq:
         f += freq[lemma_]
   return f

def getRelFreq(word):
   return getAbsFreq(word) / freqAcc

def getTags(word):
   return tagger[word] if word in tagger else {}

def isSharedElement(tags1, tags2):
   shared = False
   for tag in tags1:
      if tag in tags2:
         shared = True
         break
   return shared

def getCandidateSubwords(word, prefixes, suffixes=[], reverse=False):
   possibilities = list()
   if len(word) > 3:
      word_ = ''.join(reversed(word)) if reverse else word
      i, subwords, subwords__ = 0, [[(list(), word_.lower())]], list()
      while i < len(prefixes):
         if len(subwords) == 1 or len(prefixes[i]) > 2:
            for subword in subwords[-1]:
               if subword[1].startswith(prefixes[i]):
                  #Getting subword
                  subword_ = re.sub(r'^\W+', '', subword[1][len(prefixes[i])::]) #Deleting '-'
                  if subword_ and prefixes[i][-1] in ['a', 'e', 'i', 'o', 'u']: #Deleting duplicated r
                     subword_ = re.sub(r'^r(r)', r'\1', subword_)
                     if subword_ and len(prefixes[i]) > 1 and prefixes[i][-1] != subword_[0]: #Adding missing letter
                        subwords__.append((subword[0] + [prefixes[i]], prefixes[i][-1] + subword_))
                  startsWith_bp = prefixes[i][-1] != 'm' or (len(subword_) > 0 and subword_[0] in ['b', 'p'])
                  notStartsWith_bp = prefixes[i][-1] != 'n' or (len(subword_) > 0 and subword_[0] not in ['b', 'p'])
                  if (startsWith_bp and notStartsWith_bp) or reverse:
                     subwords__.append((subword[0] + [prefixes[i]], subword_))
         i += 1
         if i == len(prefixes) and len(subwords__) > 0:
            subwords.append(subwords__)
            i, subwords__ = 0, list()
      subwords = [([''.join(reversed(prefix)) for prefix in reversed(e[0])], ''.join(reversed(e[1]))) if reverse else (e[0], e[1]) for group in reversed(subwords[1::]) for e in group]
      wtag, fWord = getTags(word), getRelFreq(word) #Getting whole word tag and frequency
      for subword in subwords:
         fSubword = getRelFreq(subword[1])
         notShort_ = len(subword[1]) > 3
         inSuffixes_ = subword[1] in suffixes
         if (fSubword > 0 and notShort_):
            swtag = getTags(subword[1]) #Getting subword tag
            matchingTag = isSharedElement(wtag, swtag) #Comparing tags
            matching = len(wtag) == 0 or len(swtag) == 0 or matchingTag
            isVerb = ('v' in wtag and len(wtag) == 1) or ('v' in swtag and len(swtag) == 1)
            em = -1
            if word in names_es and subword[1] in names_es:
               em = cosine_similarity([nb.loc[names_es[word]].values, nb.loc[names_es[subword[1]]].values])[0][1]
            elif subword[1] in names_es:
               em = -2
            if fSubword == 0:
               rate_ = -1
            else:
               rate_ = fWord  / fSubword
            possibilities.append((subword, fWord, fSubword, rate_, em, 1 if matching else 0, 1 if isVerb else 0, len(subword[1]), len(subword[0]), len(''.join(subword[0])), 1 if inSuffixes_ else 0))
      possibilities = sorted(possibilities, key=lambda tup: (-tup[7], tup[8], tup[3], -tup[4]))
   return possibilities

def isCandidate(candidate):
   if (candidate[3] < 5) or (candidate[2] > 0.00001 and candidate[3] < 10):
      return True
   else:
      return False

def selectCandidate(candidates):
   candidates_ = [candidate for candidate in candidates if isCandidate(candidate)]
   if len(candidates_) > 0:
      min_l_subword = min([candidate[7] for candidate in candidates_])
      min_n_prefix = min([candidate[8] for candidate in candidates_ if candidate[7] == min_l_subword])
      min_l_prefix = min([candidate[9] for candidate in candidates_ if candidate[8] == min_n_prefix])
      min_conditions = [1 if candidate[7] == min_l_subword and candidate[8] == min_n_prefix and candidate[9] == min_l_prefix else 0 for candidate in candidates_]
      conditions = [1 if min_conditions[i] == 1 and candidates_[i][6] == 0 and candidates_[i][2] > 0.0000005 and (candidates_[i][4] >= 0.4 or candidates_[i][4] <= -1) else 0 for i in range(len(candidates_))]
      candidates_ = [candidates_[i] for i in range(len(candidates_)) if conditions[i] == 1]
   return candidates_

def divideCompoundWord(word):
   candidates = getCandidateSubwords(word, prefixes, suffixes)
   compositions = selectCandidate(candidates)
   if len(compositions) > 0:
      return [medicalPrefixes[prefix] for prefix in compositions[0][0][0]] + [compositions[0][0][1]]
   else:
      return [word]

#Building lemmas dictionary
lemmas = dict()
for line in lemmaList:
   key = line[0].split(' ')
   if len(key) == 1:
      key.append('')
   key = tuple(key)
   values = line[1::]
   for v in range(len(values)):
      value = values[v].split(' ')
      if len(value) == 1:
         value.append('')
      value = tuple(value)
      values[v] = value
   if key not in lemmas:
      lemmas[key] = list()
   lemmas[key] = values

#Extending tags from lemmas
tagger = {word:{tag[0] for tag in wordTag[word]} for word in wordTag}
for lemma in lemmas:
   if lemma[0] not in tagger:
      tagger[lemma[0]] = set()
   tagger[lemma[0]].add(lemma[1])

#Collecting derivational words without tags
splittedLemmas = dict()
for lemma in lemmas:
   if lemma[0] not in splittedLemmas:
      splittedLemmas[lemma[0]] = set()
   splittedLemmas[lemma[0]].update([lemma_[0] for lemma_ in lemmas[lemma]])
   word_ = unidecode(lemma[0]).lower()
   if word_ not in lemmas:
      if word_ not in splittedLemmas:
         splittedLemmas[word_] = set()
      splittedLemmas[word_].update([lemma_[0] for lemma_ in lemmas[lemma]])

#Collecting words in conceptNet embeddings
names_es = nb.index.values.tolist()
names_es = [name for name in names_es if name.split('/')[2]=='es']
names_es = {w.split('/')[3]:w for w in names_es}

#Gathering word counts from CIE-10-ES and SNOMED-CT using lemmas
freq_ = dict()
for w in freq:
   lemmas_ = [w]
   if w in splittedLemmas:
      lemmas_ = splittedLemmas[w]
   f = freq[w]
   for lemma_ in lemmas_:
      if lemma_ not in freq_:
         freq_[lemma_] = 0
      freq_[lemma_] += f

freq = Counter(freq_)
freqAcc = sum([freq[w] for w in freq])

medicalPrefixes.update(otherMedicalPrefixes)
prefixes = list(set(medicalPrefixes.keys()))
suffixes = list(set(medicalSuffixes.keys()))

from ipywidgets import widgets
labelResult = widgets.Label()
labelBox = widgets.Label('Compound word')
textBox = widgets.Text()
button = widgets.Button(description='Submit')

def submit_(b):
    division_ = divideCompoundWord(textBox.value)
    labelResult.value = ' '.join(division_)

button.on_click(submit_)
