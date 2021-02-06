import json
import math
import os.path
import re

import tldextract as tld

packagedir = os.path.dirname(__file__)
ngram_table = None
with open(os.path.join(packagedir, 'ngram_table.json')) as f:
    ngram_table = json.loads(f.readline())


def ngram(word, n):
    return [word[i:i+n] for i in range(len(word)-n+1)]


class Domain_Name_Features_Extractor:
    def __init__(self, domain):
        self.domain = re.sub(r'^www\d*', '', domain.lower()).lstrip('.')
        self.__ext = tld.extract(self.domain)
        self.length = len(self.domain)
        self.vowel_constant_convs = None
        self.alpha_numer_convs = None

    def get_length(self):
        return self.length

    def get_n_vowel_chars(self):
        return sum([self.domain.count(c) for c in 'aeiou'])

    def get_vowel_ratio(self):
        return self.get_n_vowel_chars() / self.get_length()

    def get_n_vowels(self):
        return sum([c in self.domain for c in 'aeiou'])

    def get_n_constant_chars(self):
        return sum([self.domain.count(c) for c in 'bcdfghjklmnpqrstvwxyz'])

    def get_n_constants(self):
        return sum([c in self.domain for c in 'bcdfghjklmnpqrstvwxyz'])

    def get_vowel_constant_convs(self):
        if not self.vowel_constant_convs:
            self.__calculate_convs()
        return self.vowel_constant_convs

    def get_n_nums(self):
        return sum([self.domain.count(c) for c in '0123456789'])

    def get_num_ratio(self):
        return self.get_n_nums() / self.get_length()

    def get_alpha_numer_convs(self):
        if not self.alpha_numer_convs:
            self.__calculate_convs()
        return self.alpha_numer_convs

    def get_n_other_chars(self):
        return sum([c not in 'abcdefghijklmnopqrstuvwxyz0123456789.' for c in self.domain])

    def get_max_consecutive_chars(self):
        match = re.findall(r'(([A-Za-z\-])\2*)', self.domain)
        if match:
            return max([len(group[0]) for group in match])
        else:
            return 0

    def get_rv(self):
        rv = 0.0
        for s in [self.__ext.subdomain, self.__ext.domain]:
            for i in range(3,8):
                grams = ngram(s, i)
                for gram in grams:
                    if gram in ngram_table:
                        rv += math.log2(ngram_table[gram] / i)
        return rv

    def get_entropy(self):
        probas = {i: self.domain.count(i)/len(self.domain) for i in set(self.domain)}
        return -sum((p * math.log2(p)) for p in probas.values())

    def __calculate_convs(self):
        self.vowel_constant_convs = 0
        self.alpha_numer_convs = 0
        prev_is_vowel = False
        prev_is_constant = False
        prev_is_num = False

        for c in self.domain:
            if c in 'aeiou':
                if prev_is_constant:
                    self.vowel_constant_convs += 1
                elif prev_is_num:
                    self.alpha_numer_convs += 1
                prev_is_constant = False
                prev_is_num = False
                prev_is_vowel = True
            elif c in 'bcdfghjklmnpqrstvwxyz':
                if prev_is_vowel:
                    self.vowel_constant_convs += 1
                elif prev_is_num:
                    self.alpha_numer_convs += 1
                prev_is_vowel = False
                prev_is_num = False
                prev_is_constant = True
            elif c in '0123456789':
                if prev_is_vowel or prev_is_constant:
                    self.alpha_numer_convs += 1
                prev_is_vowel = False
                prev_is_constant = False
                prev_is_num = True
