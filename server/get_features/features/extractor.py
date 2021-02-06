from .extractors import *

class Extractor:
    def __init__(self, domain, name, dns, web):
        if name:
            self.domain_name_features_extractor = Domain_Name_Features_Extractor(domain)
        if dns:
            self.dns_features_extractor = DNS_Features_Extractor(domain)
        if web:
            self.web_features_extractor = Web_Features_Extractor(domain)


    def get_length(self):
        return self.domain_name_features_extractor.get_length()
    def get_n_vowel_chars(self):
        return self.domain_name_features_extractor.get_n_vowel_chars()
    def get_vowel_ratio(self):
        return self.domain_name_features_extractor.get_vowel_ratio()
    def get_n_vowels(self):
        return self.domain_name_features_extractor.get_n_vowels()
    def get_n_constant_chars(self):
        return self.domain_name_features_extractor.get_n_constant_chars()
    def get_n_constants(self):
        return self.domain_name_features_extractor.get_n_constants()
    def get_vowel_constant_convs(self):
        return self.domain_name_features_extractor.get_vowel_constant_convs()
    def get_n_nums(self):
        return self.domain_name_features_extractor.get_n_nums()
    def get_num_ratio(self):
        return self.domain_name_features_extractor.get_num_ratio()
    def get_alpha_numer_convs(self):
        return self.domain_name_features_extractor.get_alpha_numer_convs()
    def get_n_other_chars(self):
        return self.domain_name_features_extractor.get_n_other_chars()
    def get_max_consecutive_chars(self):
        return self.domain_name_features_extractor.get_max_consecutive_chars()
    def get_rv(self):
        return self.domain_name_features_extractor.get_rv()
    def get_entropy(self):
        return self.domain_name_features_extractor.get_entropy()


    def get_n_ip(self):
        return self.dns_features_extractor.get_n_ip()
    def get_n_mx(self):
        return self.dns_features_extractor.get_n_mx()
    def get_n_ns(self):
        return self.dns_features_extractor.get_n_ns()
    def get_n_ptr(self):
        return self.dns_features_extractor.get_n_ptr()
    def get_ns_similarity(self):
        return self.dns_features_extractor.get_ns_similarity()
    def get_n_countries(self):
        return self.dns_features_extractor.get_n_countries()
    def get_mean_TTL(self):
        return self.dns_features_extractor.get_mean_TTL()
    def get_stdev_TTL(self):
        return self.dns_features_extractor.get_stdev_TTL()


    def get_n_labels(self):
        return self.web_features_extractor.get_n_labels()
    def get_life_time(self):
        return self.web_features_extractor.get_life_time()
    def get_active_time(self):
        return self.web_features_extractor.get_active_time()
