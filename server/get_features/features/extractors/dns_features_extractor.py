from difflib import SequenceMatcher
from itertools import combinations as combs
import os.path
import re
from statistics import mean, pstdev, StatisticsError

import dns.exception
import dns.resolver
import dns.reversename
from geoip2.database import Reader

packagedir = os.path.dirname(__file__)
dbpath = os.path.join(packagedir, '../../thirdparty/geoip/GeoLite2-City.mmdb')
city_reader = Reader(dbpath)


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


class DNS_Features_Extractor:
    def __init__(self, domain):
        self.domain = re.sub(r'^www\d*', '', domain.lower()).lstrip('.')

        servers = ['1.1.1.1', '8.8.8.8', '208.67.222.222']
        self.resolvers = []

        for server in servers:
            resolver = dns.resolver.Resolver()
            resolver.nameservers = [server]
            self.resolvers.append(resolver)

        self.TTLs = list()
        #self.ips = self.__get_rr('A', ttl=True)
        self.ips = None
        self.ns_names = None
        self.mx_names = None


    def get_n_ip(self):
        if not self.ips:
            self.ips = self.__get_rr('A', ttl=True)
        return len(self.ips)


    def get_n_mx(self):
        self.mx_names = self.__get_rr('MX')
        return len(self.mx_names)


    def get_n_ns(self):
        self.ns_names = self.__get_rr('NS')
        return len(self.ns_names)


    def get_n_ptr(self):
        if not self.ips:
            self.ips = self.__get_rr('A', ttl=True)
        cloudflare = dns.resolver.Resolver()
        cloudflare.nameservers = ['1.1.1.1']
        ptr_names = set()
        for ip in self.ips:
            rev_name = dns.reversename.from_address(ip)
            try:
                ptr_records = cloudflare.resolve(rev_name, 'PTR')
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer,
                dns.exception.Timeout, dns.resolver.NoNameservers):
                continue
            for record in ptr_records:
                ptr_names.add(str(record))
        return len(ptr_names)


    def get_ns_similarity(self):
        if not self.ns_names:
            self.get_n_ns()
        if not self.ips:
            self.ips = self.__get_rr('A', ttl=True)

        if len(self.ns_names) > 2:
            similarities = list()
            all_combs = combs(self.ns_names, 2)
            for comb in all_combs:
                similarities.append(similarity(*comb))
            return sum(similarities) / len(similarities)
        elif len(self.ips) == 0 or len(self.ns_names) == 0:
            return 0.0
        else:
            return 1.0


    def get_n_countries(self):
        if not self.ips:
            self.ips = self.__get_rr('A', ttl=True)
        ip_countries = set()
        for ip in self.ips:
            try:
                city_resp = city_reader.city(ip)
                ip_countries.add(city_resp.country.iso_code)
            except:
                pass

        return len(ip_countries)


    def get_mean_TTL(self):
        if not self.ips:
            self.ips = self.__get_rr('A', ttl=True)
        try:
            return mean(self.TTLs)
        except StatisticsError:
            return 0.0
            

    def get_stdev_TTL(self):
        if not self.ips:
            self.ips = self.__get_rr('A', ttl=True)
        try:
            return pstdev(self.TTLs)
        except StatisticsError:
            return 0.0


    def __get_rr(self, _type, ttl=False):
        names = set()
        for resolver in self.resolvers:
            try:
                records = resolver.resolve(self.domain, _type)
                for record in records:
                    names.add(str(record))
                if ttl:
                    self.TTLs.append(records.rrset.ttl)
            except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer,
                dns.exception.Timeout, dns.resolver.NoNameservers, dns.name.LabelTooLong):
                pass
        return names
