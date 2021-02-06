import http.client
from io import BytesIO
import os
import socket
import sys
import urllib.error
import urllib.request

from bs4 import BeautifulSoup
import tldextract as tld
import whois


def disable_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0'}


def get_html(url):
    request = urllib.request.Request('http://'+url, headers=headers)
    try:
        resp = urllib.request.urlopen(request, timeout=5)
        return resp.read()
    except (http.client.BadStatusLine, http.client.IncompleteRead, http.client.HTTPException,
        UnicodeError, UnicodeEncodeError): # possibly plaintext or HTTP/1.0
        print("error")
        return None
    except:
        raise


class Web_Features_Extractor:
    def __init__(self, domain):
        self.domain = domain.rstrip('\n')
        ext = tld.extract(self.domain)
        self.compact_domain = '.'.join(filter(None, [ext.domain, ext.suffix]))

        self.__whois = False
        self.create = None
        self.update = None
        self.expire = None


    def get_n_labels(self):
        html = None
        try:
            html = get_html(self.domain)
        except (urllib.error.HTTPError, urllib.error.URLError, 
            ConnectionResetError, socket.timeout):
            try:
                html = get_html(self.compact_domain)
            except (urllib.error.HTTPError, urllib.error.URLError, 
                ConnectionResetError, socket.timeout):
                print("error")
                pass

        if html:
            try:
                soup = BeautifulSoup(html, features='html.parser')
                return len(soup.find_all())
            except UnboundLocalError:
                print("error")
                return 0
        else:
            return 0


    def get_life_time(self):
        if not self.__whois:
            self.__get_whois()

        if self.expire and self.create:
            td = self.expire - self.create
            return td.days
        else:
            return 0


    def get_active_time(self):
        if not self.__whois:
            self.__get_whois()

        if self.update and self.create:
            td = self.update - self.create
            return td.days
        else:
            return self.get_life_time()


    def __get_whois(self):
        try:
            disable_print()
            w = whois.whois(self.compact_domain)
            enable_print()
            if w.creation_date:
                if isinstance(w.creation_date, list):
                    self.create = w.creation_date[0]
                elif isinstance(w.creation_date, str):
                    pass
                else:
                    self.create = w.creation_date
                if isinstance(self.create, str):
                    self.create = None
            if w.updated_date:
                if isinstance(w.updated_date, list):
                    self.update = w.updated_date[0]
                elif isinstance(w.updated_date, str):
                    pass 
                else:
                    self.update = w.updated_date
                if isinstance(self.update, str):
                    self.update = None
            if w.expiration_date:
                if isinstance(w.expiration_date, list):
                    self.expire = w.expiration_date[0]
                elif isinstance(w.expiration_date, str):
                    pass 
                else:
                    self.expire = w.expiration_date
                if isinstance(self.expire, str):
                    self.expire = None
        except whois.parser.PywhoisError:
            pass
        self.__whois = True
