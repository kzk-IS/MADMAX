#Real-time training
The code performs real-time training every 30 seconds.
It is necessary to load a list of 25,000 benign domains (allow25000.csv) and a list of time-series malicious domains (deny_with_time.csv).
For ethical considerations, we have not included the list of raw data here.

#Recommended environment
・beautifulsoup4==4.9.3
・dnspython==2.0.0
・geoip2==4.1.0
・joblib==1.0.0
・numpy==1.19.4
・pandas==1.2.0
・tldextract==3.1.0
・whois==0.9.7
