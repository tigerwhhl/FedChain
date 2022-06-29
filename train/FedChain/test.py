import random
import time

import schedule
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed,wait, ALL_COMPLETED, FIRST_COMPLETED

class Test(object):
    def __init__(self):
        print("init")
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(self.monitor_blockchain, 1)

    def monitor_blockchain(self,interval):
        session = requests.Session()
        org_server = "http://114.212.82.242:8080/"
        monitor_api = "test"

        def monitor_block():
            res = session.get(org_server + monitor_api).content
            block = json.loads(res) #dict
            print(block)

        schedule.clear()
        schedule.every(interval).seconds.do(monitor_block)
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == '__main__':
    test = Test()

