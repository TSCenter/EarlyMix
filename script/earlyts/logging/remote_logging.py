# coding=utf-8
from pathlib import Path
import os
import requests

telegram_config_path = os.path.join(Path.home(), ".telegram.config")

use_telegram = os.path.exists(telegram_config_path)

print("use_telegram = {}".format(use_telegram))

if use_telegram:
    with open(telegram_config_path, "r", encoding="utf-8") as f:
        telegram_url = f.readline().strip()

    def telegram_send(msg):
        requests.get(telegram_url, params={"msg": msg})


def log(msg):
    print(msg)
    try:
        if use_telegram:
            telegram_send(msg)
    except Exception as e:
        print(e)
