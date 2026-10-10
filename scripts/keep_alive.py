#!/usr/bin/env python
"""Open the deployed Streamlit demo in a real browser so that the app counts as visited and does not go to sleep.

Streamlit Community Cloud puts an app to sleep after a period without visitors, and a plain HTTP request does not count because the app only runs
once a browser opens its websocket session. This script therefore loads the page in headless Chromium, clicks "Yes, get this app back up!" if the
sleep screen is shown, waits for the app's headline to render, stays a few seconds so the session registers, and exits non-zero if the app never comes up.

  APP_URL=https://beyond-the-count1.streamlit.app/ python scripts/keep_alive.py
Environment: APP_URL (default below), BROWSER_CHANNEL (e.g. chrome, to use an installed Chrome instead of the Playwright Chromium), WAIT_SECONDS.
"""
import os
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("APP_URL", "https://beyond-the-count1.streamlit.app/")
MARKER = "Does a Cell Painting model see more than a cell count?"
WAIT_SECONDS = int(os.environ.get("WAIT_SECONDS", "300"))
STAY_SECONDS = 15


def app_is_up(page) -> bool:
    for frame in page.frames:  # Streamlit Cloud renders the app inside an iframe
        try:
            if frame.get_by_text(MARKER).first.is_visible(timeout=500):
                return True
        except Exception:
            continue
    return False


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get("BROWSER_CHANNEL") or None, headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(URL, wait_until="domcontentloaded", timeout=90_000)
        woke, deadline = False, time.time() + WAIT_SECONDS
        while time.time() < deadline:
            wake = page.get_by_role("button", name=re.compile("get this app back up", re.I))
            if wake.count():
                wake.first.click()
                woke = True
                print("sleep screen found, asked the app to wake up", flush=True)
                time.sleep(10)
            if app_is_up(page):
                time.sleep(STAY_SECONDS)
                print(f"app is up{' (it was asleep)' if woke else ''}: {URL}", flush=True)
                browser.close()
                return 0
            time.sleep(5)
        page.screenshot(path="keep_alive_failure.png", full_page=True)
        print(f"app did not render within {WAIT_SECONDS}s: {URL}", file=sys.stderr, flush=True)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
