import pandas as pd
import os
import logging
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
import time
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CSV_FILES = [
    "case_components.csv", "case_fan_components.csv", "cpu_components.csv",
    "cpu_cooler_components.csv", "custom_components.csv", "external_storage_components.csv",
    "fan_controller_components.csv", "headphones_components.csv", "keyboard_components.csv",
    "memory_components.csv", "monitor_components.csv", "motherboard_components.csv",
    "mouse_components.csv", "operating_system_components.csv", "optical_drive_components.csv",
    "power_supply_components.csv", "sound_card_components.csv",
    "speakers_components.csv", "storage_components.csv", "thermal_components.csv",
    "ups_components.csv", "video_card_components.csv", "webcam_components.csv",
    "wired_network_adapter_components.csv", "wireless_network_adapter_components.csv"
]

PROCESSED_FILE = "processed_components.csv"
LIST_SEPARATOR = "; "
NORMALIZE_UNITS = True
HEADLESS = False

def normalize_value(value):
    if not value or not NORMALIZE_UNITS:
        return value
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    value = value.replace('H₂O', 'H2O')
    if value.lower() == 'single':
        value = '1'
    return value

def sanitize_title(title):
    if not title:
        return ""
    title = re.sub(r'[\s/]+', '_', title)
    title = re.sub(r'[^\w_#.]', '', title)
    if title.startswith('Part'):
        title = title.replace('Part', 'Part_#')
    return title

def init_driver(port=9222):
    try:
        chrome_options = Options()
        chrome_options.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        if HEADLESS:
            chrome_options.add_argument("--headless=new")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        logging.info(f"Driver connected to Chrome on port {port}")
        return driver
    except Exception as e:
        logging.error(f"Error initializing driver: {e}")
        raise

def wait_for_js(driver):
    try:
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        time.sleep(2)
        WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                "div.block.xs-hide.md-block.specs section.module-subTitle h2"
            ))
        )
    except TimeoutException:
        logging.warning("JavaScript rendering or specs section timeout")

def scrape_component(component_link, component_type, component_name):
    driver = init_driver(port=9222)
    try:
        driver.get(component_link)
        logging.info(f"Navigating to {component_link}")
        wait_for_js(driver)
        specs_section = driver.find_element(
            By.XPATH,
            "//div[contains(@class, 'block xs-hide md-block specs') and .//h2[text()='Specifications']]"
        )
        spec_groups = specs_section.find_elements(By.CSS_SELECTOR, "div.group.group--spec")
        specs = {}
        title_selectors = ["h3.group__title", "[class*='group__title']", "div.group__content > strong"]
        for group in spec_groups:
            raw_title = None
            for selector in title_selectors:
                try:
                    title_elem = group.find_element(By.CSS_SELECTOR, selector)
                    raw_title = title_elem.text.strip()
                    if raw_title:
                        break
                except NoSuchElementException:
                    continue
            if not raw_title:
                try:
                    content_elem = group.find_element(By.CSS_SELECTOR, "div.group__content")
                    content_text = content_elem.text.strip()[:30].replace('\n', ' ')
                    raw_title = f"Inferred_{content_text}"
                except NoSuchElementException:
                    continue
            spec_title = sanitize_title(raw_title)
            try:
                content_elem = group.find_element(By.CSS_SELECTOR, "div.group__content")
                if content_elem.find_elements(By.CSS_SELECTOR, "ul"):
                    spec_values = [normalize_value(li.text.strip()) for li in content_elem.find_elements(By.CSS_SELECTOR, "ul li")]
                    specs[spec_title] = spec_values
                elif content_elem.find_elements(By.CSS_SELECTOR, "p"):
                    specs[spec_title] = normalize_value(content_elem.find_element(By.CSS_SELECTOR, "p").text.strip())
                else:
                    raw_text = normalize_value(content_elem.text.strip())
                    if raw_text:
                        specs[spec_title] = raw_text
            except NoSuchElementException:
                continue
        component_data = {
            "component_type": component_type,
            "component_name": component_name,
            "component_link": component_link
        }
        component_data.update(specs)
        return component_data
    finally:
        driver.quit()

def flatten_component_data(component_data):
    row = {}
    for key, value in component_data.items():
        if isinstance(value, list):
            row[key] = LIST_SEPARATOR.join(value)
        else:
            row[key] = value
    return row

def is_row_incomplete(row, headers):
    if "component_link" not in row or pd.isna(row["component_link"]):
        return False
    non_empty_columns = [col for col in headers if col != "component_link" and not pd.isna(row.get(col)) and str(row.get(col)).strip()]
    return len(non_empty_columns) <= 1

def update_processed_file(links_to_remove):
    if not os.path.exists(PROCESSED_FILE):
        logging.info(f"{PROCESSED_FILE} does not exist, skipping update...")
        return
    df_processed = pd.read_csv(PROCESSED_FILE, quoting=csv.QUOTE_ALL, on_bad_lines='warn')
    if 'component_link' not in df_processed.columns:
        logging.info(f"No 'component_link' column in {PROCESSED_FILE}, skipping...")
        return
    initial_count = len(df_processed)
    df_processed = df_processed[~df_processed['component_link'].isin(links_to_remove)]
    if len(df_processed) < initial_count:
        df_processed.to_csv(PROCESSED_FILE, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")
        logging.info(f"Removed {initial_count - len(df_processed)} links from {PROCESSED_FILE}")
    else:
        logging.info(f"No links removed from {PROCESSED_FILE}")

def main():
    for csv_file in CSV_FILES:
        csv_file="cleaned_"+csv_file
        if not os.path.exists(csv_file):
            logging.info(f"{csv_file} does not exist, skipping...")
            continue
        logging.info(f"Processing {csv_file}")
        try:
            df = pd.read_csv(csv_file, quoting=csv.QUOTE_ALL, on_bad_lines='warn', encoding="utf-8-sig")
        except pd.errors.ParserError as e:
            logging.error(f"Failed to parse {csv_file}: {e}")
            continue
        headers = df.columns.tolist()
        if "component_link" not in headers:
            logging.info(f"No 'component_link' column in {csv_file}, skipping...")
            continue
        incomplete_rows = []
        links_to_remove = set()
        for idx, row in df.iterrows():
            if is_row_incomplete(row, headers):
                incomplete_rows.append(idx)
                links_to_remove.add(row["component_link"])
        if not incomplete_rows:
            logging.info(f"No incomplete rows found in {csv_file}")
            continue
        logging.info(f"Found {len(incomplete_rows)} incomplete rows in {csv_file}")
        df = df.drop(incomplete_rows)
        df.to_csv(csv_file, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig")
        logging.info(f"Removed {len(incomplete_rows)} incomplete rows from {csv_file}")
        update_processed_file(links_to_remove)

if __name__ == "__main__":
    main()