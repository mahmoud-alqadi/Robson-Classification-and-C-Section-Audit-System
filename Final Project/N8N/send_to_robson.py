"""
Robson Monthly Report — Hospital Demo Script
=============================================
Usage:  python send_to_robson.py sample_hospital_data.csv
Output: Opens the Robson classification report in your default browser.

This script:
  1. Reads a CSV file with birth records
  2. Sends them to the n8n Robson Automated Reporting System
  3. Saves the HTML report locally
  4. Opens it in your browser
"""

import csv
import json
import urllib.request
import webbrowser
import sys
import os
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────
WEBHOOK_URL = "https://mahmoudalqadi777666.app.n8n.cloud/webhook/robson-upload"
OUTPUT_FILE = "robson_report_{}.html".format(datetime.now().strftime("%Y%m%d_%H%M%S"))

def read_csv(filepath):
    """Read birth records from CSV, handling missing values."""
    records = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            for key, value in row.items():
                key = key.strip()
                value = value.strip() if value else ''
                # Convert numeric fields
                if key in ('parity', 'previous_cs', 'number_of_fetuses'):
                    record[key] = int(value) if value else None
                elif key == 'gestational_age':
                    record[key] = float(value) if value else None
                else:
                    record[key] = value if value else None
            records.append(record)
    return records

def send_to_webhook(records):
    """POST records to the n8n webhook and return the HTML response."""
    payload = json.dumps(records, ensure_ascii=False).encode('utf-8')

    req = urllib.request.Request(
        WEBHOOK_URL,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode('utf-8')

def main():
    # Determine input file
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "sample_hospital_data.csv"

    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        sys.exit(1)

    # Step 1: Read CSV
    print(f"Reading birth records from: {csv_path}")
    records = read_csv(csv_path)
    print(f"  → {len(records)} records loaded")

    # Step 2: Send to webhook
    print(f"Sending to Robson Automated Reporting System...")
    html = send_to_webhook(records)
    print(f"  → Report received ({len(html)} bytes)")

    # Step 3: Save report
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  → Saved to: {OUTPUT_FILE}")

    # Step 4: Open in browser
    filepath = os.path.abspath(OUTPUT_FILE)
    webbrowser.open(f'file:///{filepath}')
    print(f"  → Opened in browser")
    print(f"\nDone!")

if __name__ == '__main__':
    main()
