"""
Cerebellum Speed Test
Measures the round-trip IPC latency between Python and the Rust Cerebellum core.
"""
import subprocess
import time
import logging
import sys
from collections import deque

# Add project root to path to allow importing project modules
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from cerebellum_link import CerebellumLink
from config import ASSET_UNIVERSE
from forge.data_processing.l2_collector import L2Collector

# --- Configuration ---
TEST_COMMANDS = 100
CEREBELLUM_EXE_PATH = "cerebellum_core/target/release/cerebellum_core.exe"
# ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

latencies = deque()
reports_received = 0

def report_handler(report_data):
    """Callback function to process reports from Cerebellum."""
    global reports_received
    try:
        if "OrderUpdate" in report_data:
            update = report_data["OrderUpdate"]
            if "timestamp_sent" in update:
                timestamp_sent = float(update["timestamp_sent"])
                timestamp_received = time.time()
                latency_ms = (timestamp_received - timestamp_sent) * 1000
                latencies.append(latency_ms)
                reports_received += 1
                logging.info(f"Received report #{reports_received}. Latency: {latency_ms:.2f} ms")
    except Exception as e:
        logging.error(f"Error in report handler: {e}")

def run_speed_test():
    """Starts Cerebellum, runs the test, and prints results."""
    cerebellum_process = None
    l2_collector = None
    try:
        # 1. Start Cerebellum Core
        logging.info(f"Starting Cerebellum process from: {CEREBELLUM_EXE_PATH}")
        if not os.path.exists(CEREBELLUM_EXE_PATH):
            logging.error("Cerebellum executable not found. Please compile the Rust code first (run 1_Compile_Rust.bat).")
            return
            
        cerebellum_process = subprocess.Popen([CEREBELLUM_EXE_PATH])
        time.sleep(3) # Allow time for Cerebellum to initialize

        # 2. Initialize CerebellumLink with our callback
        class DummyConfig:
            """A simple config object for the speed test."""
            ASSET_UNIVERSE = ASSET_UNIVERSE
        link = CerebellumLink(app_config=DummyConfig(), fill_callback=report_handler)
        logging.info("CerebellumLink initialized.")

        # 3. Start L2 Collector and link it to Cerebellum
        logging.info("Starting L2 Collector...")
        assets_to_collect = [s.replace('/', '') for s in ASSET_UNIVERSE]
        l2_collector = L2Collector(assets=assets_to_collect, cerebellum_link=link)
        l2_collector.start()
        
        logging.info("Waiting 10 seconds for L2 collector to stabilize and receive data...")
        time.sleep(10)
        logging.info("Wait complete. Proceeding with test.")

        # 4. Send Test Commands
        logging.info(f"Sending {TEST_COMMANDS} test commands...")
        test_symbol = ASSET_UNIVERSE[0].replace('/', '').split(':')[0]
        for i in range(TEST_COMMANDS):
            link.execute_order(
                symbol=test_symbol,
                side="Buy",
                quantity=0.001,
                strategy_id=f"speed_test_{i}",
                iel_mode="Aggressive" # Use the correct variant
            )
            time.sleep(0.01) # Send commands 10ms apart

        # 5. Wait for all reports to be received
        logging.info("All commands sent. Waiting for reports...")
        timeout = 10 # 10 second timeout
        start_wait = time.time()
        while reports_received < TEST_COMMANDS and (time.time() - start_wait) < timeout:
            time.sleep(0.1)

        if reports_received < TEST_COMMANDS:
            logging.warning(f"Test timed out. Received {reports_received}/{TEST_COMMANDS} reports.")

    finally:
        # 6. Shut Down
        if l2_collector:
            logging.info("Stopping L2 Collector.")
            l2_collector.stop()
        if cerebellum_process:
            logging.info("Terminating Cerebellum process.")
            cerebellum_process.terminate()
            cerebellum_process.wait()

    # 7. Print Results
    if latencies:
        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_latency = sum(latencies) / len(latencies)
        
        print("\n--- Cerebellum Speed Test Results ---")
        print(f"Reports Received: {len(latencies)} / {TEST_COMMANDS}")
        print(f"Average Latency:  {avg_latency:.2f} ms")
        print(f"Minimum Latency:  {min_latency:.2f} ms")
        print(f"Maximum Latency:  {max_latency:.2f} ms")
        print("-------------------------------------\n")
    else:
        print("\n--- No reports received. Could not calculate speed. ---")
        print("Please ensure you have made the required change in the Rust code to echo back the 'timestamp_sent' field.")

if __name__ == "__main__":
    run_speed_test()
