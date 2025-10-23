# verify_order_placement.py
import asyncio
import config
import logging
from cerebellum_link import CerebellumLink

# --- Setup Logger ---
logger = logging.getLogger("OrderVerifier")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --- Global variable to catch the fill report ---
fill_report_received = None

def test_fill_callback(report_data):
    """A simple callback to store the received fill report."""
    global fill_report_received
    # --- FIX: Ignore heartbeat messages ---
    if 'Heartbeat' in report_data:
        logger.info(f"Heartbeat received from Cerebellum: {report_data['Heartbeat']['status']}")
        return # Ignore heartbeats and wait for a real fill report
    # --- END FIX ---
    logger.info(f"SUCCESS: Fill report received from Cerebellum -> {report_data}")
    fill_report_received = report_data

async def main():
    """Initializes the link and sends a single test order."""
    logger.info("--- Starting Order Placement Verification Script ---")
    
    # 1. Initialize Cerebellum Link with our test callback
    cerebellum_link = CerebellumLink(config, fill_callback=test_fill_callback)
    await asyncio.sleep(2) # Give ZMQ time to establish connections

    logger.info("CerebellumLink initialized.")

    # 2. Define the test order parameters
    # IMPORTANT: This uses the Bybit TESTNET as configured in your project.
    # We use a small market order to ensure it fills and we can verify the full round trip.
    test_order = {
        "symbol": "BTCUSDT",
        "side": "Buy",
        "quantity": 0.001,
        "strategy_id": "VerificationScript",
        "iel_mode": "Aggressive" # Use a simple mode for the test
    }

    logger.info(f"Sending test order to Cerebellum: {test_order}")

    # 3. Execute the order
    try:
        cerebellum_link.execute_order(**test_order)
        logger.info("Order sent to Cerebellum successfully.")
    except Exception as e:
        logger.error(f"FAILED to send order to Cerebellum: {e}", exc_info=True)
        return

    # 4. Wait for the fill report
    logger.info("Waiting for a fill report from the exchange (max 30 seconds)...")
    for i in range(30):
        if fill_report_received:
            break
        await asyncio.sleep(1)

    # 5. Report the result
    logger.info("--- Verification Complete ---")
    if fill_report_received:
        logger.info("✅ PASSED: The full order execution pipeline is working correctly.")
        logger.info(f"Final Report: {fill_report_received}")
    else:
        logger.error("❌ FAILED: No fill report was received from Cerebellum.")
        logger.error("This could mean an issue with the Rust core, the exchange API keys, or the connection.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Verification script stopped by user.")
