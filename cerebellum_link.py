# In cerebellum_link.py
import zmq
import json
import logging
import threading
import time

class CerebellumLink:
    # Add app_config to the constructor
    def __init__(self, app_config, command_endpoint="tcp://127.0.0.1:5555", report_endpoint="tcp://127.0.0.1:5556", fill_callback=None):
        self.context = zmq.Context()
        self.app_config = app_config
        self.fill_callback = fill_callback
        
        # Command Bus (Python PUSH -> Rust PULL)
        self.command_socket = self.context.socket(zmq.PUSH)
        self.command_socket.setsockopt(zmq.LINGER, 0) # Optimize for low latency
        self.command_socket.connect(command_endpoint)
        
        # Report Bus (Python PULL <- Rust PUSH)
        self.report_socket = self.context.socket(zmq.PULL)
        self.report_socket.bind(report_endpoint)

        # Start listener thread
        self.listener_thread = threading.Thread(target=self._listen_for_reports, daemon=True)
        self.listener_thread.start()
        logging.info(f"CerebellumLink initialized. Waiting for connection stability...")

        # Allow ZMQ connections to stabilize
        time.sleep(1) 
        
        # Send Initialization command
        self.initialize_cerebellum()

    def initialize_cerebellum(self):
        """Sends the list of assets to the Rust Cerebellum."""
        assets = set()
        # Assuming ASSET_UNIVERSE is defined in app_config (from config.py)
        try:
            for asset in self.app_config.ASSET_UNIVERSE:
                # Sanitize the symbol for the exchange API format (e.g., "BTC/USDT:USDT" -> "BTCUSDT")
                base_symbol = asset.split(':')[0].replace('/', '')
                assets.add(base_symbol)
        except Exception as e:
            logging.error(f"Configuration error accessing ASSET_UNIVERSE: {e}")
            return

        if not assets:
            logging.error("Asset universe is empty. Cannot initialize Cerebellum.")
            return

        command = {
            "Initialize": {
                # JSON requires a list (array), Rust deserializes it into a HashSet
                "assets": list(assets) 
            }
        }
        logging.info(f"Sending Initialize command to Cerebellum with {len(assets)} assets.")
        self._send_command(command)

    def execute_order(self, symbol, side, quantity, strategy_id, iel_mode="AggressiveLimit"):
        # Ensure the symbol passed here is the sanitized version (e.g., "BTCUSDT")
        # Ensure capitalization for Bybit API ("Buy" or "Sell")
        command = {
            "ExecuteOrder": {
                "symbol": symbol,
                "side": side.capitalize(), 
                "quantity": quantity,
                "strategy_id": strategy_id,
                "iel_mode": iel_mode,
                "timestamp_sent": time.time() # Add high-precision timestamp
            }
        }
        self._send_command(command)

    def send_l2_snapshot(self, snapshot_data):
        """Sends a full L2 order book snapshot to the Rust Cerebellum."""
        command = {
            "L2Update": snapshot_data
        }
        self._send_command(command)

    def _send_command(self, command_dict):
        try:
            message = json.dumps(command_dict)
            self.command_socket.send_string(message, flags=zmq.NOBLOCK)
        except zmq.Again:
            logging.error("[Cortex IPC] Cerebellum command buffer full. Command dropped.")
        except Exception as e:
            logging.error(f"[Cortex IPC] Failed to send command: {e}")

    def _listen_for_reports(self):
        """Continuously listens for execution reports from Rust."""
        while True:
            try:
                message = self.report_socket.recv_string()
                report = json.loads(message)
                logging.info(f"[Cerebellum Report] {report}")
                
                if self.fill_callback:
                    self.fill_callback(report)

            except Exception as e:
                logging.error(f"[Cortex IPC] Error receiving report: {e}")