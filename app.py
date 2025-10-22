# app.py (V5.12 - Automated Startup)

# CRITICAL FIX: Must be at the very top
import os
import warnings
import multiprocessing # Import multiprocessing

# Force OpenMP/MKL/OpenBLAS libraries globally to use a single thread 
# to prevent deadlocks when multiprocessing is active.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# CRITICAL FIX: Force the 'spawn' start method for stability
# 'spawn' starts a fresh interpreter, avoiding deadlocks caused by inheriting locks/states when using 'fork'.
try:
    # Check if the start method has already been set
    current_method = multiprocessing.get_start_method(allow_none=True)
    if current_method != 'spawn':
        print(f"[Multiprocessing] Start method is '{current_method}'. Forcing to 'spawn'.")
        # Use force=True to ensure it's set, even if previously set incorrectly
        multiprocessing.set_start_method('spawn', force=True)
    else:
        print(f"[Multiprocessing] Start method confirmed as 'spawn'.")

except RuntimeError as e:
    # This error occurs if set_start_method is called too late (after processes have already been created or modules imported).
    print(f"[Multiprocessing] CRITICAL WARNING: Could not enforce 'spawn' start method: {e}. Ensure this is at the absolute top of app.py.")


# --- Suppress pkg_resources deprecation warning ---
warnings.filterwarnings("ignore", category=UserWarning, module='pandas_ta')
# --- Suppress the Gym deprecation warning from stable-baselines3 ---
warnings.filterwarnings("ignore", category=UserWarning, message=".*Gym has been unmaintained.*")

# --- Centralized Imports ---
import json
import asyncio
import threading
import subprocess
import time
import sys
import webbrowser
# Import config and rename it for clarity
import config as app_config 
from crucible_engine import CrucibleEngine
import logging
from flask import Flask, render_template, jsonify, request
import pandas as pd # Ensure pandas is imported globally for the routes

# --- Silence the default Flask web server logger ---
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def open_browser_after_delay():
    """Waits for a few seconds and then opens the web browser to the UI."""
    time.sleep(5)
    webbrowser.open("http://127.0.0.1:5001/")

def run_weekly_data_refresh():
    """Runs the data preparation script every 7 days, waiting a week before the first run."""
    while True:
        time.sleep(7 * 24 * 60 * 60)
        print("[Data Refresh] Starting weekly data refresh cycle...")
        try:
            subprocess.run([sys.executable, "prepare_all_data.py"], check=True)
            print("[Data Refresh] Weekly data refresh completed successfully.")
        except Exception as e:
            print(f"[Data Refresh] ERROR during weekly data refresh: {e}")

class ServiceManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, app_config):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self.config = app_config
        self.crucible_engine = None
        self.main_loop_thread = None
        self.stop_event = asyncio.Event()
        self.logger = logging.getLogger("ServiceManager")

    def _are_models_missing(self):
        """Checks if all essential model files for the first asset exist as a proxy for all assets."""
        model_dir = self.config.MODEL_DIR
        if not os.path.exists(model_dir):
            print("[Model Check] Model directory is missing.")
            return True

        first_asset_id = self.config.ASSET_UNIVERSE[0].split(':')[0].replace('/', '')
        
        # Correctly check for the directory-based model structure
        specialist_model_exists = False
        for item_name in os.listdir(model_dir):
            if item_name.startswith(first_asset_id) and os.path.isdir(os.path.join(model_dir, item_name)):
                specialist_model_exists = True
                break

        required_models = {
            "Specialist": specialist_model_exists,
            "RL Governor": os.path.exists(os.path.join(model_dir, f"{first_asset_id}_RLG.joblib")),
            "HDBSCAN": os.path.exists(os.path.join(model_dir, f"{first_asset_id}_HDBSCAN.joblib"))
        }
        
        for model_type, model_exists in required_models.items():
            if not model_exists:
                print(f"[Model Check] ❌ Missing essential model type '{model_type}'.")
                return True
                
        print("[Model Check] ✅ All essential models appear to be present.")
        return False
    
    def _ensure_log_files_exist(self):
        """Creates essential log files if they don't exist to prevent startup errors."""
        log_files = ["singularity_log.txt", "decisions_log.txt"]
        for log_file in log_files:
            if not os.path.exists(log_file):
                print(f"[ServiceManager] Log file '{log_file}' not found. Creating empty file.")
                with open(log_file, 'w') as f:
                    pass # Create an empty file
    
    def _initialize_services(self):
        print("[App.py] ServiceManager: Initializing services...")
        self._ensure_log_files_exist() # Ensure logs exist before anything else
        
        # --- CRITICAL FIX: Instantiate engine first ---
        self.crucible_engine = CrucibleEngine(app_config=self.config, is_main_instance=True)

        if self._are_models_missing():
            print("\n" + "="*60)
            print("="*15, "  MISSING MODELS - MANUAL START REQUIRED  ", "="*15)
            print("="*60)
            print("Models are missing. Please run '3_Start_Initial_Script.bat' to train them.")
            print("The application will not function correctly until all models are trained.")
            print("="*60 + "\n")
        
        # Start the main loop regardless of model status to prevent API crashes
        print("[App.py] Starting main loop thread...")
        self.start_loops()

        data_refresh_thread = threading.Thread(target=run_weekly_data_refresh, daemon=True)
        data_refresh_thread.start()

    def get_all_service_statuses(self):
        is_running = self.main_loop_thread is not None and self.main_loop_thread.is_alive()
        active_bots_list = []
        if self.crucible_engine:
            arena_statuses = self.crucible_engine.arena.get_all_statuses()
            for symbol, agent_list in arena_statuses.items():
                for agent_status in agent_list:
                    agent_status['symbol'] = symbol
                    active_bots_list.append(agent_status)
        return {
            "trader": { "is_running": is_running, "live_data": {"active_bots": active_bots_list} },
            "singularity": { "is_running": is_running }
        }

    async def _run_main_loops_async(self):
        self.logger.info("ServiceManager: Starting main async loops...")
        try:
            # Run CrucibleEngine asynchronously
            await self.crucible_engine.run()
        except Exception as e:
            self.logger.error(f"Critical error in main loops: {e}", exc_info=True)

    def start_loops(self):
        # Start the async loop in a new thread
        # The lambda ensures asyncio.run executes the async entry point in the new thread's event loop
        self.main_loop_thread = threading.Thread(
            target=lambda: asyncio.run(self._run_main_loops_async()), 
            daemon=True
        )
        self.main_loop_thread.start()
        self.logger.info("ServiceManager: Main loop thread started.")

    def stop_main_loop(self):
        """Stops the Crucible Engine's main async loop."""
        if not self.main_loop_thread or not self.main_loop_thread.is_alive() or not self.crucible_engine:
            print("[ServiceManager] Loop thread is not running or engine not initialized.")
            return
        
        print("[ServiceManager] Stopping main loop thread...")
        # Use the stored loop to schedule the stop coroutine
        if hasattr(self.crucible_engine, 'loop') and self.crucible_engine.loop.is_running():
            # Request the async stop() method to be run in the loop
            future = asyncio.run_coroutine_threadsafe(self.crucible_engine.stop(), self.crucible_engine.loop)
            try:
                future.result(timeout=10) # Wait for the stop coroutine to finish
            except TimeoutError:
                print("[ServiceManager] Timeout waiting for crucible_engine.stop() to complete.")
        
        self.main_loop_thread.join(timeout=10)
        if self.main_loop_thread.is_alive():
            print("[ServiceManager] WARNING: Loop thread did not terminate gracefully.")
        else:
            print("[ServiceManager] Main loop thread stopped.")
        self.main_loop_thread = None

def setup_routes(app, service_manager):
    @app.route('/api/performance/summary', methods=['GET'])
    def get_performance_summary():
        default_response = {"champion": None, "contenders": None, "most_adaptable": None}
        try:
            with open("performance_log.jsonl", "r") as f:
                lines = f.readlines()[-2000:]
                # Parse all lines and filter for EXIT actions (completed trades)
                all_logs = []
                for line in lines:
                    try:
                        log = json.loads(line)
                        all_logs.append(log)
                    except json.JSONDecodeError:
                        continue
                
                # Filter for EXIT actions which contain complete trade data
                trade_logs = [log for log in all_logs if log.get('action') == 'EXIT']
        except FileNotFoundError:
            return jsonify(default_response)
        
        if not trade_logs:
            return jsonify(default_response)
        
        df = pd.DataFrame(trade_logs)
        
        # Use 'net_pnl' column from the agent logs
        if 'net_pnl' not in df.columns:
            return jsonify(default_response)
        
        df['net_pnl'] = pd.to_numeric(df['net_pnl'], errors='coerce')
        df.dropna(subset=['net_pnl'], inplace=True)

        if df.empty:
            return jsonify(default_response)

        # Calculate performance by model
        model_performance = df.groupby('model_id')['net_pnl'].sum().sort_values(ascending=False)
        
        if model_performance.empty:
            return jsonify(default_response)
        
        champion_id = model_performance.index[0]
        champion_trades = df[df['model_id'] == champion_id]
        
        champion_stats = {
            "model_id": champion_id,
            "total_pnl": float(champion_trades['net_pnl'].sum()),
            "win_rate": float((champion_trades['net_pnl'] > 0).mean() * 100) if not champion_trades.empty else 0,
            "total_trades": int(len(champion_trades))
        }
        
        contenders_stats = {
            "average_pnl_per_trade": float(df['net_pnl'].mean()),
            "system_win_rate": float((df['net_pnl'] > 0).mean() * 100),
            "total_trades": int(len(df))
        }
        
        return jsonify({"champion": champion_stats, "contenders": contenders_stats, "most_adaptable": None})

    @app.route('/')
    def index(): return render_template('dashboard.html')
    @app.route('/dashboard')
    def dashboard(): return render_template('dashboard.html')

    @app.route('/api/system/status_all', methods=['GET'])
    def get_system_status():
        statuses = service_manager.get_all_service_statuses()
        try:
            with open("singularity_log.txt", "r") as f:
                statuses['singularity']['logs'] = f.readlines()[-20:]
        except FileNotFoundError:
            statuses['singularity']['logs'] = ["Log file not found."]
        return jsonify(statuses)

    @app.route('/api/decisions_log', methods=['GET'])
    def get_decisions_log():
        try:
            with open("decisions_log.txt", "r") as f:
                lines = f.readlines()
                return jsonify({"logs": lines[-20:]})
        except FileNotFoundError:
            return jsonify({"logs": ["Decisions log file not found."]})

    @app.route('/api/loop/start', methods=['POST'])
    def start_main_loop():
        service_manager.start_loops()
        return jsonify({"status": "started"})

    @app.route('/api/loop/stop', methods=['POST'])
    def stop_main_loop():
        service_manager.stop_main_loop()
        return jsonify({"status": "stopped"})

    @app.route('/api/prices/live', methods=['GET'])
    def get_live_prices():
        """Fetches current live prices for all active trading symbols."""
        prices = {}
        try:
            if service_manager.crucible_engine and service_manager.crucible_engine.exchange:
                exchange = service_manager.crucible_engine.exchange
                symbols = service_manager.crucible_engine.config.ASSET_UNIVERSE

                for symbol in symbols:
                    try:
                        # Run async fetch_ticker in the crucible event loop
                        if service_manager.crucible_engine.loop and service_manager.crucible_engine.loop.is_running():
                            future = asyncio.run_coroutine_threadsafe(
                                exchange.fetch_ticker(symbol),
                                service_manager.crucible_engine.loop
                            )
                            ticker = future.result(timeout=2)  # 2 second timeout
                            prices[symbol] = {
                                'price': ticker.get('last', 0),
                                'bid': ticker.get('bid', 0),
                                'ask': ticker.get('ask', 0),
                                'volume': ticker.get('baseVolume', 0),
                                'change_24h': ticker.get('percentage', 0)
                            }
                        else:
                            prices[symbol] = {'price': 0}
                    except Exception as e:
                        print(f"Error fetching price for {symbol}: {e}")
                        prices[symbol] = {'price': 0}
        except Exception as e:
            print(f"Error in get_live_prices: {e}")

        return jsonify({"prices": prices})

    @app.route('/api/models/list', methods=['GET'])
    def get_models_list():
        # ... (existing code to scan models_dir) ...
        models_list = []
        models_dir = app_config.MODEL_DIR
        if not os.path.exists(models_dir):
            return jsonify({"models": [], "error": "Models directory not found."})

        for item_name in os.listdir(models_dir):
            item_path = os.path.join(models_dir, item_name)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, 'metadata.json')
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)

                        # --- ADD FILTERING LOGIC ---
                        # Check if the model type is GP 2.0 (adjust key if needed)
                        model_type = metadata.get("blueprint", {}).get("model_type") # Check metadata structure
                        is_gp_model = model_type == "GP 2.0" # Adjust if type name is different

                        if is_gp_model: # Only add GP 2.0 models
                            # --- END FILTERING LOGIC ---
                            models_list.append({
                                'model_id': metadata.get('model_id', item_name),
                                'asset_symbol': metadata.get('asset_symbol', 'N/A'),
                                'status': metadata.get('status', 'purgatory'),
                                # Adjust key based on actual metadata if needed
                                'fitness_score': metadata.get('validation_metrics', {}).get('wfa_metrics', {}).get('sharpe_ratio', 'N/A'),
                                'registration_time': metadata.get('registration_time', 'N/A'),
                                # Add architecture if needed by JS updateModels
                                'architecture': model_type
                            })
                    except Exception as e:
                        print(f"[API] Error reading metadata for {item_name}: {e}")

        # Optional: Sort models by registration time descending
        models_list.sort(key=lambda x: x.get('registration_time', '0'), reverse=True)

        return jsonify({"models": models_list})

    @app.route('/api/models/activate', methods=['POST'])
    async def activate_model():
        """Activates a model by loading it into the Crucible Engine."""
        data = request.get_json()
        model_id = data.get('model_id')
        if not model_id:
            return jsonify({"status": "error", "message": "model_id not provided"}), 400

        print(f"[API] Activating model: {model_id}")

        if service_manager.crucible_engine:
            # Schedule the async load_agent method to run on the engine's event loop
            future = asyncio.run_coroutine_threadsafe(
                service_manager.crucible_engine.load_agent(model_id),
                service_manager.crucible_engine.loop
            )
            # Wait for the result (optional, but good for confirming it was called)
            try:
                future.result(timeout=10)
            except Exception as e:
                print(f"[API] Error calling load_agent: {e}")
                return jsonify({"status": "error", "message": f"Failed to schedule agent loading: {e}"}), 500

            return jsonify({"status": "success", "message": f"Activation request for {model_id} sent."})
        else:
            return jsonify({"status": "error", "message": "Crucible Engine not available."}), 503

    # === BACKWARD COMPATIBILITY ENDPOINTS ===
    # These endpoints provide compatibility with dashboard.js expectations

    @app.route('/api/master_status', methods=['GET'])
    def get_master_status():
        """Returns master status with dashboard data for active bots."""
        statuses = service_manager.get_all_service_statuses()

        # Build dashboard data structure expected by UI
        dashboard_data = {}
        if statuses.get('trader', {}).get('live_data', {}).get('active_bots'):
            for bot in statuses['trader']['live_data']['active_bots']:
                symbol = bot.get('symbol')
                if symbol:
                    dashboard_data[symbol] = bot

        return jsonify({
            "trader": statuses.get('trader', {"is_running": False}),
            "singularity": statuses.get('singularity', {"is_running": False}),
            "dashboard": dashboard_data
        })

    @app.route('/api/prices', methods=['GET'])
    def get_prices():
        """Proxy endpoint for /api/prices/live - returns simplified price data."""
        try:
            if service_manager.crucible_engine and service_manager.crucible_engine.exchange:
                exchange = service_manager.crucible_engine.exchange
                symbols = service_manager.crucible_engine.config.ASSET_UNIVERSE
                prices = {}

                for symbol in symbols:
                    try:
                        if service_manager.crucible_engine.loop and service_manager.crucible_engine.loop.is_running():
                            future = asyncio.run_coroutine_threadsafe(
                                exchange.fetch_ticker(symbol),
                                service_manager.crucible_engine.loop
                            )
                            ticker = future.result(timeout=2)
                            # Simplified format: just the price as a float
                            prices[symbol] = ticker.get('last', 0)
                        else:
                            prices[symbol] = 0
                    except Exception as e:
                        print(f"Error fetching price for {symbol}: {e}")
                        prices[symbol] = 0
        except Exception as e:
            print(f"Error in get_prices: {e}")
            prices = {}

        return jsonify(prices)



    @app.route('/api/start_loop', methods=['POST'])
    def start_loop_compat():
        """Compatibility endpoint for /api/loop/start"""
        return start_main_loop()

    @app.route('/api/stop_loop', methods=['POST'])
    def stop_loop_compat():
        """Compatibility endpoint for /api/loop/stop"""
        return stop_main_loop()


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    print("[App.py] Initializing ServiceManager (Main Process)...")
    service_manager = ServiceManager(app_config=app_config)
    print("[App.py] ServiceManager initialized.")
    
    app = Flask(__name__)
    setup_routes(app, service_manager)

    service_manager._initialize_services()

    print("[App.py] Main execution block started.")
    try:
        # Run Flask in a separate thread
        flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5001, debug=False), daemon=True)
        flask_thread.start()

        browser_thread = threading.Thread(target=open_browser_after_delay, daemon=True)
        browser_thread.start()
        
        # Keep the main thread alive to allow daemon threads to run
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[App.py] Keyboard interrupt received. Shutting down...")
    except Exception as e:
        print(f"\n\n--- UNCAUGHT EXCEPTION IN MAIN BLOCK ---\n{e}\n")
    finally:
        print("[App.py] Main execution block finished.")
        if 'service_manager' in locals() and service_manager:
           service_manager.stop_main_loop()
