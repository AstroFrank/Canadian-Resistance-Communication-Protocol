"""
Quantum-Resilient C2 System - Monitoring Dashboard

This module implements a comprehensive web-based monitoring dashboard for the 
quantum-resilient command and control system. The dashboard provides real-time 
visualization of system components, including blockchain status, radio transmissions, 
and torrent activity.

The implementation utilizes Flask for the backend API, Flask-SocketIO for real-time 
updates, and a minimal but effective frontend with Bootstrap for responsiveness.
"""

import os
import sys
import time
import json
import logging
import datetime
import threading
import schedule
from functools import wraps
import hashlib
import requests
from flask import Flask, render_template, jsonify, request, redirect, url_for, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Dashboard")

# Flask application setup
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24).hex())
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
DEFAULT_CONFIG = {
    "blockchain_api": "http://localhost:5000",
    "refresh_interval": 30,  # seconds
    "admin_username": "admin",
    "admin_password_hash": generate_password_hash("changeme"),  # Default password, should be changed
    "transmitter_log_path": "/var/log/hf_transmitter.log",
    "receiver_log_path": "/var/log/rtl_receiver.log",
    "torrent_status_command": "transmission-remote -l",
    "max_log_entries": 1000,
    "alert_thresholds": {
        "blockchain_sync_delay": 3600,  # seconds
        "transmission_missed": True,
        "low_receiver_snr": -25,  # dB
        "torrent_stalled": True
    }
}

# Load configuration
def load_config():
    """Load dashboard configuration from file or environment."""
    config_path = os.environ.get('DASHBOARD_CONFIG', 'dashboard_config.json')
    config = DEFAULT_CONFIG.copy()
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config)
                logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        
    # Override with environment variables if present
    if os.environ.get('BLOCKCHAIN_API'):
        config['blockchain_api'] = os.environ.get('BLOCKCHAIN_API')
    if os.environ.get('ADMIN_USERNAME'):
        config['admin_username'] = os.environ.get('ADMIN_USERNAME')
    if os.environ.get('ADMIN_PASSWORD'):
        config['admin_password_hash'] = generate_password_hash(os.environ.get('ADMIN_PASSWORD'))
        
    return config

CONFIG = load_config()

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_credentials(auth.username, auth.password):
            return Response(
                'Authentication required',
                401,
                {'WWW-Authenticate': 'Basic realm="Login Required"'}
            )
        return f(*args, **kwargs)
    return decorated_function

def check_credentials(username, password):
    """Check if the provided credentials are valid."""
    return (username == CONFIG['admin_username'] and 
            check_password_hash(CONFIG['admin_password_hash'], password))

# Data models
class SystemStatus:
    """Class to track and manage overall system status."""
    
    def __init__(self):
        self.blockchain_status = {
            "last_update": None,
            "latest_block": None,
            "chain_valid": None,
            "chain_length": 0,
            "last_check": None
        }
        
        self.radio_status = {
            "last_transmission": None,
            "next_transmission": None,
            "transmission_success": None,
            "current_frequency": None,
            "last_check": None
        }
        
        self.receiver_status = {
            "last_reception": None,
            "signal_strength": None,
            "decoded_messages": [],
            "active_receivers": [],
            "last_check": None
        }
        
        self.torrent_status = {
            "active_torrents": [],
            "total_seeders": 0,
            "total_leechers": 0,
            "latest_command_file": None,
            "download_status": None,
            "last_check": None
        }
        
        self.alerts = []
        self.logs = {
            "blockchain": [],
            "radio": [],
            "receiver": [],
            "torrent": []
        }
        
    def add_alert(self, component, level, message, timestamp=None):
        """Add a new alert to the system."""
        if timestamp is None:
            timestamp = datetime.datetime.utcnow().isoformat()
            
        alert = {
            "component": component,
            "level": level,
            "message": message,
            "timestamp": timestamp,
            "acknowledged": False,
            "id": hashlib.md5(f"{timestamp}:{message}".encode()).hexdigest()
        }
        
        self.alerts.append(alert)
        
        # Keep only the most recent 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
            
        # Emit the alert to connected clients
        try:
            socketio.emit('new_alert', alert)
        except Exception as e:
            logger.error(f"Failed to emit alert: {e}")
            
        return alert
        
    def add_log(self, component, message, level="INFO", timestamp=None):
        """Add a log entry to the specified component."""
        if timestamp is None:
            timestamp = datetime.datetime.utcnow().isoformat()
            
        log_entry = {
            "message": message,
            "level": level,
            "timestamp": timestamp
        }
        
        # Add to the appropriate log
        if component in self.logs:
            self.logs[component].append(log_entry)
            
            # Limit the size of the log
            if len(self.logs[component]) > CONFIG["max_log_entries"]:
                self.logs[component] = self.logs[component][-CONFIG["max_log_entries"]:]
                
        # Emit the log to connected clients
        try:
            socketio.emit('new_log', {"component": component, "log": log_entry})
        except Exception as e:
            logger.error(f"Failed to emit log: {e}")
            
        return log_entry
        
    def check_for_issues(self):
        """Check for system issues and create alerts if needed."""
        now = datetime.datetime.utcnow()
        
        # Check blockchain sync
        if self.blockchain_status["last_update"]:
            last_update = datetime.datetime.fromisoformat(self.blockchain_status["last_update"])
            if (now - last_update).total_seconds() > CONFIG["alert_thresholds"]["blockchain_sync_delay"]:
                self.add_alert(
                    "blockchain", 
                    "warning", 
                    f"Blockchain not updated in {CONFIG['alert_thresholds']['blockchain_sync_delay']/3600:.1f} hours"
                )
                
        # Check for missed transmissions
        if (CONFIG["alert_thresholds"]["transmission_missed"] and 
            self.radio_status["last_transmission"] and 
            self.radio_status["next_transmission"]):
            
            last_tx = datetime.datetime.fromisoformat(self.radio_status["last_transmission"])
            next_tx = datetime.datetime.fromisoformat(self.radio_status["next_transmission"])
            
            # If the next transmission time has passed but last_transmission hasn't been updated
            if now > next_tx and last_tx < next_tx:
                self.add_alert(
                    "radio", 
                    "error", 
                    f"Missed scheduled transmission at {next_tx.isoformat()}"
                )
                
        # Check receiver signal strength
        if (self.receiver_status["signal_strength"] is not None and 
            self.receiver_status["signal_strength"] < CONFIG["alert_thresholds"]["low_receiver_snr"]):
            
            self.add_alert(
                "receiver", 
                "warning", 
                f"Low receiver signal strength: {self.receiver_status['signal_strength']} dB"
            )
            
        # Check for stalled torrents
        if CONFIG["alert_thresholds"]["torrent_stalled"]:
            for torrent in self.torrent_status["active_torrents"]:
                if torrent.get("status") == "Stalled" and torrent.get("is_command_file", False):
                    self.add_alert(
                        "torrent", 
                        "error", 
                        f"Command file torrent stalled: {torrent.get('name', 'Unknown')}"
                    )

# Initialize system status
system_status = SystemStatus()

# Blockchain monitoring
def fetch_blockchain_status():
    """Fetch and update blockchain status from the API."""
    try:
        # Fetch latest block
        response = requests.get(f"{CONFIG['blockchain_api']}/blocks/latest", timeout=5)
        if response.status_code == 200:
            latest_block = response.json()
            
            # Update status
            system_status.blockchain_status["latest_block"] = latest_block
            system_status.blockchain_status["last_update"] = latest_block.get("timestamp", datetime.datetime.utcnow().isoformat())
            
            # Log the update
            system_status.add_log(
                "blockchain", 
                f"Fetched latest block #{latest_block.get('index', 'unknown')}", 
                "INFO"
            )
            
            # Fetch chain length
            response = requests.get(f"{CONFIG['blockchain_api']}/blocks", timeout=5)
            if response.status_code == 200:
                blocks = response.json()
                system_status.blockchain_status["chain_length"] = len(blocks)
                
            # Check chain validity
            response = requests.get(f"{CONFIG['blockchain_api']}/validate", timeout=5)
            if response.status_code == 200:
                validation = response.json()
                system_status.blockchain_status["chain_valid"] = validation.get("valid", False)
                
                if not validation.get("valid", False):
                    system_status.add_alert(
                        "blockchain", 
                        "error", 
                        "Blockchain validation failed!"
                    )
                    
            system_status.blockchain_status["last_check"] = datetime.datetime.utcnow().isoformat()
            
            # Emit update to clients
            socketio.emit('blockchain_update', system_status.blockchain_status)
            
        else:
            logger.error(f"Failed to fetch blockchain status: {response.status_code}")
            system_status.add_log(
                "blockchain", 
                f"Failed to fetch blockchain status: {response.status_code}", 
                "ERROR"
            )
            
    except Exception as e:
        logger.error(f"Error fetching blockchain status: {e}")
        system_status.add_log(
            "blockchain", 
            f"Error fetching blockchain status: {e}", 
            "ERROR"
        )

# Radio monitoring
def parse_transmitter_log():
    """Parse the HF transmitter log file to update radio status."""
    try:
        log_path = CONFIG["transmitter_log_path"]
        if not os.path.exists(log_path):
            logger.warning(f"Transmitter log not found: {log_path}")
            return
            
        # Read the last N lines of the log file
        with open(log_path, 'r') as f:
            # Read the last 1000 lines at most
            lines = f.readlines()[-1000:]
            
        # Process logs to extract status information
        last_tx_time = None
        next_tx_time = None
        current_freq = None
        last_success = None
        
        for line in reversed(lines):  # Start from the end
            # Extract timestamp from log line
            try:
                timestamp_str = line.split(" - ")[0].strip()
                log_time = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            except (ValueError, IndexError):
                continue
                
            # Look for transmission info
            if "PTT ON" in line:
                last_tx_time = log_time.isoformat()
                
            if "Set frequency to" in line:
                try:
                    freq_str = line.split("Set frequency to ")[1].split(" MHz")[0]
                    current_freq = float(freq_str)
                except (IndexError, ValueError):
                    pass
                    
            if "Next transmission frequency:" in line:
                try:
                    # Extract the frequency value
                    freq_str = line.split("Next transmission frequency: ")[1].split(" MHz")[0]
                    
                    # If we already found a transmission time, this is the next scheduled one
                    if last_tx_time and not next_tx_time:
                        # Assume the next transmission is at the next even-numbered hour
                        # This is a simplification; in a real system, you'd have actual scheduling info
                        tx_time = log_time + datetime.timedelta(hours=2)
                        # Round to the nearest even hour
                        tx_time = tx_time.replace(minute=0, second=0, microsecond=0)
                        if tx_time.hour % 2 == 1:
                            tx_time = tx_time + datetime.timedelta(hours=1)
                            
                        next_tx_time = tx_time.isoformat()
                except (IndexError, ValueError):
                    pass
                    
            if "WSPR audio file created successfully" in line or "PTT OFF" in line:
                last_success = True
                
            if "Failed to" in line and "transmission" in line:
                last_success = False
                system_status.add_alert(
                    "radio", 
                    "error", 
                    f"Transmission failure: {line.strip()}"
                )
                
            # Once we've found all the info we need, break
            if last_tx_time and current_freq and next_tx_time is not None and last_success is not None:
                break
                
        # Update status
        if last_tx_time:
            system_status.radio_status["last_transmission"] = last_tx_time
        if next_tx_time:
            system_status.radio_status["next_transmission"] = next_tx_time
        if current_freq:
            system_status.radio_status["current_frequency"] = current_freq
        if last_success is not None:
            system_status.radio_status["transmission_success"] = last_success
            
        system_status.radio_status["last_check"] = datetime.datetime.utcnow().isoformat()
        
        # Emit update to clients
        socketio.emit('radio_update', system_status.radio_status)
        
    except Exception as e:
        logger.error(f"Error parsing transmitter log: {e}")
        system_status.add_log(
            "radio", 
            f"Error parsing transmitter log: {e}", 
            "ERROR"
        )

# Receiver monitoring
def parse_receiver_log():
    """Parse the RTL-SDR receiver log file to update receiver status."""
    try:
        log_path = CONFIG["receiver_log_path"]
        if not os.path.exists(log_path):
            logger.warning(f"Receiver log not found: {log_path}")
            return
            
        # Read the last N lines of the log file
        with open(log_path, 'r') as f:
            # Read the last 1000 lines at most
            lines = f.readlines()[-1000:]
            
        # Process logs to extract status information
        last_reception = None
        signal_strength = None
        decoded_messages = []
        active_receivers = set()
        
        for line in reversed(lines):  # Start from the end
            # Extract timestamp from log line
            try:
                timestamp_str = line.split(" - ")[0].strip()
                log_time = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            except (ValueError, IndexError):
                continue
                
            # Look for reception info
            if "Decoded" in line and "WSPR spots" in line:
                if not last_reception:
                    last_reception = log_time.isoformat()
                    
            if "snr" in line.lower() and ":" in line:
                try:
                    # Extract SNR value
                    snr_part = line.split("SNR")[1].split("dB")[0].strip()
                    snr_value = float(snr_part.strip(": "))
                    if signal_strength is None or snr_value > signal_strength:
                        signal_strength = snr_value
                except (IndexError, ValueError):
                    pass
                    
            if "Decoded command:" in line:
                try:
                    command = line.split("Decoded command:")[1].strip()
                    if len(decoded_messages) < 5:  # Keep last 5 messages
                        decoded_messages.append({
                            "timestamp": log_time.isoformat(),
                            "message": command
                        })
                except IndexError:
                    pass
                    
            # Identify active receiver nodes
            if "RTL-SDR initialized" in line:
                try:
                    # Extract receiver ID or name if available
                    node_id = f"Receiver-{len(active_receivers)+1}"  # Default ID
                    if "device=" in line:
                        device_part = line.split("device=")[1].split(",")[0]
                        node_id = f"Receiver-{device_part}"
                        
                    active_receivers.add(node_id)
                except (IndexError, ValueError):
                    pass
                    
        # Update status
        if last_reception:
            system_status.receiver_status["last_reception"] = last_reception
        if signal_strength is not None:
            system_status.receiver_status["signal_strength"] = signal_strength
        if decoded_messages:
            system_status.receiver_status["decoded_messages"] = decoded_messages + system_status.receiver_status["decoded_messages"]
            # Keep only the most recent 10 messages
            system_status.receiver_status["decoded_messages"] = system_status.receiver_status["decoded_messages"][:10]
        
        system_status.receiver_status["active_receivers"] = list(active_receivers)
        system_status.receiver_status["last_check"] = datetime.datetime.utcnow().isoformat()
        
        # Emit update to clients
        socketio.emit('receiver_update', system_status.receiver_status)
        
    except Exception as e:
        logger.error(f"Error parsing receiver log: {e}")
        system_status.add_log(
            "receiver", 
            f"Error parsing receiver log: {e}", 
            "ERROR"
        )

# Torrent monitoring
def fetch_torrent_status():
    """Fetch and update torrent status."""
    try:
        # Execute torrent client status command
        cmd = CONFIG["torrent_status_command"]
        result = os.popen(cmd).read()
        
        # Parse torrent status output
        torrents = []
        total_seeders = 0
        total_leechers = 0
        latest_command = None
        
        for line in result.splitlines():
            if "ID" in line and "Name" in line:
                continue  # Skip header
                
            parts = line.split()
            if len(parts) < 5:
                continue
                
            try:
                torrent_id = parts[0]
                status = parts[1]
                progress = parts[2]
                seeders = int(parts[3])
                leechers = int(parts[4])
                name = " ".join(parts[5:])
                
                # Detect if this is likely a command file
                is_command = "command" in name.lower() or "c2" in name.lower()
                
                torrent = {
                    "id": torrent_id,
                    "status": status,
                    "progress": progress,
                    "seeders": seeders,
                    "leechers": leechers,
                    "name": name,
                    "is_command_file": is_command
                }
                
                torrents.append(torrent)
                total_seeders += seeders
                total_leechers += leechers
                
                # Track the latest command file
                if is_command and (latest_command is None or torrent["id"] > latest_command["id"]):
                    latest_command = torrent
                    
            except (IndexError, ValueError):
                continue
                
        # Update status
        system_status.torrent_status["active_torrents"] = torrents
        system_status.torrent_status["total_seeders"] = total_seeders
        system_status.torrent_status["total_leechers"] = total_leechers
        system_status.torrent_status["latest_command_file"] = latest_command
        
        if latest_command:
            system_status.torrent_status["download_status"] = latest_command["progress"]
            
        system_status.torrent_status["last_check"] = datetime.datetime.utcnow().isoformat()
        
        # Emit update to clients
        socketio.emit('torrent_update', system_status.torrent_status)
        
    except Exception as e:
        logger.error(f"Error fetching torrent status: {e}")
        system_status.add_log(
            "torrent", 
            f"Error fetching torrent status: {e}", 
            "ERROR"
        )

# Scheduled monitoring tasks
def start_monitoring():
    """Start all monitoring tasks."""
    schedule.every(CONFIG["refresh_interval"]).seconds.do(fetch_blockchain_status)
    schedule.every(CONFIG["refresh_interval"]).seconds.do(parse_transmitter_log)
    schedule.every(CONFIG["refresh_interval"]).seconds.do(parse_receiver_log)
    schedule.every(CONFIG["refresh_interval"]).seconds.do(fetch_torrent_status)
    schedule.every(CONFIG["refresh_interval"]).seconds.do(system_status.check_for_issues)
    
    # Run immediately on startup
    fetch_blockchain_status()
    parse_transmitter_log()
    parse_receiver_log()
    fetch_torrent_status()
    
    # Start the scheduler in a background thread
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)
            
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    logger.info("Monitoring tasks started")

# Routes
@app.route('/')
@login_required
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/blockchain')
@login_required
def blockchain_page():
    """Render the blockchain status page."""
    return render_template('blockchain.html')

@app.route('/radio')
@login_required
def radio_page():
    """Render the radio status page."""
    return render_template('radio.html')

@app.route('/receiver')
@login_required
def receiver_page():
    """Render the receiver status page."""
    return render_template('receiver.html')

@app.route('/torrent')
@login_required
def torrent_page():
    """Render the torrent status page."""
    return render_template('torrent.html')

@app.route('/logs')
@login_required
def logs_page():
    """Render the logs page."""
    return render_template('logs.html')

@app.route('/settings')
@login_required
def settings_page():
    """Render the settings page."""
    return render_template('settings.html', config=CONFIG)

# API endpoints
@app.route('/api/status')
@login_required
def get_status():
    """Get the current system status."""
    return jsonify({
        "blockchain": system_status.blockchain_status,
        "radio": system_status.radio_status,
        "receiver": system_status.receiver_status,
        "torrent": system_status.torrent_status,
        "alerts": system_status.alerts
    })

@app.route('/api/logs/<component>')
@login_required
def get_logs(component):
    """Get logs for a specific component."""
    if component in system_status.logs:
        return jsonify(system_status.logs[component])
    return jsonify([])

@app.route('/api/alerts')
@login_required
def get_alerts():
    """Get all system alerts."""
    return jsonify(system_status.alerts)

@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
@login_required
def acknowledge_alert(alert_id):
    """Acknowledge an alert."""
    for alert in system_status.alerts:
        if alert["id"] == alert_id:
            alert["acknowledged"] = True
            break
            
    # Emit update to clients
    socketio.emit('alerts_update', system_status.alerts)
    return jsonify({"success": True})

@app.route('/api/blockchain/publish', methods=['POST'])
@login_required
def publish_to_blockchain():
    """Publish a new command to the blockchain."""
    try:
        data = request.get_json()
        if not data or "command" not in data:
            return jsonify({"error": "Command data required"}), 400
            
        # Post to the blockchain API
        response = requests.post(
            f"{CONFIG['blockchain_api']}/blocks",
            json={"data": data},
            timeout=5
        )
        
        if response.status_code == 201:
            new_block = response.json()
            system_status.add_log(
                "blockchain", 
                f"Published new block #{new_block.get('index', 'unknown')}", 
                "INFO"
            )
            return jsonify({"success": True, "block": new_block})
        else:
            error_message = f"Failed to publish to blockchain: {response.status_code}"
            system_status.add_log(
                "blockchain", 
                error_message, 
                "ERROR"
            )
            return jsonify({"error": error_message}), 400
            
    except Exception as e:
        logger.error(f"Error publishing to blockchain: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/radio/transmit', methods=['POST'])
@login_required
def trigger_transmission():
    """Trigger a manual radio transmission."""
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "Message data required"}), 400
            
        # In a real implementation, this would call the transmitter API
        # For now, we'll just log it
        system_status.add_log(
            "radio", 
            f"Manual transmission requested: {data['message']}", 
            "INFO"
        )
        
        system_status.add_alert(
            "radio", 
            "info", 
            "Manual transmission requested"
        )
        
        return jsonify({"success": True, "message": "Transmission requested"})
        
    except Exception as e:
        logger.error(f"Error triggering transmission: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/torrent/add', methods=['POST'])
@login_required
def add_torrent():
    """Add a new torrent for distribution."""
    try:
        data = request.get_json()
        if not data or "magnet" not in data:
            return jsonify({"error": "Magnet URI required"}), 400
            
        # In a real implementation, this would call the torrent client API
        # For now, we'll just log it
        system_status.add_log(
            "torrent", 
            f"Torrent added: {data['magnet'][:30]}...", 
            "INFO"
        )
        
        return jsonify({"success": True, "message": "Torrent added"})
        
    except Exception as e:
        logger.error(f"Error adding torrent: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/settings', methods=['POST'])
@login_required
def update_settings():
    """Update dashboard settings."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Settings data required"}), 400
            
        # Update the configuration
        for key, value in data.items():
            if key in CONFIG:
                CONFIG[key] = value
                
        # Special handling for password
        if "new_password" in data and data["new_password"]:
            CONFIG["admin_password_hash"] = generate_password_hash(data["new_password"])
            
        # Save to file
        config_path = os.environ.get('DASHBOARD_CONFIG', 'dashboard_config.json')
        with open(config_path, 'w') as f:
            # Don't save sensitive data to file
            safe_config = CONFIG.copy()
            if "admin_password_hash" in safe_config:
                del safe_config["admin_password_hash"]
                
            json.dump(safe_config, f, indent=2)
            
        system_status.add_log(
            "blockchain", 
            "Settings updated", 
            "INFO"
        )
        
        return jsonify({"success": True})
        
    except Exception as e:
        logger.error(f"Error updating settings: {e}")
        return jsonify({"error": str(e)}), 500

# SocketIO events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    
    # Send initial data
    emit('blockchain_update', system_status.blockchain_status)
    emit('radio_update', system_status.radio_status)
    emit('receiver_update', system_status.receiver_status)
    emit('torrent_update', system_status.torrent_status)
    emit('alerts_update', system_status.alerts)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

# HTML Templates
def create_templates():
    """Create the HTML templates for the dashboard."""
    os.makedirs('templates', exist_ok=True)
    
    # Base template
    base_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1