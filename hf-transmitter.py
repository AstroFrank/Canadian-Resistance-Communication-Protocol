#!/usr/bin/env python3
"""
HF Radio Transmitter for C2 System - WSPR-like beacon transmission

This module manages the transmission of command information via HF radio using
WSPR-like signals for covert, highly robust data dissemination. It interfaces
with FlexRadio hardware and uses frequency hopping for stealth.
"""

import os
import sys
import time
import json
import base64
import signal
import hashlib
import logging
import argparse
import subprocess
import numpy as np
from datetime import datetime, timedelta
import requests
import threading
import schedule
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hf_transmitter.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HF-Transmitter")

# Try to import FlexRadio API
try:
    import Hamlib
    HAS_HAMLIB = True
except ImportError:
    logger.warning("Hamlib not found. FlexRadio control will be simulated.")
    HAS_HAMLIB = False

# Try to import PQ crypto
try:
    from pqcrypto.kem.kyber import generate_keypair as kyber_generate
    from pqcrypto.kem.kyber import encrypt as kyber_encrypt
    HAS_PQCRYPTO = True
except ImportError:
    logger.warning("PQ crypto libraries not found. Using fallback encryption.")
    HAS_PQCRYPTO = False

    def kyber_generate():
        """Mock implementation of Kyber key generation."""
        priv = hashlib.sha256(b"kyber_private_key").digest()
        pub = hashlib.sha256(b"kyber_public_key").digest()
        return priv, pub

    def kyber_encrypt(message, public_key):
        """Mock implementation of Kyber encryption."""
        ciphertext = hashlib.sha256(message + public_key).digest()
        shared_secret = hashlib.sha256(message + b"shared").digest()
        return ciphertext, shared_secret


class WSPREncoder:
    """
    Encodes data into WSPR-compatible format for transmission.
    WSPR normally encodes callsign, grid locator, and power level.
    We repurpose these fields to carry encrypted command data.
    """
    
    def __init__(self, callsign_prefix="K1ABC", grid="FN20", power=10):
        """
        Initialize the WSPR encoder.
        
        Args:
            callsign_prefix (str): Base callsign to use (modified with data)
            grid (str): Grid locator to use (can be repurposed for data)
            power (int): Power level in dBm (can be repurposed for data)
        """
        self.callsign_prefix = callsign_prefix
        self.grid = grid
        self.power = power
        
        # WSPR message is 50 bits: 28 bits for callsign, 15 for grid, 7 for power
        self.callsign_bits = 28
        self.grid_bits = 15
        self.power_bits = 7
        
    def encode_data(self, data, client_pub_key=None):
        """
        Encode data into WSPR message fields.
        
        Args:
            data (bytes/str): Data to encode (typically a command reference)
            client_pub_key (bytes, optional): Public key to encrypt for
            
        Returns:
            tuple: (callsign, grid, power) - WSPR fields with encoded data
        """
        # If data is a string, convert to bytes
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        # If we have a client public key, encrypt the data with Kyber
        if client_pub_key and HAS_PQCRYPTO:
            logger.info("Encrypting payload with Kyber1024")
            ciphertext, _ = kyber_encrypt(data, client_pub_key)
            data_to_encode = ciphertext[:10]  # Take first 10 bytes
        else:
            # Otherwise, just hash the data as a reference
            data_hash = hashlib.sha256(data).digest()
            data_to_encode = data_hash[:10]  # Take first 10 bytes
            
        # Encode the data into Base32 for callsign-compatible chars
        # Using custom Base32 with ham radio compatible characters
        # This approach ensures the callsign looks legitimate
        encoded = base64.b32encode(data_to_encode).decode('ascii')
        
        # Create a valid-looking callsign with data embedded
        # Format: prefix + first 2-3 chars of encoded data
        callsign = f"{self.callsign_prefix}{encoded[:2]}"
        
        # Use rest of encoded data for grid and power
        grid_data = encoded[2:6]
        power_value = min(60, (int.from_bytes(data_to_encode[-1:], byteorder='big') % 59) + 1)
        
        return callsign, grid_data, power_value
        
    def generate_wsprsim_command(self, callsign, grid, power):
        """
        Generate command to execute wsprsim for creating audio.
        
        Args:
            callsign (str): Callsign with encoded data
            grid (str): Grid locator with encoded data
            power (int): Power level with encoded data
            
        Returns:
            str: Command line for wsprsim
        """
        return f"wsprsim -o output.wav {callsign} {grid} {power}"
        
    def create_audio_file(self, data, client_pub_key=None):
        """
        Create a WSPR audio file containing the encoded data.
        
        Args:
            data (bytes/str): Data to encode
            client_pub_key (bytes, optional): Public key for encryption
            
        Returns:
            str: Path to generated audio file
        """
        callsign, grid, power = self.encode_data(data, client_pub_key)
        
        cmd = self.generate_wsprsim_command(callsign, grid, power)
        logger.info(f"Running command: {cmd}")
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            logger.info("WSPR audio file created successfully")
            return "output.wav"
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create WSPR audio: {e.stderr.decode()}")
            return None


class FrequencyHopper:
    """
    Manages frequency hopping based on a pseudo-random schedule
    shared between transmitter and receivers.
    """
    
    def __init__(self, seed=None, bands=None):
        """
        Initialize the frequency hopper.
        
        Args:
            seed (str/bytes, optional): Seed for the PRNG
            bands (list, optional): List of (start_freq, end_freq) tuples
        """
        # Default to common WSPR frequencies if none provided
        self.bands = bands or [
            (1.836600, 1.836800),  # 160m
            (3.592600, 3.592800),  # 80m
            (5.287200, 5.287400),  # 60m
            (7.038600, 7.038800),  # 40m
            (10.138700, 10.138900),  # 30m
            (14.095600, 14.095800),  # 20m
            (18.104600, 18.104800),  # 17m
            (21.094600, 21.094800),  # 15m
            (24.924600, 24.924800),  # 12m
            (28.124600, 28.124800),  # 10m
        ]
        
        # Initialize the PRNG with the provided seed or a default
        if seed:
            if isinstance(seed, str):
                seed = seed.encode('utf-8')
            seed_hash = hashlib.sha256(seed).digest()
            seed_int = int.from_bytes(seed_hash[:4], byteorder='big')
        else:
            # Use current date as seed for daily rotation
            today = datetime.utcnow().strftime("%Y-%m-%d")
            seed_hash = hashlib.sha256(today.encode('utf-8')).digest()
            seed_int = int.from_bytes(seed_hash[:4], byteorder='big')
            
        # Initialize the random number generator with the seed
        self.rng = random.Random(seed_int)
        logger.info(f"Frequency hopper initialized with seed derived from: {seed or today}")
        
    def get_next_frequency(self):
        """
        Get the next frequency to use for transmission.
        
        Returns:
            float: Frequency in MHz
        """
        # Select a random band
        band = self.rng.choice(self.bands)
        
        # Choose a specific frequency within the band
        # For WSPR, we typically use the center of the band
        freq = (band[0] + band[1]) / 2
        
        logger.info(f"Next transmission frequency: {freq:.6f} MHz")
        return freq
        
    def get_daily_frequencies(self, count=24):
        """
        Generate a list of frequencies for a full day.
        
        Args:
            count (int): Number of frequencies to generate
            
        Returns:
            list: List of frequencies in MHz
        """
        frequencies = []
        for _ in range(count):
            frequencies.append(self.get_next_frequency())
        return frequencies


class FlexRadioController:
    """
    Controls the FlexRadio hardware for transmission.
    Uses Hamlib for radio control or simulates for testing.
    """
    
    def __init__(self, model="FLEX-6700", port="/dev/ttyUSB0", baudrate=115200):
        """
        Initialize the radio controller.
        
        Args:
            model (str): Radio model
            port (str): Serial port or IP address
            baudrate (int): Baud rate for serial connection
        """
        self.model = model
        self.port = port
        self.baudrate = baudrate
        self.radio = None
        
        # Initialize the radio connection
        if HAS_HAMLIB:
            self._init_hamlib()
        else:
            logger.info("Using simulated radio operations")
            
    def _init_hamlib(self):
        """Initialize the connection to the radio using Hamlib."""
        try:
            # Initialize Hamlib
            Hamlib.rig_set_debug(Hamlib.RIG_DEBUG_NONE)
            
            # Look up the model number
            model_num = None
            for item in dir(Hamlib):
                if item.startswith("RIG_MODEL_") and self.model in item:
                    model_num = getattr(Hamlib, item)
                    break
                    
            if not model_num:
                logger.warning(f"Model {self.model} not found, using DUMMY")
                model_num = Hamlib.RIG_MODEL_DUMMY
                
            # Create the radio object
            self.radio = Hamlib.Rig(model_num)
            
            # Set the connection parameters
            self.radio.set_conf("rig_pathname", self.port)
            self.radio.set_conf("serial_speed", str(self.baudrate))
            
            # Open the connection
            self.radio.open()
            logger.info(f"Connected to radio: {self.model}")
            
            # Get radio info
            info = self.radio.get_info()
            logger.info(f"Radio info: {info}")
            
        except Exception as e:
            logger.error(f"Failed to initialize radio: {e}")
            self.radio = None
            
    def set_frequency(self, freq_mhz):
        """
        Set the radio's frequency.
        
        Args:
            freq_mhz (float): Frequency in MHz
            
        Returns:
            bool: True if successful
        """
        freq_hz = int(freq_mhz * 1e6)
        
        if HAS_HAMLIB and self.radio:
            try:
                self.radio.set_freq(Hamlib.RIG_VFO_CURR, freq_hz)
                logger.info(f"Set frequency to {freq_mhz:.6f} MHz")
                return True
            except Exception as e:
                logger.error(f"Failed to set frequency: {e}")
                return False
        else:
            # Simulate setting frequency
            logger.info(f"[SIMULATED] Set frequency to {freq_mhz:.6f} MHz")
            return True
            
    def set_mode(self, mode="USB", bandwidth=2800):
        """
        Set the radio's mode.
        
        Args:
            mode (str): Mode (USB, LSB, CW, etc.)
            bandwidth (int): Bandwidth in Hz
            
        Returns:
            bool: True if successful
        """
        if HAS_HAMLIB and self.radio:
            try:
                mode_value = getattr(Hamlib, f"RIG_MODE_{mode}")
                self.radio.set_mode(Hamlib.RIG_VFO_CURR, mode_value, bandwidth)
                logger.info(f"Set mode to {mode}, bandwidth {bandwidth} Hz")
                return True
            except Exception as e:
                logger.error(f"Failed to set mode: {e}")
                return False
        else:
            # Simulate setting mode
            logger.info(f"[SIMULATED] Set mode to {mode}, bandwidth {bandwidth} Hz")
            return True
            
    def set_power(self, power_percent):
        """
        Set the radio's power level.
        
        Args:
            power_percent (float): Power level (0-100%)
            
        Returns:
            bool: True if successful
        """
        if HAS_HAMLIB and self.radio:
            try:
                # Hamlib uses a 0-1 scale for power
                power_norm = power_percent / 100.0
                self.radio.set_level("RFPOWER", power_norm)
                logger.info(f"Set power to {power_percent}%")
                return True
            except Exception as e:
                logger.error(f"Failed to set power: {e}")
                return False
        else:
            # Simulate setting power
            logger.info(f"[SIMULATED] Set power to {power_percent}%")
            return True
            
    def ptt_on(self):
        """
        Turn on the transmitter (PTT).
        
        Returns:
            bool: True if successful
        """
        if HAS_HAMLIB and self.radio:
            try:
                self.radio.set_ptt(Hamlib.RIG_VFO_CURR, Hamlib.RIG_PTT_ON)
                logger.info("PTT ON")
                return True
            except Exception as e:
                logger.error(f"Failed to set PTT ON: {e}")
                return False
        else:
            # Simulate PTT
            logger.info("[SIMULATED] PTT ON")
            return True
            
    def ptt_off(self):
        """
        Turn off the transmitter (PTT).
        
        Returns:
            bool: True if successful
        """
        if HAS_HAMLIB and self.radio:
            try:
                self.radio.set_ptt(Hamlib.RIG_VFO_CURR, Hamlib.RIG_PTT_OFF)
                logger.info("PTT OFF")
                return True
            except Exception as e:
                logger.error(f"Failed to set PTT OFF: {e}")
                return False
        else:
            # Simulate PTT
            logger.info("[SIMULATED] PTT OFF")
            return True
            
    def play_audio_file(self, audio_file, timeout=120):
        """
        Play an audio file through the radio.
        
        Args:
            audio_file (str): Path to audio file
            timeout (int): Timeout in seconds
            
        Returns:
            bool: True if successful
        """
        if not os.path.exists(audio_file):
            logger.error(f"Audio file not found: {audio_file}")
            return False
            
        try:
            # Turn on PTT
            self.ptt_on()
            
            # Play audio through system audio device (connected to radio's audio input)
            # This is a simple approach - in production would use a direct audio API
            cmd = f"aplay {audio_file}"
            
            logger.info(f"Playing audio file: {audio_file}")
            subprocess.run(cmd, shell=True, check=True, timeout=timeout)
            
            # Wait for WSPR transmission to complete (typically 110.6 seconds)
            time.sleep(2)  # Additional safety margin
            
            # Turn off PTT
            self.ptt_off()
            
            logger.info("Audio playback completed successfully")
            return True
            
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to play audio: {e}")
            self.ptt_off()  # Ensure PTT is off
            return False
            
    def close(self):
        """Close the connection to the radio."""
        if HAS_HAMLIB and self.radio:
            try:
                self.radio.close()
                logger.info("Radio connection closed")
            except Exception as e:
                logger.error(f"Error closing radio connection: {e}")


class CommandTransmitter:
    """
    Main controller for transmitting commands via HF radio.
    """
    
    def __init__(self, blockchain_url="http://localhost:5000", config_file="transmitter_config.json"):
        """
        Initialize the command transmitter.
        
        Args:
            blockchain_url (str): URL of the blockchain API
            config_file (str): Path to configuration file
        """
        self.blockchain_url = blockchain_url
        self.config_file = config_file
        self.running = False
        self.config = self._load_config()
        
        # Initialize components
        self.wspr_encoder = WSPREncoder(
            callsign_prefix=self.config.get("