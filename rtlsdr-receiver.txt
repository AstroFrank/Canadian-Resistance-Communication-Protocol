#!/usr/bin/env python3
"""
RTL-SDR Receiver for Quantum-Resilient C2 System

This module implements the client-side receiver component using RTL-SDR hardware
to monitor HF bands for WSPR-like transmissions containing command data.
The system follows a predetermined frequency hopping pattern and integrates
with existing WSPR decoders for optimal signal detection in poor conditions.

Key features:
- Automated frequency hopping based on shared secret schedule
- Integration with WSJT-X or other WSPR decoders
- Post-quantum cryptographic verification of received data
- Integration with blockchain and torrent systems for command retrieval
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
import datetime
import threading
import schedule
import random
import re
import tempfile
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rtl_receiver.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("RTL-SDR-Receiver")

# Try to import SDR libraries
try:
    import numpy as np
    from scipy import signal as sig
    import rtlsdr
    HAS_RTLSDR = True
except ImportError:
    logger.warning("RTL-SDR libraries not found. Running in simulation mode.")
    HAS_RTLSDR = False

# Try to import PQ crypto libraries
try:
    from pqcrypto.kem.kyber import decrypt as kyber_decrypt
    from pqcrypto.sign.dilithium import verify as dilithium_verify
    HAS_PQCRYPTO = True
except ImportError:
    logger.warning("PQ crypto libraries not found. Using fallback verification.")
    HAS_PQCRYPTO = False
    
    def kyber_decrypt(ciphertext, private_key):
        """Mock implementation of Kyber decryption."""
        return hashlib.sha256(ciphertext + b"shared").digest()
        
    def dilithium_verify(message, signature, public_key):
        """Mock implementation of Dilithium verification."""
        expected_sig = hashlib.sha256(message + b"dilithium_private_key").digest()
        return signature == expected_sig


class FrequencySchedule:
    """
    Manages the frequency hopping schedule based on a shared secret.
    This must be synchronized with the transmitter's schedule.
    """
    
    def __init__(self, seed=None, bands=None):
        """
        Initialize the frequency schedule.
        
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
            today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
            seed_hash = hashlib.sha256(today.encode('utf-8')).digest()
            seed_int = int.from_bytes(seed_hash[:4], byteorder='big')
            
        # Initialize the random number generator with the seed
        self.rng = random.Random(seed_int)
        logger.info(f"Frequency schedule initialized with seed derived from: {seed or today}")
        
    def get_frequency_for_time(self, target_time=None):
        """
        Get the frequency to use for a specific time.
        
        Args:
            target_time (datetime, optional): Time to get frequency for, defaults to now
            
        Returns:
            float: Frequency in MHz
        """
        if target_time is None:
            target_time = datetime.datetime.utcnow()
            
        # Reset the RNG state based on the date
        date_str = target_time.strftime("%Y-%m-%d")
        date_hash = hashlib.sha256(date_str.encode('utf-8')).digest()
        date_seed = int.from_bytes(date_hash[:4], byteorder='big')
        self.rng.seed(date_seed)
        
        # Calculate how many 2-hour periods have elapsed in the day
        # WSPR traditionally transmits on even-numbered hours
        hour_index = target_time.hour // 2
        
        # Advance the RNG to the appropriate state for this hour
        for _ in range(hour_index):
            self.rng.random()
            
        # Select a band from the list
        band = self.rng.choice(self.bands)
        
        # Return the center frequency of the band
        freq = (band[0] + band[1]) / 2
        logger.info(f"Frequency for {target_time.isoformat()}: {freq:.6f} MHz")
        
        return freq
        
    def get_current_and_next_frequencies(self, count=3):
        """
        Get the current frequency and upcoming frequencies.
        
        Args:
            count (int): Number of future frequencies to return
            
        Returns:
            list: List of (datetime, frequency) tuples
        """
        now = datetime.datetime.utcnow()
        # Round down to the nearest even hour
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        if current_hour.hour % 2 == 1:
            current_hour = current_hour.replace(hour=current_hour.hour - 1)
            
        result = []
        for i in range(count):
            target_time = current_hour + datetime.timedelta(hours=i*2)
            freq = self.get_frequency_for_time(target_time)
            result.append((target_time, freq))
            
        return result


class RTLSDRController:
    """
    Controls the RTL-SDR hardware for reception.
    """
    
    def __init__(self, device_index=0, sample_rate=2048000, gain='auto'):
        """
        Initialize the RTL-SDR controller.
        
        Args:
            device_index (int): RTL-SDR device index
            sample_rate (int): Sample rate in Hz
            gain (str/float): Gain setting, 'auto' or value in dB
        """
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.gain = gain
        self.device = None
        self.center_freq = None
        
        if HAS_RTLSDR:
            try:
                self.device = rtlsdr.RtlSdr(device_index)
                self.device.sample_rate = sample_rate
                
                if gain == 'auto':
                    self.device.set_agc_mode(True)
                else:
                    self.device.set_agc_mode(False)
                    self.device.gain = float(gain)
                    
                logger.info(f"RTL-SDR initialized: device={device_index}, sample_rate={sample_rate}, gain={gain}")
            except Exception as e:
                logger.error(f"Failed to initialize RTL-SDR: {e}")
                self.device = None
        else:
            logger.info("Using simulated RTL-SDR operations")
            
    def set_frequency(self, freq_mhz):
        """
        Set the center frequency of the RTL-SDR.
        
        Args:
            freq_mhz (float): Frequency in MHz
            
        Returns:
            bool: True if successful
        """
        if not HAS_RTLSDR or not self.device:
            logger.info(f"[SIMULATED] Setting frequency to {freq_mhz:.6f} MHz")
            self.center_freq = freq_mhz * 1e6
            return True
            
        try:
            freq_hz = int(freq_mhz * 1e6)
            self.device.center_freq = freq_hz
            self.center_freq = freq_hz
            logger.info(f"Set RTL-SDR frequency to {freq_mhz:.6f} MHz")
            return True
        except Exception as e:
            logger.error(f"Failed to set frequency: {e}")
            return False
            
    def read_samples(self, num_samples=131072):
        """
        Read samples from the RTL-SDR.
        
        Args:
            num_samples (int): Number of samples to read
            
        Returns:
            numpy.ndarray: Complex samples
        """
        if not HAS_RTLSDR or not self.device:
            # Generate simulated noise samples
            logger.info(f"[SIMULATED] Reading {num_samples} samples")
            return np.random.normal(0, 0.1, num_samples) + 1j * np.random.normal(0, 0.1, num_samples)
            
        try:
            samples = self.device.read_samples(num_samples)
            logger.debug(f"Read {len(samples)} samples from RTL-SDR")
            return samples
        except Exception as e:
            logger.error(f"Failed to read samples: {e}")
            return np.array([])
            
    def save_iq_data(self, filename, duration_seconds=120):
        """
        Save raw IQ data to a file for later processing.
        
        Args:
            filename (str): Output filename
            duration_seconds (int): Duration in seconds
            
        Returns:
            bool: True if successful
        """
        if not HAS_RTLSDR or not self.device:
            logger.info(f"[SIMULATED] Saving IQ data to {filename}")
            # Create a simulated IQ file with random data
            with open(filename, 'wb') as f:
                simulated_samples = np.random.normal(0, 0.1, int(self.sample_rate * duration_seconds)) + \
                                   1j * np.random.normal(0, 0.1, int(self.sample_rate * duration_seconds))
                simulated_samples.astype(np.complex64).tofile(f)
            return True
            
        try:
            # Calculate the total number of samples to collect
            num_samples = int(self.sample_rate * duration_seconds)
            
            # Read the samples
            samples = self.device.read_samples(num_samples)
            
            # Save to file
            with open(filename, 'wb') as f:
                samples.astype(np.complex64).tofile(f)
                
            logger.info(f"Saved {len(samples)} samples to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save IQ data: {e}")
            return False
            
    def close(self):
        """Close the connection to the RTL-SDR."""
        if HAS_RTLSDR and self.device:
            try:
                self.device.close()
                logger.info("RTL-SDR connection closed")
            except Exception as e:
                logger.error(f"Error closing RTL-SDR: {e}")


class WSPRDecoder:
    """
    Decodes WSPR signals from IQ data, either using internal methods
    or by calling external decoders like WSJT-X's wsprd.
    """
    
    def __init__(self, decoder_path=None, working_dir=None):
        """
        Initialize the WSPR decoder.
        
        Args:
            decoder_path (str, optional): Path to external decoder binary
            working_dir (str, optional): Working directory for decoder
        """
        self.decoder_path = decoder_path or self._find_decoder()
        self.working_dir = working_dir or tempfile.mkdtemp()
        
        # Ensure working directory exists
        os.makedirs(self.working_dir, exist_ok=True)
        
        if self.decoder_path:
            logger.info(f"Using external WSPR decoder: {self.decoder_path}")
        else:
            logger.warning("No external WSPR decoder found, will use internal methods")
            
    def _find_decoder(self):
        """
        Find an installed WSPR decoder.
        
        Returns:
            str: Path to decoder binary, or None if not found
        """
        # Try to find wsprd from WSJT-X
        paths_to_check = [
            "/usr/bin/wsprd",
            "/usr/local/bin/wsprd",
            "/opt/wsjt/bin/wsprd",
            "C:\\WSJT-X\\bin\\wsprd.exe",
        ]
        
        for path in paths_to_check:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
                
        return None
        
    def decode_iq_file(self, iq_filename, sample_rate, center_freq):
        """
        Decode WSPR signals from an IQ file.
        
        Args:
            iq_filename (str): Path to IQ data file
            sample_rate (int): Sample rate in Hz
            center_freq (int): Center frequency in Hz
            
        Returns:
            list: Decoded WSPR messages
        """
        # Convert IQ file to wav for external decoder
        wav_filename = os.path.join(self.working_dir, "wspr_decode.wav")
        
        if not self._convert_iq_to_wav(iq_filename, wav_filename, sample_rate, center_freq):
            logger.error("Failed to convert IQ to WAV")
            return []
            
        # Run the decoder
        if self.decoder_path:
            return self._run_external_decoder(wav_filename, center_freq)
        else:
            return self._run_internal_decoder(wav_filename, center_freq)
            
    def _convert_iq_to_wav(self, iq_filename, wav_filename, sample_rate, center_freq):
        """
        Convert IQ data to WAV format for decoding.
        
        Args:
            iq_filename (str): Input IQ file
            wav_filename (str): Output WAV file
            sample_rate (int): Sample rate in Hz
            center_freq (int): Center frequency in Hz
            
        Returns:
            bool: True if successful
        """
        try:
            # Use sox for conversion if available
            sox_cmd = f"sox -t raw -r {sample_rate} -e floating-point -b 32 -c 2 -L " \
                      f"{iq_filename} {wav_filename} rate 12000"
                      
            result = subprocess.run(sox_cmd, shell=True, check=False, 
                                   stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                                   
            if result.returncode != 0:
                logger.warning(f"Sox conversion failed: {result.stderr.decode()}")
                logger.info("Falling back to internal conversion")
                
                # Fall back to internal conversion
                if HAS_RTLSDR:
                    # Read the IQ data
                    iq_data = np.fromfile(iq_filename, dtype=np.complex64)
                    
                    # Resample to 12kHz (required for WSPR decoding)
                    resampled = sig.resample_poly(iq_data, 12000, sample_rate)
                    
                    # Convert to audio (WSPR decoders expect real audio)
                    # Just take the real component as a simple approach
                    audio = np.real(resampled)
                    
                    # Normalize
                    audio = audio / np.max(np.abs(audio))
                    
                    # Save as WAV
                    from scipy.io import wavfile
                    wavfile.write(wav_filename, 12000, audio.astype(np.float32))
                else:
                    # In simulation mode, create a dummy WAV file
                    with open(wav_filename, 'wb') as f:
                        f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\x3e\x00\x00\x00\x7d\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
                        
            logger.info(f"Converted IQ data to WAV: {wav_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert IQ to WAV: {e}")
            return False
            
    def _run_external_decoder(self, wav_filename, center_freq):
        """
        Run an external WSPR decoder.
        
        Args:
            wav_filename (str): Input WAV file
            center_freq (int): Center frequency in Hz
            
        Returns:
            list: Decoded WSPR messages
        """
        try:
            # Create a directory for output files
            output_dir = os.path.join(self.working_dir, "decode_out")
            os.makedirs(output_dir, exist_ok=True)
            
            # Set up the command for wsprd
            # wsprd expects center frequency in MHz
            freq_mhz = center_freq / 1e6
            base_filename = os.path.basename(wav_filename)
            
            # Format: wsprd [options] file1 [file2 ...]
            # -a all_wspr, -c (c2 files) -d deep_search -f freq -H (halt when complete) 
            cmd = f"{self.decoder_path} -a -f {freq_mhz:.6f} -H {wav_filename}"
            
            # Run the decoder
            logger.info(f"Running WSPR decoder: {cmd}")
            result = subprocess.run(cmd, shell=True, check=False, 
                                   stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                                   cwd=output_dir)
                                   
            if result.returncode != 0:
                logger.warning(f"Decoder returned non-zero: {result.stderr.decode()}")
                
            # Parse the output
            stdout = result.stdout.decode()
            logger.debug(f"Decoder output: {stdout}")
            
            # Parse the spots from stdout or from the ALL_WSPR.TXT file
            spots = []
            
            # Try to parse from stdout first
            for line in stdout.splitlines():
                if len(line.strip()) > 20:  # Skip short lines
                    spots.append(self._parse_wspr_spot(line))
                    
            # If no spots found in stdout, check the output file
            if not spots:
                wspr_file = os.path.join(output_dir, "ALL_WSPR.TXT")
                if os.path.exists(wspr_file):
                    with open(wspr_file, 'r') as f:
                        for line in f:
                            spots.append(self._parse_wspr_spot(line))
                            
            # Filter out None values (failed parses)
            spots = [s for s in spots if s]
            
            logger.info(f"Decoded {len(spots)} WSPR spots")
            return spots
            
        except Exception as e:
            logger.error(f"Error running external decoder: {e}")
            return []
            
    def _run_internal_decoder(self, wav_filename, center_freq):
        """
        Run a simple internal WSPR decoder for fallback.
        This is a placeholder - a real internal decoder would be much more complex.
        
        Args:
            wav_filename (str): Input WAV file
            center_freq (int): Center frequency in Hz
            
        Returns:
            list: Decoded WSPR messages
        """
        logger.warning("Internal WSPR decoder not implemented, returning test data")
        
        # Return simulated WSPR spots for testing
        freq_mhz = center_freq / 1e6
        return [
            {
                'timestamp': datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
                'snr': -19,
                'dt': 1.0,
                'freq': freq_mhz + 0.001458,
                'drift': 0,
                'callsign': 'K1ABC',
                'grid': 'FN20',
                'power': 37,
                'valid': True
            }
        ]
        
    def _parse_wspr_spot(self, line):
        """
        Parse a WSPR spot from text output.
        
        Args:
            line (str): Line from decoder output
            
        Returns:
            dict: Parsed WSPR spot or None if parsing failed
        """
        try:
            # Format: 2020-04-15 14:28  -19  1.0  14.097154  0  K1ABC FN20 37
            # Some versions of wsprd have different output formats
            
            # Try the standard WSJT-X format first
            pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})\s+(-?\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+)\s+(\S+)\s+(\S+)\s+(\d+)'
            match = re.match(pattern, line.strip())
            
            if not match:
                # Try alternative format
                pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(\S+)\s+(-?\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+)\s+(\d+)'
                match = re.match(pattern, line.strip())
                
                if match:
                    # Rearrange the fields for the alternative format
                    timestamp, callsign, grid, snr, dt, freq, drift, power = match.groups()
                else:
                    # Try even simpler format as last resort
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        timestamp = parts[0] + " " + parts[1]
                        snr = parts[2]
                        dt = parts[3]
                        freq = parts[4]
                        drift = parts[5]
                        callsign = parts[6]
                        grid = parts[7]
                        power = parts[8] if len(parts) > 8 else "0"
                    else:
                        logger.warning(f"Could not parse WSPR spot: {line}")
                        return None
            else:
                timestamp, snr, dt, freq, drift, callsign, grid, power = match.groups()
                
            return {
                'timestamp': timestamp,
                'snr': int(snr),
                'dt': float(dt),
                'freq': float(freq),
                'drift': int(drift),
                'callsign': callsign,
                'grid': grid,
                'power': int(power),
                'valid': True  # Assume valid
            }
            
        except Exception as e:
            logger.warning(f"Failed to parse WSPR spot: {line}, error: {e}")
            return None


class CommandDecoder:
    """
    Extracts and validates command data from WSPR messages.
    """
    
    def __init__(self, private_key=None, operator_public_key=None):
        """
        Initialize the command decoder.
        
        Args:
            private_key (bytes, optional): Client's Kyber private key
            operator_public_key (bytes, optional): Operator's Dilithium public key
        """
        self.private_key = private_key
        self.operator_public_key = operator_public_key
        
    def decode_command_from_wspr(self, wspr_spots):
        """
        Extract command data from WSPR spots.
        
        Args:
            wspr_spots (list): List of WSPR spot dictionaries
            
        Returns:
            dict: Decoded command or None if no valid command
        """
        if not wspr_spots:
            logger.info("No WSPR spots to decode")
            return None
            
        valid_commands = []
        
        for spot in wspr_spots:
            if not spot.get('valid', False):
                continue
                
            try:
                # Extract data from the fields
                callsign = spot.get('callsign', '')
                grid = spot.get('grid', '')
                power = spot.get('power', 0)
                
                # Check for our command pattern
                if not callsign.startswith('K1'):
                    continue
                    
                # Extract encoded data
                encoded_data = callsign[3:] + grid
                
                # Attempt to decode
                try:
                    # Convert to Base32 and handle padding
                    padding = '=' * (8 - (len(encoded_data) % 8))
                    decoded_bytes = base64.b32decode(encoded_data + padding)
                    
                    # If encrypted, decrypt with Kyber
                    if HAS_PQCRYPTO and self.private_key:
                        try:
                            decrypted = kyber_decrypt(decoded_bytes, self.private_key)
                            command_data = decrypted
                        except Exception as e:
                            logger.warning(f"Failed to decrypt with Kyber: {e}")
                            command_data = decoded_bytes
                    else:
                        command_data = decoded_bytes
                        
                    # Try to parse as JSON
                    try:
                        command_json = json.loads(command_data)
                        logger.info(f"Decoded command: {command_json}")
                        valid_commands.append(command_json)
                    except json.JSONDecodeError:
                        # Not JSON, treat as string or binary data
                        try:
                            command_str = command_data.decode('utf-8')
                            logger.info(f"Decoded command string: {command_str}")
                            valid_commands.append({'command': command_str})
                        except UnicodeDecodeError:
                            # Binary data, possibly a hash or reference
                            logger.info(f"Decoded binary data: {command_data.hex()}")
                            valid_commands.append({'command_reference': command_data.hex()})
                            
                except Exception as e:
                    logger.warning(f"Failed to decode Base32 data: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error processing WSPR spot: {e}")
                continue
                
        # Return the first valid command, or None if none found
        return valid_commands[0] if valid_commands else None
        
    def verify_command_signature(self, command, signature):
        """
        Verify a command's signature using Dilithium.
        
        Args:
            command (bytes/dict): Command data
            signature (bytes): Dilithium signature
            
        Returns:
            bool: True if signature is valid
        """
        if not self.operator_public_key:
            logger.warning("No operator public key available for verification")
            return False
            
        if not HAS_PQCRYPTO:
            logger.warning("PQ crypto not available for signature verification")
            return False
            
        try:
            # Convert command to bytes if it's a dict
            if isinstance(command, dict):
                command_bytes = json.dumps(command, sort_keys=True).encode()
            elif isinstance(command, str):
                command_bytes = command.encode()
            else:
                command_bytes = command
                
            # Verify the signature
            return dilithium_verify(command_bytes, signature, self.operator_public_key)
            
        except Exception as e:
            logger.error(f"Error verifying command signature: {e}")
            return False


class TorrentClient:
    """
    Simple interface to a BitTorrent client for retrieving command files.
    """
    
    def __init__(self, download_dir=None):
        """
        Initialize the torrent client interface.
        
        Args:
            download_dir (str, optional): Directory for downloaded files
        """
        self.download_dir = download_dir or os.path.join(os.path.expanduser("~"), "downloads")
        
        # Ensure download directory exists
        os.makedirs(self.download_dir, exist_ok=True)
        
    def download_magnet(self, magnet_uri, timeout=3600):
        """
        Download content from a magnet URI.
        
        Args:
            magnet_uri (str): Magnet URI
            timeout (int): Timeout in seconds
            
        Returns:
            str: Path to downloaded file, or None on failure
        """
        logger.info(f"Starting download of magnet URI: {magnet_uri}")
        
        try:
            # Call an external torrent client
            # This is a simplified example - in production you'd want to use
            # a proper torrent library like libtorrent, or interface with
            # a running torrent client via API
            cmd = f"transmission-cli --download-dir {self.download_dir} \"{magnet_uri}\""
            
            # For simulation purposes, we'll just log the command
            logger.info(f"Would execute: {cmd}")
            
            # In a real implementation, you would:
            # 1. Execute the command or use a proper library
            # 2. Monitor the download progress
            # 3. Wait for completion
            # 4. Return the path to the downloaded file
            
            # For now, just simulate a successful download
            download_path = os.path.join(self.download_dir, "command_data.bin")
            
            # Create a dummy file for testing
            with open(download_path, 'wb') as f:
                f.write(b'This is simulated torrent download content')
                
            logger.info(f"Downloaded file to: {download_path}")
            return download_path
            
        except Exception as e:
            logger.error(f"Failed to download magnet URI: {e}")
            return None
            

class BlockchainClient:
    """
    Client for interacting with the blockchain API.
    """
    
    def __init__(self, api_url="http://localhost:5000"):
        """
        Initialize the blockchain client.
        
        Args:
            api_url (str): URL of the blockchain API
        """
        self.api_url = api_url
        
    def get_latest_command(self):
        """
        Get the latest command from the blockchain.