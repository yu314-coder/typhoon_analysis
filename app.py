import os
import argparse
import logging
import pickle
import threading
import time
import warnings
from datetime import datetime, timedelta
from collections import defaultdict
import csv 
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='umap')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.interpolate import interp1d, RBFInterpolator
import statsmodels.api as sm
import requests
import tempfile
import shutil
import xarray as xr

# NEW: Advanced ML imports
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available - clustering features limited")

# Optional CNN imports with robust error handling
CNN_AVAILABLE = False
try:
    # Set environment variables before importing TensorFlow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
    import tensorflow as tf
    from tensorflow.keras import layers, models
    # Test if TensorFlow actually works
    tf.config.set_visible_devices([], 'GPU')  # Disable GPU to avoid conflicts
    CNN_AVAILABLE = True
    print("TensorFlow successfully loaded - CNN features enabled")
except Exception as e:
    CNN_AVAILABLE = False
    print(f"TensorFlow not available - CNN features disabled: {str(e)[:100]}...")

try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    CDSAPI_AVAILABLE = False

import tropycal.tracks as tracks

# -----------------------------
# Configuration and Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Remove argument parser to simplify startup
DATA_PATH = '/tmp/typhoon_data' if 'SPACE_ID' in os.environ else tempfile.gettempdir()

# Ensure directory exists and is writable
try:
    os.makedirs(DATA_PATH, exist_ok=True)
    # Test write permissions
    test_file = os.path.join(DATA_PATH, 'test_write.txt')
    with open(test_file, 'w') as f:
        f.write('test')
    os.remove(test_file)
    logging.info(f"Data directory is writable: {DATA_PATH}")
except Exception as e:
    logging.warning(f"Data directory not writable, using temp dir: {e}")
    DATA_PATH = tempfile.mkdtemp()
    logging.info(f"Using temporary directory: {DATA_PATH}")

# Update file paths
ONI_DATA_PATH = os.path.join(DATA_PATH, 'oni_data.csv')
TYPHOON_DATA_PATH = os.path.join(DATA_PATH, 'processed_typhoon_data.csv')
MERGED_DATA_CSV = os.path.join(DATA_PATH, 'merged_typhoon_era5_data.csv')

# IBTrACS settings
BASIN_FILES = {
    'EP': 'ibtracs.EP.list.v04r01.csv',
    'NA': 'ibtracs.NA.list.v04r01.csv',
    'WP': 'ibtracs.WP.list.v04r01.csv'
}
IBTRACS_BASE_URL = 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/'
LOCAL_IBTRACS_PATH = os.path.join(DATA_PATH, 'ibtracs.WP.list.v04r01.csv')
CACHE_FILE = os.path.join(DATA_PATH, 'ibtracs_cache.pkl')
CACHE_EXPIRY_DAYS = 1

# -----------------------------
# ENHANCED: Color Maps and Standards with TD Support (FIXED TAIWAN CLASSIFICATION)
# -----------------------------
# Enhanced color mapping with TD support (for Plotly)
enhanced_color_map = {
    'Unknown': 'rgb(200, 200, 200)',
    'Tropical Depression': 'rgb(128, 128, 128)',  # Gray for TD
    'Tropical Storm': 'rgb(0, 0, 255)',
    'C1 Typhoon': 'rgb(0, 255, 255)',
    'C2 Typhoon': 'rgb(0, 255, 0)',
    'C3 Strong Typhoon': 'rgb(255, 255, 0)',
    'C4 Very Strong Typhoon': 'rgb(255, 165, 0)',
    'C5 Super Typhoon': 'rgb(255, 0, 0)'
}

# Matplotlib-compatible color mapping (hex colors)
matplotlib_color_map = {
    'Unknown': '#C8C8C8',
    'Tropical Depression': '#808080',  # Gray for TD
    'Tropical Storm': '#0000FF',       # Blue
    'C1 Typhoon': '#00FFFF',          # Cyan
    'C2 Typhoon': '#00FF00',          # Green
    'C3 Strong Typhoon': '#FFFF00',   # Yellow
    'C4 Very Strong Typhoon': '#FFA500', # Orange
    'C5 Super Typhoon': '#FF0000'     # Red
}

# FIXED Taiwan color mapping with correct categories
taiwan_color_map = {
    'Tropical Depression': '#808080',   # Gray
    'Tropical Storm': '#0000FF',       # Blue  
    'Moderate Typhoon': '#FFA500',     # Orange
    'Intense Typhoon': '#FF0000'       # Red
}

def rgb_string_to_hex(rgb_string):
    """Convert 'rgb(r,g,b)' string to hex color for matplotlib"""
    try:
        # Extract numbers from 'rgb(r,g,b)' format
        import re
        numbers = re.findall(r'\d+', rgb_string)
        if len(numbers) == 3:
            r, g, b = map(int, numbers)
            return f'#{r:02x}{g:02x}{b:02x}'
        else:
            return '#808080'  # Default gray
    except:
        return '#808080'  # Default gray

def get_matplotlib_color(category):
    """Get matplotlib-compatible color for a storm category"""
    return matplotlib_color_map.get(category, '#808080')

def get_taiwan_color(category):
    """Get Taiwan standard color for a storm category"""
    return taiwan_color_map.get(category, '#808080')

# Cluster colors for route visualization
CLUSTER_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D2B4DE'
]

# Route prediction colors
ROUTE_COLORS = [
    '#FF0066', '#00FF66', '#6600FF', '#FF6600', '#0066FF',
    '#FF00CC', '#00FFCC', '#CC00FF', '#CCFF00', '#00CCFF'
]

# Original color map for backward compatibility
color_map = {
    'C5 Super Typhoon': 'rgb(255, 0, 0)',
    'C4 Very Strong Typhoon': 'rgb(255, 165, 0)',
    'C3 Strong Typhoon': 'rgb(255, 255, 0)',
    'C2 Typhoon': 'rgb(0, 255, 0)',
    'C1 Typhoon': 'rgb(0, 255, 255)',
    'Tropical Storm': 'rgb(0, 0, 255)',
    'Tropical Depression': 'rgb(128, 128, 128)'
}

atlantic_standard = {
    'C5 Super Typhoon': {'wind_speed': 137, 'color': 'Red', 'hex': '#FF0000'},
    'C4 Very Strong Typhoon': {'wind_speed': 113, 'color': 'Orange', 'hex': '#FFA500'},
    'C3 Strong Typhoon': {'wind_speed': 96, 'color': 'Yellow', 'hex': '#FFFF00'},
    'C2 Typhoon': {'wind_speed': 83, 'color': 'Green', 'hex': '#00FF00'},
    'C1 Typhoon': {'wind_speed': 64, 'color': 'Cyan', 'hex': '#00FFFF'},
    'Tropical Storm': {'wind_speed': 34, 'color': 'Blue', 'hex': '#0000FF'},
    'Tropical Depression': {'wind_speed': 0, 'color': 'Gray', 'hex': '#808080'}
}

# FIXED Taiwan standard with correct official CWA thresholds
taiwan_standard = {
    'Intense Typhoon': {'wind_speed': 51.0, 'color': 'Red', 'hex': '#FF0000'},      # 100+ knots (51.0+ m/s)
    'Moderate Typhoon': {'wind_speed': 32.7, 'color': 'Orange', 'hex': '#FFA500'},  # 64-99 knots (32.7-50.9 m/s)
    'Tropical Storm': {'wind_speed': 17.2, 'color': 'Blue', 'hex': '#0000FF'},      # 34-63 knots (17.2-32.6 m/s)
    'Tropical Depression': {'wind_speed': 0, 'color': 'Gray', 'hex': '#808080'}     # <34 knots (<17.2 m/s)
}

# -----------------------------
# Utility Functions for HF Spaces
# -----------------------------

def safe_file_write(file_path, data_frame, backup_dir=None):
    """Safely write DataFrame to CSV with backup and error handling"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Try to write to a temporary file first
        temp_path = file_path + '.tmp'
        data_frame.to_csv(temp_path, index=False)
        
        # If successful, rename to final file
        os.rename(temp_path, file_path)
        logging.info(f"Successfully saved {len(data_frame)} records to {file_path}")
        return True
        
    except PermissionError as e:
        logging.warning(f"Permission denied writing to {file_path}: {e}")
        if backup_dir:
            try:
                backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                data_frame.to_csv(backup_path, index=False)
                logging.info(f"Saved to backup location: {backup_path}")
                return True
            except Exception as backup_e:
                logging.error(f"Failed to save to backup location: {backup_e}")
        return False
        
    except Exception as e:
        logging.error(f"Error saving file {file_path}: {e}")
        # Clean up temp file if it exists
        temp_path = file_path + '.tmp'
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False

def get_fallback_data_dir():
    """Get a fallback data directory that's guaranteed to be writable"""
    fallback_dirs = [
        tempfile.gettempdir(),
        '/tmp',
        os.path.expanduser('~'),
        os.getcwd()
    ]
    
    for directory in fallback_dirs:
        try:
            test_dir = os.path.join(directory, 'typhoon_fallback')
            os.makedirs(test_dir, exist_ok=True)
            test_file = os.path.join(test_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            return test_dir
        except:
            continue
    
    # If all else fails, use current directory
    return os.getcwd()

# -----------------------------
# ONI and Typhoon Data Functions
# -----------------------------

def download_oni_file(url, filename):
    """Download ONI file with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(response.content)
            return True
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1} failed to download ONI: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logging.error(f"Failed to download ONI after {max_retries} attempts")
                return False

def convert_oni_ascii_to_csv(input_file, output_file):
    """Convert ONI ASCII format to CSV"""
    data = defaultdict(lambda: [''] * 12)
    season_to_month = {'DJF':12, 'JFM':1, 'FMA':2, 'MAM':3, 'AMJ':4, 'MJJ':5,
                       'JJA':6, 'JAS':7, 'ASO':8, 'SON':9, 'OND':10, 'NDJ':11}
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.split()
                if len(parts) >= 4:
                    season, year, anom = parts[0], parts[1], parts[-1]
                    if season in season_to_month:
                        month = season_to_month[season]
                        if season == 'DJF':
                            year = str(int(year)-1)
                        data[year][month-1] = anom
        
        # Write to CSV with safe write
        df = pd.DataFrame(data).T.reset_index()
        df.columns = ['Year','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        df = df.sort_values('Year').reset_index(drop=True)
        
        return safe_file_write(output_file, df, get_fallback_data_dir())
        
    except Exception as e:
        logging.error(f"Error converting ONI file: {e}")
        return False

def update_oni_data():
    """Update ONI data with error handling"""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    temp_file = os.path.join(DATA_PATH, "temp_oni.ascii.txt")
    input_file = os.path.join(DATA_PATH, "oni.ascii.txt")
    output_file = ONI_DATA_PATH
    
    try:
        if download_oni_file(url, temp_file):
            if not os.path.exists(input_file) or not os.path.exists(output_file):
                os.rename(temp_file, input_file)
                convert_oni_ascii_to_csv(input_file, output_file)
            else:
                os.remove(temp_file)
        else:
            # Create fallback ONI data if download fails
            logging.warning("Creating fallback ONI data")
            create_fallback_oni_data(output_file)
    except Exception as e:
        logging.error(f"Error updating ONI data: {e}")
        create_fallback_oni_data(output_file)

def create_fallback_oni_data(output_file):
    """Create minimal ONI data for testing"""
    years = range(2000, 2026)  # Extended to include 2025
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    # Create synthetic ONI data
    data = []
    for year in years:
        row = [year]
        for month in months:
            # Generate some realistic ONI values
            value = np.random.normal(0, 1) * 0.5
            row.append(f"{value:.2f}")
        data.append(row)
    
    df = pd.DataFrame(data, columns=['Year'] + months)
    safe_file_write(output_file, df, get_fallback_data_dir())

# -----------------------------
# FIXED: IBTrACS Data Loading
# -----------------------------

def download_ibtracs_file(basin, force_download=False):
    """Download specific basin file from IBTrACS"""
    filename = BASIN_FILES[basin]
    local_path = os.path.join(DATA_PATH, filename)
    url = IBTRACS_BASE_URL + filename
    
    # Check if file exists and is recent (less than 7 days old)
    if os.path.exists(local_path) and not force_download:
        file_age = time.time() - os.path.getmtime(local_path)
        if file_age < 7 * 24 * 3600:  # 7 days
            logging.info(f"Using cached {basin} basin file")
            return local_path
    
    try:
        logging.info(f"Downloading {basin} basin file from {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        logging.info(f"Successfully downloaded {basin} basin file")
        return local_path
    except Exception as e:
        logging.error(f"Failed to download {basin} basin file: {e}")
        return None

def examine_ibtracs_structure(file_path):
    """Examine the actual structure of an IBTrACS CSV file"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Show first 5 lines
        logging.info("First 5 lines of IBTrACS file:")
        for i, line in enumerate(lines[:5]):
            logging.info(f"Line {i}: {line.strip()}")
        
        # The first line contains the actual column headers
        # No need to skip rows for IBTrACS v04r01
        df = pd.read_csv(file_path, nrows=5)
        logging.info(f"Columns from first row: {list(df.columns)}")
        
        return list(df.columns)
    except Exception as e:
        logging.error(f"Error examining IBTrACS structure: {e}")
        return None

def load_ibtracs_csv_directly(basin='WP'):
    """Load IBTrACS data directly from CSV - FIXED VERSION"""
    filename = BASIN_FILES[basin]
    local_path = os.path.join(DATA_PATH, filename)
    
    # Download if not exists
    if not os.path.exists(local_path):
        downloaded_path = download_ibtracs_file(basin)
        if not downloaded_path:
            return None
    
    try:
        # First, examine the structure
        actual_columns = examine_ibtracs_structure(local_path)
        if not actual_columns:
            logging.error("Could not examine IBTrACS file structure")
            return None
        
        # Read IBTrACS CSV - DON'T skip any rows for v04r01
        # The first row contains proper column headers
        logging.info(f"Reading IBTrACS CSV file: {local_path}")
        df = pd.read_csv(local_path, low_memory=False)  # Don't skip any rows
        
        logging.info(f"Original columns: {list(df.columns)}")
        logging.info(f"Data shape before cleaning: {df.shape}")
        
        # Check which essential columns exist
        required_cols = ['SID', 'ISO_TIME', 'LAT', 'LON']
        available_required = [col for col in required_cols if col in df.columns]
        
        if len(available_required) < 2:
            logging.error(f"Missing critical columns. Available: {list(df.columns)}")
            return None
        
        # Clean and standardize the data with format specification
        if 'ISO_TIME' in df.columns:
            df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        
        # Clean numeric columns
        numeric_columns = ['LAT', 'LON', 'WMO_WIND', 'WMO_PRES', 'USA_WIND', 'USA_PRES']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter out invalid/missing critical data
        valid_rows = df['LAT'].notna() & df['LON'].notna()
        df = df[valid_rows]
        
        # Ensure LAT/LON are in reasonable ranges
        df = df[(df['LAT'] >= -90) & (df['LAT'] <= 90)]
        df = df[(df['LON'] >= -180) & (df['LON'] <= 180)]
        
        # Add basin info if missing
        if 'BASIN' not in df.columns:
            df['BASIN'] = basin
        
        # Add default columns if missing
        if 'NAME' not in df.columns:
            df['NAME'] = 'UNNAMED'
        
        if 'SEASON' not in df.columns and 'ISO_TIME' in df.columns:
            df['SEASON'] = df['ISO_TIME'].dt.year
        
        logging.info(f"Successfully loaded {len(df)} records from {basin} basin")
        return df
        
    except Exception as e:
        logging.error(f"Error reading IBTrACS CSV file: {e}")
        return None

def load_ibtracs_data_fixed():
    """Fixed version of IBTrACS data loading"""
    ibtracs_data = {}
    
    # Try to load each basin, but prioritize WP for this application
    load_order = ['WP', 'EP', 'NA']
    
    for basin in load_order:
        try:
            logging.info(f"Loading {basin} basin data...")
            df = load_ibtracs_csv_directly(basin)
            
            if df is not None and not df.empty:
                ibtracs_data[basin] = df
                logging.info(f"Successfully loaded {basin} basin with {len(df)} records")
            else:
                logging.warning(f"No data loaded for basin {basin}")
                ibtracs_data[basin] = None
                
        except Exception as e:
            logging.error(f"Failed to load basin {basin}: {e}")
            ibtracs_data[basin] = None
    
    return ibtracs_data

def load_data_fixed(oni_path, typhoon_path):
    """Fixed version of load_data function"""
    # Load ONI data
    oni_data = pd.DataFrame({'Year': [], 'Jan': [], 'Feb': [], 'Mar': [], 'Apr': [], 
                           'May': [], 'Jun': [], 'Jul': [], 'Aug': [], 'Sep': [], 
                           'Oct': [], 'Nov': [], 'Dec': []})
    
    if not os.path.exists(oni_path):
        logging.warning(f"ONI data file not found: {oni_path}")
        update_oni_data()
    
    try:
        oni_data = pd.read_csv(oni_path)
        logging.info(f"Successfully loaded ONI data with {len(oni_data)} years")
    except Exception as e:
        logging.error(f"Error loading ONI data: {e}")
        update_oni_data()
        try:
            oni_data = pd.read_csv(oni_path)
        except Exception as e:
            logging.error(f"Still can't load ONI data: {e}")
    
    # Load typhoon data - NEW APPROACH
    typhoon_data = None
    
    # First, try to load from existing processed file
    if os.path.exists(typhoon_path):
        try:
            typhoon_data = pd.read_csv(typhoon_path, low_memory=False)
            # Ensure basic columns exist and are valid
            required_cols = ['LAT', 'LON']
            if all(col in typhoon_data.columns for col in required_cols):
                if 'ISO_TIME' in typhoon_data.columns:
                    typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'], errors='coerce')
                logging.info(f"Loaded processed typhoon data with {len(typhoon_data)} records")
            else:
                logging.warning("Processed typhoon data missing required columns, will reload from IBTrACS")
                typhoon_data = None
        except Exception as e:
            logging.error(f"Error loading processed typhoon data: {e}")
            typhoon_data = None
    
    # If no valid processed data, load from IBTrACS
    if typhoon_data is None or typhoon_data.empty:
        logging.info("Loading typhoon data from IBTrACS...")
        ibtracs_data = load_ibtracs_data_fixed()
        
        # Combine all available basin data, prioritizing WP
        combined_dfs = []
        for basin in ['WP', 'EP', 'NA']:
            if basin in ibtracs_data and ibtracs_data[basin] is not None:
                df = ibtracs_data[basin].copy()
                df['BASIN'] = basin
                combined_dfs.append(df)
        
        if combined_dfs:
            typhoon_data = pd.concat(combined_dfs, ignore_index=True)
            # Ensure SID has proper format
            if 'SID' not in typhoon_data.columns and 'BASIN' in typhoon_data.columns:
                # Create SID from basin and other identifiers if missing
                if 'SEASON' in typhoon_data.columns:
                    typhoon_data['SID'] = (typhoon_data['BASIN'].astype(str) + 
                                         typhoon_data.index.astype(str).str.zfill(2) + 
                                         typhoon_data['SEASON'].astype(str))
                else:
                    typhoon_data['SID'] = (typhoon_data['BASIN'].astype(str) + 
                                         typhoon_data.index.astype(str).str.zfill(2) + 
                                         '2000')
            
            # Save the processed data for future use
            safe_file_write(typhoon_path, typhoon_data, get_fallback_data_dir())
            logging.info(f"Combined IBTrACS data: {len(typhoon_data)} total records")
        else:
            logging.error("Failed to load any IBTrACS basin data")
            # Create minimal fallback data
            typhoon_data = create_fallback_typhoon_data()
    
    # Final validation of typhoon data
    if typhoon_data is not None:
        # Ensure required columns exist with fallback values
        required_columns = {
            'SID': 'UNKNOWN',
            'ISO_TIME': pd.Timestamp('2000-01-01'),
            'LAT': 0.0,
            'LON': 0.0,
            'USA_WIND': np.nan,
            'USA_PRES': np.nan,
            'NAME': 'UNNAMED',
            'SEASON': 2000
        }
        
        for col, default_val in required_columns.items():
            if col not in typhoon_data.columns:
                typhoon_data[col] = default_val
                logging.warning(f"Added missing column {col} with default value")
        
        # Ensure data types
        if 'ISO_TIME' in typhoon_data.columns:
            typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'], errors='coerce')
        typhoon_data['LAT'] = pd.to_numeric(typhoon_data['LAT'], errors='coerce')
        typhoon_data['LON'] = pd.to_numeric(typhoon_data['LON'], errors='coerce')
        typhoon_data['USA_WIND'] = pd.to_numeric(typhoon_data['USA_WIND'], errors='coerce')
        typhoon_data['USA_PRES'] = pd.to_numeric(typhoon_data['USA_PRES'], errors='coerce')
        
        # Remove rows with invalid coordinates
        typhoon_data = typhoon_data.dropna(subset=['LAT', 'LON'])
        
        logging.info(f"Final typhoon data: {len(typhoon_data)} records after validation")
    
    return oni_data, typhoon_data

def create_fallback_typhoon_data():
    """Create minimal fallback typhoon data - FIXED VERSION"""
    # Use proper pandas date_range instead of numpy
    dates = pd.date_range(start='2000-01-01', end='2025-12-31', freq='D')  # Extended to 2025
    storm_dates = dates[np.random.choice(len(dates), size=100, replace=False)]
    
    data = []
    for i, date in enumerate(storm_dates):
        # Create realistic WP storm tracks
        base_lat = np.random.uniform(10, 30)
        base_lon = np.random.uniform(130, 160)
        
        # Generate 20-50 data points per storm
        track_length = np.random.randint(20, 51)
        sid = f"WP{i+1:02d}{date.year}"
        
        for j in range(track_length):
            lat = base_lat + j * 0.2 + np.random.normal(0, 0.1)
            lon = base_lon + j * 0.3 + np.random.normal(0, 0.1)
            wind = max(25, 70 + np.random.normal(0, 20))
            pres = max(950, 1000 - wind + np.random.normal(0, 5))
            
            data.append({
                'SID': sid,
                'ISO_TIME': date + pd.Timedelta(hours=j*6),  # Use pd.Timedelta instead
                'NAME': f'FALLBACK_{i+1}',
                'SEASON': date.year,
                'LAT': lat,
                'LON': lon,
                'USA_WIND': wind,
                'USA_PRES': pres,
                'BASIN': 'WP'
            })
    
    df = pd.DataFrame(data)
    logging.info(f"Created fallback typhoon data with {len(df)} records")
    return df

def process_oni_data(oni_data):
    """Process ONI data into long format"""
    oni_long = oni_data.melt(id_vars=['Year'], var_name='Month', value_name='ONI')
    month_map = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                 'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'}
    oni_long['Month'] = oni_long['Month'].map(month_map)
    oni_long['Date'] = pd.to_datetime(oni_long['Year'].astype(str)+'-'+oni_long['Month']+'-01')
    oni_long['ONI'] = pd.to_numeric(oni_long['ONI'], errors='coerce')
    return oni_long

def process_typhoon_data(typhoon_data):
    """Process typhoon data"""
    if 'ISO_TIME' in typhoon_data.columns:
        typhoon_data['ISO_TIME'] = pd.to_datetime(typhoon_data['ISO_TIME'], errors='coerce')
    typhoon_data['USA_WIND'] = pd.to_numeric(typhoon_data['USA_WIND'], errors='coerce')
    typhoon_data['USA_PRES'] = pd.to_numeric(typhoon_data['USA_PRES'], errors='coerce')
    typhoon_data['LON'] = pd.to_numeric(typhoon_data['LON'], errors='coerce')
    
    logging.info(f"Unique basins in typhoon_data: {typhoon_data['SID'].str[:2].unique()}")
    
    typhoon_max = typhoon_data.groupby('SID').agg({
        'USA_WIND':'max','USA_PRES':'min','ISO_TIME':'first','SEASON':'first','NAME':'first',
        'LAT':'first','LON':'first'
    }).reset_index()
    
    if 'ISO_TIME' in typhoon_max.columns:
        typhoon_max['Month'] = typhoon_max['ISO_TIME'].dt.strftime('%m')
        typhoon_max['Year'] = typhoon_max['ISO_TIME'].dt.year
    else:
        # Fallback if no ISO_TIME
        typhoon_max['Month'] = '01'
        typhoon_max['Year'] = typhoon_max['SEASON']
    
    typhoon_max['Category'] = typhoon_max['USA_WIND'].apply(categorize_typhoon_enhanced)
    return typhoon_max

def merge_data(oni_long, typhoon_max):
    """Merge ONI and typhoon data"""
    return pd.merge(typhoon_max, oni_long, on=['Year','Month'])

# -----------------------------
# ENHANCED: Categorization Functions (FIXED TAIWAN)
# -----------------------------

def categorize_typhoon_enhanced(wind_speed):
    """Enhanced categorization that properly includes Tropical Depressions"""
    if pd.isna(wind_speed):
        return 'Unknown'
    
    # Convert to knots if in m/s (some datasets use m/s)
    if wind_speed < 10:  # Likely in m/s, convert to knots
        wind_speed = wind_speed * 1.94384
    
    # FIXED thresholds to include TD
    if wind_speed < 34:  # Below 34 knots = Tropical Depression
        return 'Tropical Depression'
    elif wind_speed < 64:  # 34-63 knots = Tropical Storm
        return 'Tropical Storm'
    elif wind_speed < 83:  # 64-82 knots = Category 1 Typhoon
        return 'C1 Typhoon'
    elif wind_speed < 96:  # 83-95 knots = Category 2 Typhoon
        return 'C2 Typhoon'
    elif wind_speed < 113:  # 96-112 knots = Category 3 Strong Typhoon
        return 'C3 Strong Typhoon'
    elif wind_speed < 137:  # 113-136 knots = Category 4 Very Strong Typhoon
        return 'C4 Very Strong Typhoon'
    else:  # 137+ knots = Category 5 Super Typhoon
        return 'C5 Super Typhoon'

def categorize_typhoon_taiwan(wind_speed):
    """FIXED Taiwan categorization system according to official CWA standards"""
    if pd.isna(wind_speed):
        return 'Tropical Depression'
    
    # Convert from knots to m/s (official CWA uses m/s thresholds)
    if wind_speed > 200:  # Likely already in m/s
        wind_speed_ms = wind_speed
    else:  # Likely in knots, convert to m/s
        wind_speed_ms = wind_speed * 0.514444
    
    # Official CWA Taiwan classification thresholds
    if wind_speed_ms >= 51.0:              # 100+ knots
        return 'Intense Typhoon'
    elif wind_speed_ms >= 32.7:           # 64-99 knots
        return 'Moderate Typhoon'
    elif wind_speed_ms >= 17.2:           # 34-63 knots
        return 'Tropical Storm'
    else:                                  # <34 knots
        return 'Tropical Depression'

# Original function for backward compatibility
def categorize_typhoon(wind_speed):
    """Original categorize typhoon function for backward compatibility"""
    return categorize_typhoon_enhanced(wind_speed)

def classify_enso_phases(oni_value):
    """Classify ENSO phases based on ONI value"""
    if isinstance(oni_value, pd.Series):
        oni_value = oni_value.iloc[0]
    if pd.isna(oni_value):
        return 'Neutral'
    if oni_value >= 0.5:
        return 'El Nino'
    elif oni_value <= -0.5:
        return 'La Nina'
    else:
        return 'Neutral'

def categorize_typhoon_by_standard(wind_speed, standard='atlantic'):
    """FIXED categorization function with correct Taiwan standards"""
    if pd.isna(wind_speed):
        return 'Tropical Depression', '#808080'
    
    if standard == 'taiwan':
        category = categorize_typhoon_taiwan(wind_speed)
        color = taiwan_color_map.get(category, '#808080')
        return category, color
    else:
        # Atlantic/International standard (existing logic is correct)
        if wind_speed >= 137:
            return 'C5 Super Typhoon', '#FF0000'      # Red
        elif wind_speed >= 113:
            return 'C4 Very Strong Typhoon', '#FFA500' # Orange
        elif wind_speed >= 96:
            return 'C3 Strong Typhoon', '#FFFF00'     # Yellow
        elif wind_speed >= 83:
            return 'C2 Typhoon', '#00FF00'            # Green
        elif wind_speed >= 64:
            return 'C1 Typhoon', '#00FFFF'            # Cyan
        elif wind_speed >= 34:
            return 'Tropical Storm', '#0000FF'        # Blue
        return 'Tropical Depression', '#808080'       # Gray

# -----------------------------
# FIXED: Genesis Potential Index (GPI) Based Prediction System
# -----------------------------

def calculate_genesis_potential_index(sst, rh, vorticity, wind_shear, lat, lon, month, oni_value):
    """
    Calculate Genesis Potential Index based on environmental parameters
    Following Emanuel and Nolan (2004) formulation with modifications for monthly predictions
    """
    try:
        # Base environmental parameters
        
        # SST factor - optimal range 26-30°C
        sst_factor = max(0, (sst - 26.5) / 4.0) if sst > 26.5 else 0
        
        # Humidity factor - mid-level relative humidity (600 hPa)
        rh_factor = max(0, (rh - 40) / 50.0)  # Normalized 40-90%
        
        # Vorticity factor - low-level absolute vorticity (850 hPa)
        vort_factor = max(0, min(vorticity / 5e-5, 3.0))  # Cap at reasonable max
        
        # Wind shear factor - vertical wind shear (inverse relationship)
        shear_factor = max(0, (20 - wind_shear) / 15.0) if wind_shear < 20 else 0
        
        # Coriolis factor - latitude dependency
        coriolis_factor = max(0, min(abs(lat) / 20.0, 1.0)) if abs(lat) > 5 else 0
        
        # Seasonal factor
        seasonal_weights = {
            1: 0.3, 2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1.0,
            7: 1.2, 8: 1.4, 9: 1.5, 10: 1.3, 11: 0.9, 12: 0.5
        }
        seasonal_factor = seasonal_weights.get(month, 1.0)
        
        # ENSO modulation
        if oni_value > 0.5:  # El Niño
            enso_factor = 0.6 if lon > 140 else 0.8  # Suppress in WP
        elif oni_value < -0.5:  # La Niña
            enso_factor = 1.4 if lon > 140 else 1.1  # Enhance in WP
        else:  # Neutral
            enso_factor = 1.0
        
        # Regional modulation (Western Pacific specifics)
        if 10 <= lat <= 25 and 120 <= lon <= 160:  # Main Development Region
            regional_factor = 1.3
        elif 5 <= lat <= 15 and 130 <= lon <= 150:  # Prime genesis zone
            regional_factor = 1.5
        else:
            regional_factor = 0.8
        
        # Calculate GPI
        gpi = (sst_factor * rh_factor * vort_factor * shear_factor * 
               coriolis_factor * seasonal_factor * enso_factor * regional_factor)
        
        return max(0, min(gpi, 5.0))  # Cap at reasonable maximum
        
    except Exception as e:
        logging.error(f"Error calculating GPI: {e}")
        return 0.0

def get_environmental_conditions(lat, lon, month, oni_value):
    """
    Get realistic environmental conditions for a given location and time
    Based on climatological patterns and ENSO modulation
    """
    try:
        # Base SST calculation (climatological)
        base_sst = 28.5 - 0.15 * abs(lat - 15)  # Peak at 15°N
        seasonal_sst_adj = 2.0 * np.cos(2 * np.pi * (month - 9) / 12)  # Peak in Sep
        enso_sst_adj = oni_value * 0.8 if lon > 140 else oni_value * 0.4
        sst = base_sst + seasonal_sst_adj + enso_sst_adj
        
        # Relative humidity (600 hPa)
        base_rh = 75 - 0.5 * abs(lat - 12)  # Peak around 12°N
        seasonal_rh_adj = 10 * np.cos(2 * np.pi * (month - 8) / 12)  # Peak in Aug
        monsoon_effect = 5 if 100 <= lon <= 120 and month in [6,7,8,9] else 0
        rh = max(40, min(90, base_rh + seasonal_rh_adj + monsoon_effect))
        
        # Low-level vorticity (850 hPa)
        base_vort = 2e-5 * (1 + 0.1 * np.sin(2 * np.pi * lat / 30))
        seasonal_vort_adj = 1e-5 * np.cos(2 * np.pi * (month - 8) / 12)
        itcz_effect = 1.5e-5 if 5 <= lat <= 15 else 0
        vorticity = max(0, base_vort + seasonal_vort_adj + itcz_effect)
        
        # Vertical wind shear (200-850 hPa)
        base_shear = 8 + 0.3 * abs(lat - 20)  # Lower near 20°N
        seasonal_shear_adj = 4 * np.cos(2 * np.pi * (month - 2) / 12)  # Low in Aug-Sep
        enso_shear_adj = oni_value * 3 if lon > 140 else 0  # El Niño increases shear
        wind_shear = max(2, base_shear + seasonal_shear_adj + enso_shear_adj)
        
        return {
            'sst': sst,
            'relative_humidity': rh,
            'vorticity': vorticity,
            'wind_shear': wind_shear
        }
        
    except Exception as e:
        logging.error(f"Error getting environmental conditions: {e}")
        return {
            'sst': 28.0,
            'relative_humidity': 70.0,
            'vorticity': 2e-5,
            'wind_shear': 10.0
        }

def generate_genesis_prediction_monthly(month, oni_value, year=2025):
    """
    Generate realistic typhoon genesis prediction for a given month using GPI
    Returns day-by-day genesis potential and storm development scenarios
    """
    try:
        logging.info(f"Generating GPI-based prediction for month {month}, ONI {oni_value}")
        
        # Define the Western Pacific domain
        lat_range = np.arange(5, 35, 2.5)  # 5°N to 35°N
        lon_range = np.arange(110, 180, 2.5)  # 110°E to 180°E
        
        # Number of days in the month
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
        if month == 2 and year % 4 == 0:  # Leap year
            days_in_month = 29
        
        # Daily GPI evolution
        daily_gpi_maps = []
        genesis_events = []
        
        for day in range(1, days_in_month + 1):
            # Calculate GPI for each grid point
            gpi_field = np.zeros((len(lat_range), len(lon_range)))
            
            for i, lat in enumerate(lat_range):
                for j, lon in enumerate(lon_range):
                    # Get environmental conditions
                    env_conditions = get_environmental_conditions(lat, lon, month, oni_value)
                    
                    # Add daily variability
                    daily_variation = 0.1 * np.sin(2 * np.pi * day / 30) + np.random.normal(0, 0.05)
                    
                    # Calculate GPI
                    gpi = calculate_genesis_potential_index(
                        sst=env_conditions['sst'] + daily_variation,
                        rh=env_conditions['relative_humidity'],
                        vorticity=env_conditions['vorticity'],
                        wind_shear=env_conditions['wind_shear'],
                        lat=lat,
                        lon=lon,
                        month=month,
                        oni_value=oni_value
                    )
                    
                    gpi_field[i, j] = gpi
            
            daily_gpi_maps.append({
                'day': day,
                'gpi_field': gpi_field,
                'lat_range': lat_range,
                'lon_range': lon_range
            })
            
            # Check for genesis events (GPI > threshold)
            genesis_threshold = 1.5  # Adjusted threshold
            if np.max(gpi_field) > genesis_threshold:
                # Find peak genesis locations
                peak_indices = np.where(gpi_field > genesis_threshold)
                if len(peak_indices[0]) > 0:
                    # Select strongest genesis point
                    max_idx = np.argmax(gpi_field)
                    max_i, max_j = np.unravel_index(max_idx, gpi_field.shape)
                    
                    genesis_lat = lat_range[max_i]
                    genesis_lon = lon_range[max_j]
                    genesis_gpi = gpi_field[max_i, max_j]
                    
                    # Determine probability of actual genesis
                    genesis_prob = min(0.8, genesis_gpi / 3.0)
                    
                    if np.random.random() < genesis_prob:
                        genesis_events.append({
                            'day': day,
                            'lat': genesis_lat,
                            'lon': genesis_lon,
                            'gpi': genesis_gpi,
                            'probability': genesis_prob,
                            'date': f"{year}-{month:02d}-{day:02d}"
                        })
        
        # Generate storm tracks for genesis events
        storm_predictions = []
        for i, genesis in enumerate(genesis_events):
            storm_track = generate_storm_track_from_genesis(
                genesis['lat'],
                genesis['lon'],
                genesis['day'],
                month,
                oni_value,
                storm_id=i+1
            )
            
            if storm_track:
                storm_predictions.append({
                    'storm_id': i + 1,
                    'genesis_event': genesis,
                    'track': storm_track,
                    'uncertainty': calculate_track_uncertainty(storm_track)
                })

        logging.info(
            f"Monthly genesis prediction: {len(genesis_events)} events, {len(storm_predictions)} tracks"
        )
        
        return {
            'month': month,
            'year': year,
            'oni_value': oni_value,
            'daily_gpi_maps': daily_gpi_maps,
            'genesis_events': genesis_events,
            'storm_predictions': storm_predictions,
            'summary': {
                'total_genesis_events': len(genesis_events),
                'total_storm_predictions': len(storm_predictions),
                'peak_gpi_day': max(daily_gpi_maps, key=lambda x: np.max(x['gpi_field']))['day'],
                'peak_gpi_value': max(np.max(day_data['gpi_field']) for day_data in daily_gpi_maps)
            }
        }
        
    except Exception as e:
        logging.error(f"Error in genesis prediction: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'month': month,
            'oni_value': oni_value,
            'storm_predictions': []
        }

def generate_storm_track_from_genesis(genesis_lat, genesis_lon, genesis_day, month, oni_value, storm_id=1):
    """
    Generate a realistic storm track from a genesis location
    """
    try:
        track_points = []
        current_lat = genesis_lat
        current_lon = genesis_lon
        current_intensity = 25  # Start as TD
        
        # Track duration (3-10 days typically)
        track_duration_hours = np.random.randint(72, 240)
        
        for hour in range(0, track_duration_hours + 6, 6):
            # Calculate storm motion
            # Base motion patterns for Western Pacific
            if current_lat < 20:  # Low latitude - westward motion
                lat_speed = 0.1 + np.random.normal(0, 0.05)  # Slight poleward
                lon_speed = -0.3 + np.random.normal(0, 0.1)  # Westward
            elif current_lat < 25:  # Mid latitude - WNW motion
                lat_speed = 0.15 + np.random.normal(0, 0.05)
                lon_speed = -0.2 + np.random.normal(0, 0.1)
            else:  # High latitude - recurvature
                lat_speed = 0.2 + np.random.normal(0, 0.05)
                lon_speed = 0.1 + np.random.normal(0, 0.1)  # Eastward
            
            # ENSO effects on motion
            if oni_value > 0.5:  # El Niño - more eastward
                lon_speed += 0.05
            elif oni_value < -0.5:  # La Niña - more westward
                lon_speed -= 0.05
            
            # Update position
            current_lat += lat_speed
            current_lon += lon_speed
            
            # Intensity evolution
            # Get environmental conditions for intensity change
            env_conditions = get_environmental_conditions(current_lat, current_lon, month, oni_value)
            
            # Intensity change factors
            sst_factor = max(0, env_conditions['sst'] - 26.5)
            shear_factor = max(0, (15 - env_conditions['wind_shear']) / 10)
            
            # Basic intensity change
            if hour < 48:  # Development phase
                intensity_change = 3 + sst_factor + shear_factor + np.random.normal(0, 2)
            elif hour < 120:  # Mature phase
                intensity_change = 1 + sst_factor * 0.5 + np.random.normal(0, 1.5)
            else:  # Weakening phase
                intensity_change = -2 + sst_factor * 0.3 + np.random.normal(0, 1)
            
            # Environmental limits
            if current_lat > 30:  # Cool waters
                intensity_change -= 5
            if current_lon < 120:  # Land interaction
                intensity_change -= 8
            
            current_intensity += intensity_change
            current_intensity = max(15, min(180, current_intensity))  # Realistic bounds
            
            # Calculate pressure
            pressure = max(900, 1013 - (current_intensity - 25) * 0.9)
            
            # Add uncertainty
            position_uncertainty = 0.5 + (hour / 120) * 1.5  # Growing uncertainty
            intensity_uncertainty = 5 + (hour / 120) * 15
            
            track_points.append({
                'hour': hour,
                'day': genesis_day + hour / 24.0,
                'lat': current_lat,
                'lon': current_lon,
                'intensity': current_intensity,
                'pressure': pressure,
                'category': categorize_typhoon_enhanced(current_intensity),
                'position_uncertainty': position_uncertainty,
                'intensity_uncertainty': intensity_uncertainty
            })
            
            # Stop if storm moves too far or weakens significantly
            if current_lat > 40 or current_lat < 0 or current_lon < 100 or current_intensity < 20:
                break
        
        return track_points
        
    except Exception as e:
        logging.error(f"Error generating storm track: {e}")
        return None

def calculate_track_uncertainty(track_points):
    """Calculate uncertainty metrics for a storm track"""
    if not track_points:
        return {'position': 0, 'intensity': 0}
    
    # Position uncertainty grows with time
    position_uncertainty = [point['position_uncertainty'] for point in track_points]
    
    # Intensity uncertainty
    intensity_uncertainty = [point['intensity_uncertainty'] for point in track_points]
    
    return {
        'position_mean': np.mean(position_uncertainty),
        'position_max': np.max(position_uncertainty),
        'intensity_mean': np.mean(intensity_uncertainty),
        'intensity_max': np.max(intensity_uncertainty),
        'track_length': len(track_points)
    }
def create_predict_animation(prediction_data, enable_animation=True):
    """
    Typhoon genesis PREDICT tab animation:
    shows monthly genesis-potential + progressive storm positions
    """
    try:
        daily_maps = prediction_data.get('daily_gpi_maps', [])
        if not daily_maps:
            return create_error_plot("No GPI data for prediction")

        storms   = prediction_data.get('storm_predictions', [])
        month    = prediction_data['month']
        oni      = prediction_data['oni_value']
        year     = prediction_data.get('year', 2025)

        # -- 1) static underlay: full storm routes (dashed gray lines)
        static_routes = []
        for s in storms:
            track = s.get('track', [])
            if not track: continue
            lats = [pt['lat'] for pt in track]
            lons = [pt['lon'] for pt in track]
            static_routes.append(
                go.Scattergeo(
                    lat=lats, lon=lons,
                    mode='lines',
                    line=dict(width=2, dash='dash', color='gray'),
                    showlegend=False, hoverinfo='skip'
                )
            )

        # figure out map bounds
        all_lats = [pt['lat'] for s in storms for pt in s.get('track',[])]
        all_lons = [pt['lon'] for s in storms for pt in s.get('track',[])]
        mb = {
            'lat_min': min(5,  min(all_lats)-5) if all_lats else 5,
            'lat_max': max(35, max(all_lats)+5) if all_lats else 35,
            'lon_min': min(110, min(all_lons)-10) if all_lons else 110,
            'lon_max': max(180, max(all_lons)+10) if all_lons else 180
        }

        # -- 2) build frames
        frames = []
        for idx, day_data in enumerate(daily_maps):
            day = day_data['day']
            gpi = day_data['gpi_field']
            lats = day_data['lat_range']
            lons = day_data['lon_range']

            traces = []
            # genesis‐potential scatter
            traces.append(go.Scattergeo(
                lat=np.repeat(lats, len(lons)),
                lon=np.tile(lons, len(lats)),
                mode='markers',
                marker=dict(
                    size=6, color=gpi.flatten(),
                    colorscale='Viridis', cmin=0, cmax=3, opacity=0.6,
                    showscale=(idx==0),
                    colorbar=(dict(
                        title=dict(text="Genesis<br>Potential<br>Index", side="right")
                    ) if idx==0 else None)
                ),
                name='GPI',
                showlegend=(idx==0),
                hovertemplate=(
                    'GPI: %{marker.color:.2f}<br>'
                    'Lat: %{lat:.1f}°N<br>'
                    'Lon: %{lon:.1f}°E<br>'
                    f'Day {day} of {month:02d}/{year}<extra></extra>'
                )
            ))

            # storm positions up to this day
            for s in storms:
                past = [pt for pt in s.get('track',[]) if pt['day'] <= day]
                if not past: continue
                lats_p = [pt['lat'] for pt in past]
                lons_p = [pt['lon'] for pt in past]
                intens = [pt['intensity'] for pt in past]
                cats   = [pt['category']  for pt in past]

                # line history
                traces.append(go.Scattergeo(
                    lat=lats_p, lon=lons_p, mode='lines',
                    line=dict(width=2, color='gray'),
                    showlegend=(idx==0), hoverinfo='skip'
                ))
                # current position
                traces.append(go.Scattergeo(
                    lat=[lats_p[-1]], lon=[lons_p[-1]],
                    mode='markers',
                    marker=dict(size=10, symbol='circle', color='red'),
                    showlegend=(idx==0),
                    hovertemplate=(
                        f"{s['storm_id']}<br>"
                        f"Intensity: {intens[-1]} kt<br>"
                        f"Category: {cats[-1]}<extra></extra>"
                    )
                ))

            frames.append(go.Frame(
                data=traces,
                name=str(day),                       # ← name is REQUIRED as string :contentReference[oaicite:1]{index=1}
                layout=go.Layout(
                    geo=dict(
                        projection_type="natural earth",
                        showland=True, landcolor="lightgray",
                        showocean=True, oceancolor="lightblue",
                        showcoastlines=True, coastlinecolor="darkgray",
                        center=dict(lat=(mb['lat_min']+mb['lat_max'])/2,
                                    lon=(mb['lon_min']+mb['lon_max'])/2),
                        lonaxis_range=[mb['lon_min'], mb['lon_max']],
                        lataxis_range=[mb['lat_min'], mb['lat_max']],
                        resolution=50
                    ),
                    title=f"Day {day} of {month:02d}/{year} | ONI: {oni:.2f}"
                )
            ))

        # -- 3) initial Figure (static + first frame)
        init_data = static_routes + list(frames[0].data)
        fig = go.Figure(data=init_data, frames=frames)

        # -- 4) play/pause + slider (redraw=True!)
        if enable_animation and len(frames)>1:
            steps = [
                dict(method="animate",
                     args=[[fr.name],
                           {"mode":"immediate",
                            "frame":{"duration":600,"redraw":True},
                            "transition":{"duration":0}}],
                     label=fr.name)
                for fr in frames
            ]

            fig.update_layout(
                updatemenus=[dict(
                    type="buttons", showactive=False,
                    x=1.05, y=0.05, xanchor="right", yanchor="bottom",
                    buttons=[
                        dict(label="▶ Play",
                             method="animate",
                             args=[None,   # None＝all frames
                                   {"frame":{"duration":600,"redraw":True},  # ← redraw fixes dead ▶
                                    "fromcurrent":True,"transition":{"duration":0}}]),
                        dict(label="⏸ Pause",
                             method="animate",
                             args=[[None],
                                   {"frame":{"duration":0,"redraw":False},
                                    "mode":"immediate"}])
                    ]
                )],
                sliders=[dict(active=0, pad=dict(t=50), steps=steps)]
            )
        else:
            # fallback: show only final day + static routes
            final = static_routes + list(frames[-1].data)
            fig = go.Figure(data=final)

        # -- 5) shared layout styling
        fig.update_layout(
            title={
                'text': f"🌊 Typhoon Prediction — {month:02d}/{year} | ONI: {oni:.2f}",
                'x':0.5,'font':{'size':18}
            },
            geo=dict(
                projection_type="natural earth",
                showland=True, landcolor="lightgray",
                showocean=True, oceancolor="lightblue",
                showcoastlines=True, coastlinecolor="darkgray",
                showlakes=True, lakecolor="lightblue",
                showcountries=True, countrycolor="gray",
                resolution=50,
                center=dict(lat=20, lon=140),
                lonaxis_range=[110,180], lataxis_range=[5,35]
            ),
            width=1100, height=750,
            showlegend=True,
            legend=dict(
                x=0.02,y=0.98,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="gray",borderwidth=1
            )
        )

        return fig

    except Exception as e:
        logging.error(f"Error in predict animation: {e}")
        import traceback; traceback.print_exc()
        return create_error_plot(f"Animation error: {e}")
def create_genesis_animation(prediction_data, enable_animation=True):
    """
    Create professional typhoon track animation showing daily genesis potential and storm development
    Following NHC/JTWC visualization standards with proper geographic map and time controls
    """
    try:
        daily_maps = prediction_data.get('daily_gpi_maps', [])
        if not daily_maps:
            return create_error_plot("No GPI data available for animation")
        
        storm_predictions = prediction_data.get('storm_predictions', [])
        month = prediction_data['month']
        oni_value = prediction_data['oni_value']
        year = prediction_data.get('year', 2025)

        # ---- 1) Prepare static full-track routes ----
        static_routes = []
        for storm in storm_predictions:
            track = storm.get('track', [])
            if not track:
                continue
            lats = [pt['lat'] for pt in track]
            lons = [pt['lon'] for pt in track]
            static_routes.append(
                go.Scattergeo(
                    lat=lats,
                    lon=lons,
                    mode='lines',
                    line=dict(width=2, dash='dash', color='gray'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )

        # ---- 2) Build animation frames ----
        frames = []
        # determine map bounds from all storm tracks
        all_lats = [pt['lat'] for storm in storm_predictions for pt in storm.get('track', [])]
        all_lons = [pt['lon'] for storm in storm_predictions for pt in storm.get('track', [])]
        map_bounds = {
            'lat_min': min(5, min(all_lats) - 5) if all_lats else 5,
            'lat_max': max(35, max(all_lats) + 5) if all_lats else 35,
            'lon_min': min(110, min(all_lons) - 10) if all_lons else 110,
            'lon_max': max(180, max(all_lons) + 10) if all_lons else 180
        }

        for day_idx, day_data in enumerate(daily_maps):
            day = day_data['day']
            gpi = day_data['gpi_field']
            lats = day_data['lat_range']
            lons = day_data['lon_range']

            traces = []
            # Genesis potential dots
            traces.append(go.Scattergeo(
                lat=np.repeat(lats, len(lons)),
                lon=np.tile(lons, len(lats)),
                mode='markers',
                marker=dict(
                    size=6,
                    color=gpi.flatten(),
                    colorscale='Viridis',
                    cmin=0, cmax=3, opacity=0.6,
                    showscale=(day_idx == 0),
                    colorbar=(dict(
                        title=dict(text="Genesis<br>Potential<br>Index", side="right")
                    ) if day_idx == 0 else None)
                ),
                name='Genesis Potential',
                showlegend=(day_idx == 0),
                hovertemplate=(
                    'GPI: %{marker.color:.2f}<br>' +
                    'Lat: %{lat:.1f}°N<br>' +
                    'Lon: %{lon:.1f}°E<br>' +
                    f'Day {day} of {month:02d}/{year}<extra></extra>'
                )
            ))

            # Storm positions up to this day
            for storm in storm_predictions:
                past = [pt for pt in storm.get('track', []) if pt['day'] <= day]
                if not past:
                    continue
                lats_p = [pt['lat'] for pt in past]
                lons_p = [pt['lon'] for pt in past]
                intens = [pt['intensity'] for pt in past]
                cats = [pt['category'] for pt in past]

                # historical line
                traces.append(go.Scattergeo(
                    lat=lats_p, lon=lons_p, mode='lines',
                    line=dict(width=2, color='gray'),
                    name=f"{storm['storm_id']} Track",
                    showlegend=(day_idx == 0),
                    hoverinfo='skip'
                ))
                # current position
                traces.append(go.Scattergeo(
                    lat=[lats_p[-1]], lon=[lons_p[-1]], mode='markers',
                    marker=dict(size=10, symbol='circle', color='red'),
                    name=f"{storm['storm_id']} Position",
                    showlegend=(day_idx == 0),
                    hovertemplate=(
                        f"{storm['storm_id']}<br>"
                        f"Intensity: {intens[-1]} kt<br>"
                        f"Category: {cats[-1]}<extra></extra>"
                    )
                ))

            frames.append(go.Frame(
                data=traces,
                name=str(day),
                layout=go.Layout(
                    geo=dict(
                        projection_type="natural earth",
                        showland=True, landcolor="lightgray",
                        showocean=True, oceancolor="lightblue",
                        showcoastlines=True, coastlinecolor="darkgray",
                        center=dict(
                            lat=(map_bounds['lat_min'] + map_bounds['lat_max'])/2,
                            lon=(map_bounds['lon_min'] + map_bounds['lon_max'])/2
                        ),
                        lonaxis_range=[map_bounds['lon_min'], map_bounds['lon_max']],
                        lataxis_range=[map_bounds['lat_min'], map_bounds['lat_max']],
                        resolution=50
                    ),
                    title=f"Day {day} of {month:02d}/{year}   ONI: {oni_value:.2f}"
                )
            ))

        # ---- 3) Initialize figure with static routes + first frame ----
        initial_data = static_routes + list(frames[0].data)
        fig = go.Figure(data=initial_data, frames=frames)

        # ---- 4) Add play/pause buttons with redraw=True ----
        if enable_animation and len(frames) > 1:
            # slider steps
            steps = [
                dict(method="animate",
                     args=[[fr.name],
                           {"mode": "immediate",
                            "frame": {"duration": 600, "redraw": True},
                            "transition": {"duration": 0}}],
                     label=fr.name)
                for fr in frames
            ]

            fig.update_layout(
                updatemenus=[dict(
                    type="buttons", showactive=False,
                    x=1.05, y=0.05, xanchor="right", yanchor="bottom",
                    buttons=[
                        dict(label="▶ Play",
                             method="animate",
                             args=[None,  # None means “all frames”
                                   {"frame": {"duration": 600, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0}}
                                  ]),  # redraw=True fixes the dead play button :contentReference[oaicite:1]{index=1}
                        dict(label="⏸ Pause",
                             method="animate",
                             args=[[None],
                                   {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate"}])
                    ]
                )],
                sliders=[dict(active=0, pad=dict(t=50), steps=steps)]
            )
        # No-animation fallback: just show final day + routes
        else:
            final = static_routes + list(frames[-1].data)
            fig = go.Figure(data=final)

        # ---- 5) Common layout styling ----
        fig.update_layout(
            title={
                'text': f"🌊 Typhoon Genesis & Development Forecast<br>"
                        f"<sub>{month:02d}/{year} | ONI: {oni_value:.2f}</sub>",
                'x': 0.5, 'font': {'size': 18}
            },
            geo=dict(
                projection_type="natural earth",
                showland=True, landcolor="lightgray",
                showocean=True, oceancolor="lightblue",
                showcoastlines=True, coastlinecolor="darkgray",
                showlakes=True, lakecolor="lightblue",
                showcountries=True, countrycolor="gray",
                resolution=50,
                center=dict(lat=20, lon=140),
                lonaxis_range=[110, 180], lataxis_range=[5, 35]
            ),
            width=1100, height=750,
            showlegend=True,
            legend=dict(x=0.02, y=0.98,
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="gray", borderwidth=1)
        )

        return fig

    except Exception as e:
        logging.error(f"Error creating professional genesis animation: {e}")
        import traceback; traceback.print_exc()
        return create_error_plot(f"Animation error: {e}")


def create_error_plot(error_message):
    """Create a simple error plot"""
    fig = go.Figure()
    fig.add_annotation(
        text=error_message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False, font_size=16
    )
    fig.update_layout(title="Error in Visualization")
    return fig

def create_prediction_summary(prediction_data):
    """Create a comprehensive summary of the prediction"""
    try:
        if 'error' in prediction_data:
            return f"Error in prediction: {prediction_data['error']}"
        
        month = prediction_data['month']
        oni_value = prediction_data['oni_value']
        summary = prediction_data.get('summary', {})
        genesis_events = prediction_data.get('genesis_events', [])
        storm_predictions = prediction_data.get('storm_predictions', [])
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_name = month_names[month - 1]
        
        summary_text = f"""
TYPHOON GENESIS PREDICTION SUMMARY - {month_name.upper()} 2025
{'='*70}

🌊 ENVIRONMENTAL CONDITIONS:
• Month: {month_name} (Month {month})
• ONI Value: {oni_value:.2f} {'(El Niño)' if oni_value > 0.5 else '(La Niña)' if oni_value < -0.5 else '(Neutral)'}
• Season Phase: {'Peak Season' if month in [7,8,9,10] else 'Off Season' if month in [1,2,3,4,11,12] else 'Transition Season'}

📊 GENESIS POTENTIAL ANALYSIS:
• Peak GPI Day: Day {summary.get('peak_gpi_day', 'Unknown')}
• Peak GPI Value: {summary.get('peak_gpi_value', 0):.2f}
• Total Genesis Events: {summary.get('total_genesis_events', 0)}
• Storm Development Success: {summary.get('total_storm_predictions', 0)}/{summary.get('total_genesis_events', 0)} events

🎯 GENESIS EVENTS BREAKDOWN:
"""
        
        if genesis_events:
            for i, event in enumerate(genesis_events, 1):
                summary_text += f"""
Event {i}:
• Date: {event['date']}
• Location: {event['lat']:.1f}°N, {event['lon']:.1f}°E
• GPI Value: {event['gpi']:.2f}
• Genesis Probability: {event['probability']*100:.0f}%
"""
        else:
            summary_text += "\n• No significant genesis events predicted for this month\n"
        
        summary_text += f"""

🌪️ STORM TRACK PREDICTIONS:
"""
        
        if storm_predictions:
            for storm in storm_predictions:
                track = storm['track']
                if track:
                    genesis = storm['genesis_event']
                    max_intensity = max(pt['intensity'] for pt in track)
                    max_category = categorize_typhoon_enhanced(max_intensity)
                    track_duration = len(track) * 6  # hours
                    final_pos = track[-1]
                    
                    summary_text += f"""
Storm {storm['storm_id']}:
• Genesis: Day {genesis['day']}, {genesis['lat']:.1f}°N {genesis['lon']:.1f}°E
• Peak Intensity: {max_intensity:.0f} kt ({max_category})
• Track Duration: {track_duration} hours ({track_duration/24:.1f} days)
• Final Position: {final_pos['lat']:.1f}°N, {final_pos['lon']:.1f}°E
• Uncertainty: ±{storm['uncertainty']['position_mean']:.1f}° position, ±{storm['uncertainty']['intensity_mean']:.0f} kt intensity
"""
        else:
            summary_text += "\n• No storm development predicted from genesis events\n"
        
        # Add climatological context
        summary_text += f"""

📈 CLIMATOLOGICAL CONTEXT:
• {month_name} Typical Activity: {'Very High' if month in [8,9] else 'High' if month in [7,10] else 'Moderate' if month in [6,11] else 'Low'}
• ENSO Influence: {'Strong suppression expected' if oni_value > 1.0 else 'Moderate suppression' if oni_value > 0.5 else 'Strong enhancement likely' if oni_value < -1.0 else 'Moderate enhancement' if oni_value < -0.5 else 'Near-normal activity'}
• Regional Focus: Western Pacific Main Development Region (10-25°N, 120-160°E)

🔧 METHODOLOGY DETAILS:
• Genesis Potential Index: Emanuel & Nolan (2004) formulation
• Environmental Factors: SST, humidity, vorticity, wind shear, Coriolis effect
• Temporal Resolution: Daily evolution throughout month
• Spatial Resolution: 2.5° grid spacing
• ENSO Modulation: Integrated ONI effects on environmental parameters
• Track Prediction: Physics-based storm motion and intensity evolution

⚠️  UNCERTAINTY & LIMITATIONS:
• Genesis timing: ±2-3 days typical uncertainty
• Track position: Growing uncertainty with time (±0.5° to ±2°)
• Intensity prediction: ±5-15 kt uncertainty range
• Environmental assumptions: Based on climatological patterns
• Model limitations: Simplified compared to operational NWP systems

🎯 FORECAST CONFIDENCE:
• Genesis Location: {'High' if summary.get('peak_gpi_value', 0) > 2 else 'Moderate' if summary.get('peak_gpi_value', 0) > 1 else 'Low'}
• Genesis Timing: {'High' if month in [7,8,9] else 'Moderate' if month in [6,10] else 'Low'}
• Track Prediction: Moderate (physics-based motion patterns)
• Intensity Evolution: Moderate (environmental constraints applied)

📋 OPERATIONAL IMPLICATIONS:
• Monitor Days {', '.join([str(event['day']) for event in genesis_events[:3]])} for potential development
• Focus regions: {', '.join([f"{event['lat']:.0f}°N {event['lon']:.0f}°E" for event in genesis_events[:3]])}
• Preparedness level: {'High' if len(storm_predictions) > 2 else 'Moderate' if len(storm_predictions) > 0 else 'Routine'}

🔬 RESEARCH APPLICATIONS:
• Suitable for seasonal planning and climate studies
• Genesis mechanism investigation
• ENSO-typhoon relationship analysis
• Environmental parameter sensitivity studies

⚠️  IMPORTANT DISCLAIMERS:
• This is a research prediction system, not operational forecast
• Use official meteorological services for real-time warnings
• Actual conditions may differ from climatological assumptions
• Model simplified compared to operational prediction systems
• Uncertainty grows significantly beyond 5-7 day lead times
"""
        
        return summary_text
        
    except Exception as e:
        logging.error(f"Error creating prediction summary: {e}")
        return f"Error generating summary: {str(e)}"

# -----------------------------
# FIXED: ADVANCED ML FEATURES WITH ROBUST ERROR HANDLING
# -----------------------------

def extract_storm_features(typhoon_data):
    """Extract comprehensive features for clustering analysis - FIXED VERSION"""
    try:
        if typhoon_data is None or typhoon_data.empty:
            logging.error("No typhoon data provided for feature extraction")
            return None
        
        # Basic features - ensure columns exist
        basic_features = []
        for sid in typhoon_data['SID'].unique():
            storm_data = typhoon_data[typhoon_data['SID'] == sid].copy()
            
            if len(storm_data) == 0:
                continue
            
            # Initialize feature dict with safe defaults
            features = {'SID': sid}
            
            # Wind statistics
            if 'USA_WIND' in storm_data.columns:
                wind_values = pd.to_numeric(storm_data['USA_WIND'], errors='coerce').dropna()
                if len(wind_values) > 0:
                    features['USA_WIND_max'] = wind_values.max()
                    features['USA_WIND_mean'] = wind_values.mean()
                    features['USA_WIND_std'] = wind_values.std() if len(wind_values) > 1 else 0
                else:
                    features['USA_WIND_max'] = 30
                    features['USA_WIND_mean'] = 30
                    features['USA_WIND_std'] = 0
            else:
                features['USA_WIND_max'] = 30
                features['USA_WIND_mean'] = 30
                features['USA_WIND_std'] = 0
                
            # Pressure statistics
            if 'USA_PRES' in storm_data.columns:
                pres_values = pd.to_numeric(storm_data['USA_PRES'], errors='coerce').dropna()
                if len(pres_values) > 0:
                    features['USA_PRES_min'] = pres_values.min()
                    features['USA_PRES_mean'] = pres_values.mean()
                    features['USA_PRES_std'] = pres_values.std() if len(pres_values) > 1 else 0
                else:
                    features['USA_PRES_min'] = 1000
                    features['USA_PRES_mean'] = 1000
                    features['USA_PRES_std'] = 0
            else:
                features['USA_PRES_min'] = 1000
                features['USA_PRES_mean'] = 1000
                features['USA_PRES_std'] = 0
            
            # Location statistics
            if 'LAT' in storm_data.columns and 'LON' in storm_data.columns:
                lat_values = pd.to_numeric(storm_data['LAT'], errors='coerce').dropna()
                lon_values = pd.to_numeric(storm_data['LON'], errors='coerce').dropna()
                
                if len(lat_values) > 0 and len(lon_values) > 0:
                    features['LAT_mean'] = lat_values.mean()
                    features['LAT_std'] = lat_values.std() if len(lat_values) > 1 else 0
                    features['LAT_max'] = lat_values.max()
                    features['LAT_min'] = lat_values.min()
                    features['LON_mean'] = lon_values.mean()
                    features['LON_std'] = lon_values.std() if len(lon_values) > 1 else 0
                    features['LON_max'] = lon_values.max()
                    features['LON_min'] = lon_values.min()
                    
                    # Genesis location (first valid position)
                    features['genesis_lat'] = lat_values.iloc[0]
                    features['genesis_lon'] = lon_values.iloc[0]
                    features['genesis_intensity'] = features['USA_WIND_mean']  # Use mean as fallback
                    
                    # Track characteristics
                    features['lat_range'] = lat_values.max() - lat_values.min()
                    features['lon_range'] = lon_values.max() - lon_values.min()
                    
                    # Calculate track distance
                    if len(lat_values) > 1:
                        distances = []
                        for i in range(1, len(lat_values)):
                            dlat = lat_values.iloc[i] - lat_values.iloc[i-1]
                            dlon = lon_values.iloc[i] - lon_values.iloc[i-1]
                            distances.append(np.sqrt(dlat**2 + dlon**2))
                        features['total_distance'] = sum(distances)
                        features['avg_speed'] = np.mean(distances) if distances else 0
                    else:
                        features['total_distance'] = 0
                        features['avg_speed'] = 0
                        
                    # Track curvature
                    if len(lat_values) > 2:
                        bearing_changes = []
                        for i in range(1, len(lat_values)-1):
                            dlat1 = lat_values.iloc[i] - lat_values.iloc[i-1]
                            dlon1 = lon_values.iloc[i] - lon_values.iloc[i-1]
                            dlat2 = lat_values.iloc[i+1] - lat_values.iloc[i]
                            dlon2 = lon_values.iloc[i+1] - lon_values.iloc[i]
                            
                            angle1 = np.arctan2(dlat1, dlon1)
                            angle2 = np.arctan2(dlat2, dlon2)
                            change = abs(angle2 - angle1)
                            bearing_changes.append(change)
                        
                        features['avg_curvature'] = np.mean(bearing_changes) if bearing_changes else 0
                    else:
                        features['avg_curvature'] = 0
                else:
                    # Default location values
                    features.update({
                        'LAT_mean': 20, 'LAT_std': 0, 'LAT_max': 20, 'LAT_min': 20,
                        'LON_mean': 140, 'LON_std': 0, 'LON_max': 140, 'LON_min': 140,
                        'genesis_lat': 20, 'genesis_lon': 140, 'genesis_intensity': 30,
                        'lat_range': 0, 'lon_range': 0, 'total_distance': 0,
                        'avg_speed': 0, 'avg_curvature': 0
                    })
            else:
                # Default location values if columns missing
                features.update({
                    'LAT_mean': 20, 'LAT_std': 0, 'LAT_max': 20, 'LAT_min': 20,
                    'LON_mean': 140, 'LON_std': 0, 'LON_max': 140, 'LON_min': 140,
                    'genesis_lat': 20, 'genesis_lon': 140, 'genesis_intensity': 30,
                    'lat_range': 0, 'lon_range': 0, 'total_distance': 0,
                    'avg_speed': 0, 'avg_curvature': 0
                })
            
            # Track length
            features['track_length'] = len(storm_data)
            
            # Add seasonal information
            if 'SEASON' in storm_data.columns:
                features['season'] = storm_data['SEASON'].iloc[0]
            else:
                features['season'] = 2000
                
            # Add basin information
            if 'BASIN' in storm_data.columns:
                features['basin'] = storm_data['BASIN'].iloc[0]
            elif 'SID' in storm_data.columns:
                features['basin'] = sid[:2] if len(sid) >= 2 else 'WP'
            else:
                features['basin'] = 'WP'
            
            basic_features.append(features)
        
        if not basic_features:
            logging.error("No valid storm features could be extracted")
            return None
            
        # Convert to DataFrame
        storm_features = pd.DataFrame(basic_features)
        
        # Ensure all numeric columns are properly typed
        numeric_columns = [col for col in storm_features.columns if col not in ['SID', 'basin']]
        for col in numeric_columns:
            storm_features[col] = pd.to_numeric(storm_features[col], errors='coerce').fillna(0)
        
        logging.info(f"Successfully extracted features for {len(storm_features)} storms")
        logging.info(f"Feature columns: {list(storm_features.columns)}")
        
        return storm_features
        
    except Exception as e:
        logging.error(f"Error in extract_storm_features: {e}")
        import traceback
        traceback.print_exc()
        return None

def perform_dimensionality_reduction(storm_features, method='umap', n_components=2):
    """Perform UMAP or t-SNE dimensionality reduction - FIXED VERSION"""
    try:
        if storm_features is None or storm_features.empty:
            raise ValueError("No storm features provided")
        
        # Select numeric features for clustering - FIXED
        feature_cols = []
        for col in storm_features.columns:
            if col not in ['SID', 'basin'] and storm_features[col].dtype in ['float64', 'int64']:
                # Check if column has valid data
                valid_data = storm_features[col].dropna()
                if len(valid_data) > 0 and valid_data.std() > 0:  # Only include columns with variance
                    feature_cols.append(col)
        
        if len(feature_cols) == 0:
            raise ValueError("No valid numeric features found for clustering")
        
        logging.info(f"Using {len(feature_cols)} features for clustering: {feature_cols}")
        
        X = storm_features[feature_cols].fillna(0)
        
        # Check if we have enough samples
        if len(X) < 2:
            raise ValueError("Need at least 2 storms for clustering")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform dimensionality reduction
        if method.lower() == 'umap' and UMAP_AVAILABLE and len(X_scaled) >= 4:
            # UMAP parameters optimized for typhoon data - fixed warnings
            n_neighbors = min(15, len(X_scaled) - 1)
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric='euclidean',
                random_state=42,
                n_jobs=1  # Explicitly set to avoid warning
            )
        elif method.lower() == 'tsne' and len(X_scaled) >= 4:
            # t-SNE parameters
            perplexity = min(30, len(X_scaled) // 4)
            perplexity = max(1, perplexity)  # Ensure perplexity is at least 1
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=200,
                n_iter=1000,
                random_state=42
            )
        else:
            # Fallback to PCA
            reducer = PCA(n_components=n_components, random_state=42)
        
        # Fit and transform
        embedding = reducer.fit_transform(X_scaled)
        
        logging.info(f"Dimensionality reduction successful: {X_scaled.shape} -> {embedding.shape}")
        
        return embedding, feature_cols, scaler
        
    except Exception as e:
        logging.error(f"Error in perform_dimensionality_reduction: {e}")
        raise

def cluster_storms_data(embedding, method='dbscan', eps=0.5, min_samples=3):
    """Cluster storms based on their embedding - FIXED NAME VERSION"""
    try:
        if len(embedding) < 2:
            return np.array([0] * len(embedding))  # Single cluster for insufficient data
        
        if method.lower() == 'dbscan':
            # Adjust min_samples based on data size
            min_samples = min(min_samples, max(2, len(embedding) // 5))
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        elif method.lower() == 'kmeans':
            # Adjust n_clusters based on data size
            n_clusters = min(5, max(2, len(embedding) // 3))
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        else:
            raise ValueError("Method must be 'dbscan' or 'kmeans'")
        
        clusters = clusterer.fit_predict(embedding)
        
        logging.info(f"Clustering complete: {len(np.unique(clusters))} clusters found")
        
        return clusters
        
    except Exception as e:
        logging.error(f"Error in cluster_storms_data: {e}")
        # Return single cluster as fallback
        return np.array([0] * len(embedding))

def create_separate_clustering_plots(storm_features, typhoon_data, method='umap'):
    """Create separate plots for clustering analysis - ENHANCED CLARITY VERSION"""
    try:
        # Validate inputs
        if storm_features is None or storm_features.empty:
            raise ValueError("No storm features available for clustering")
            
        if typhoon_data is None or typhoon_data.empty:
            raise ValueError("No typhoon data available for route visualization")
        
        logging.info(f"Starting clustering visualization with {len(storm_features)} storms")
        
        # Perform dimensionality reduction
        embedding, feature_cols, scaler = perform_dimensionality_reduction(storm_features, method)
        
        # Perform clustering
        cluster_labels = cluster_storms_data(embedding, 'dbscan')
        
        # Add clustering results to storm features
        storm_features_viz = storm_features.copy()
        storm_features_viz['cluster'] = cluster_labels
        storm_features_viz['dim1'] = embedding[:, 0]
        storm_features_viz['dim2'] = embedding[:, 1]
        
        # Merge with typhoon data for additional info - SAFE MERGE
        try:
            storm_info = typhoon_data.groupby('SID').first()[['NAME', 'SEASON']].reset_index()
            storm_features_viz = storm_features_viz.merge(storm_info, on='SID', how='left')
            # Fill missing values
            storm_features_viz['NAME'] = storm_features_viz['NAME'].fillna('UNNAMED')
            storm_features_viz['SEASON'] = storm_features_viz['SEASON'].fillna(2000)
        except Exception as merge_error:
            logging.warning(f"Could not merge storm info: {merge_error}")
            storm_features_viz['NAME'] = 'UNNAMED'
            storm_features_viz['SEASON'] = 2000
        
        # Get unique clusters and assign distinct colors
        unique_clusters = sorted([c for c in storm_features_viz['cluster'].unique() if c != -1])
        noise_count = len(storm_features_viz[storm_features_viz['cluster'] == -1])
        
        # 1. Enhanced clustering scatter plot with clear cluster identification
        fig_cluster = go.Figure()
        
        # Add noise points first
        if noise_count > 0:
            noise_data = storm_features_viz[storm_features_viz['cluster'] == -1]
            fig_cluster.add_trace(
                go.Scatter(
                    x=noise_data['dim1'],
                    y=noise_data['dim2'],
                    mode='markers',
                    marker=dict(color='lightgray', size=8, opacity=0.5, symbol='x'),
                    name=f'Noise ({noise_count} storms)',
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        'Season: %{customdata[1]}<br>'
                        'Cluster: Noise<br>'
                        f'{method.upper()} Dim 1: %{{x:.2f}}<br>'
                        f'{method.upper()} Dim 2: %{{y:.2f}}<br>'
                        '<extra></extra>'
                    ),
                    customdata=np.column_stack((
                        noise_data['NAME'].fillna('UNNAMED'),
                        noise_data['SEASON'].fillna(2000)
                    ))
                )
            )
        
        # Add clusters with distinct colors and shapes
        cluster_symbols = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 
                          'pentagon', 'hexagon', 'star', 'cross', 'circle-open']
        
        for i, cluster in enumerate(unique_clusters):
            cluster_data = storm_features_viz[storm_features_viz['cluster'] == cluster]
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            symbol = cluster_symbols[i % len(cluster_symbols)]
            
            fig_cluster.add_trace(
                go.Scatter(
                    x=cluster_data['dim1'],
                    y=cluster_data['dim2'],
                    mode='markers',
                    marker=dict(color=color, size=10, symbol=symbol, line=dict(width=1, color='white')),
                    name=f'Cluster {cluster} ({len(cluster_data)} storms)',
                    hovertemplate=(
                        '<b>%{customdata[0]}</b><br>'
                        'Season: %{customdata[1]}<br>'
                        f'Cluster: {cluster}<br>'
                        f'{method.upper()} Dim 1: %{{x:.2f}}<br>'
                        f'{method.upper()} Dim 2: %{{y:.2f}}<br>'
                        'Intensity: %{customdata[2]:.0f} kt<br>'
                        '<extra></extra>'
                    ),
                    customdata=np.column_stack((
                        cluster_data['NAME'].fillna('UNNAMED'),
                        cluster_data['SEASON'].fillna(2000),
                        cluster_data['USA_WIND_max'].fillna(0)
                    ))
                )
            )
        
        fig_cluster.update_layout(
            title=f'Storm Clustering Analysis using {method.upper()}<br><sub>Each symbol/color represents a distinct storm pattern group</sub>',
            xaxis_title=f'{method.upper()} Dimension 1',
            yaxis_title=f'{method.upper()} Dimension 2',
            height=600,
            showlegend=True
        )
        
        # 2. ENHANCED route map with cluster legends and clearer representation
        fig_routes = go.Figure()
        
        # Create a comprehensive legend showing cluster characteristics
        cluster_info_text = []
        
        for i, cluster in enumerate(unique_clusters):
            cluster_storm_ids = storm_features_viz[storm_features_viz['cluster'] == cluster]['SID'].tolist()
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            
            # Get cluster statistics for legend
            cluster_data = storm_features_viz[storm_features_viz['cluster'] == cluster]
            avg_intensity = cluster_data['USA_WIND_max'].mean() if 'USA_WIND_max' in cluster_data.columns else 0
            avg_pressure = cluster_data['USA_PRES_min'].mean() if 'USA_PRES_min' in cluster_data.columns else 1000
            
            cluster_info_text.append(
                f"Cluster {cluster}: {len(cluster_storm_ids)} storms, "
                f"Avg: {avg_intensity:.0f}kt/{avg_pressure:.0f}hPa"
            )
            
            # Add multiple storms per cluster with clear identification
            storms_added = 0
            for j, sid in enumerate(cluster_storm_ids[:8]):  # Show up to 8 storms per cluster
                try:
                    storm_track = typhoon_data[typhoon_data['SID'] == sid].sort_values('ISO_TIME')
                    if len(storm_track) > 1:
                        # Ensure valid coordinates
                        valid_coords = storm_track['LAT'].notna() & storm_track['LON'].notna()
                        storm_track = storm_track[valid_coords]
                        
                        if len(storm_track) > 1:
                            storm_name = storm_track['NAME'].iloc[0] if pd.notna(storm_track['NAME'].iloc[0]) else 'UNNAMED'
                            storm_season = storm_track['SEASON'].iloc[0] if 'SEASON' in storm_track.columns else 'Unknown'
                            
                            # Vary line style for different storms in same cluster
                            line_styles = ['solid', 'dash', 'dot', 'dashdot']
                            line_style = line_styles[j % len(line_styles)]
                            line_width = 3 if j == 0 else 2  # First storm thicker
                            
                            fig_routes.add_trace(
                                go.Scattergeo(
                                    lon=storm_track['LON'],
                                    lat=storm_track['LAT'],
                                    mode='lines+markers',
                                    line=dict(color=color, width=line_width, dash=line_style),
                                    marker=dict(color=color, size=3),
                                    name=f'C{cluster}: {storm_name} ({storm_season})',
                                    showlegend=True,
                                    legendgroup=f'cluster_{cluster}',
                                    hovertemplate=(
                                        f'<b>Cluster {cluster}: {storm_name}</b><br>'
                                        'Lat: %{lat:.1f}°<br>'
                                        'Lon: %{lon:.1f}°<br>'
                                        f'Season: {storm_season}<br>'
                                        f'Pattern Group: {cluster}<br>'
                                        '<extra></extra>'
                                    )
                                )
                            )
                            storms_added += 1
                except Exception as track_error:
                    logging.warning(f"Error adding track for storm {sid}: {track_error}")
                    continue
            
            # Add cluster centroid marker
            if len(cluster_storm_ids) > 0:
                # Calculate average genesis location for cluster
                cluster_storm_data = storm_features_viz[storm_features_viz['cluster'] == cluster]
                if 'genesis_lat' in cluster_storm_data.columns and 'genesis_lon' in cluster_storm_data.columns:
                    avg_lat = cluster_storm_data['genesis_lat'].mean()
                    avg_lon = cluster_storm_data['genesis_lon'].mean()
                    
                    fig_routes.add_trace(
                        go.Scattergeo(
                            lon=[avg_lon],
                            lat=[avg_lat],
                            mode='markers',
                            marker=dict(
                                color=color, 
                                size=20, 
                                symbol='star',
                                line=dict(width=2, color='white')
                            ),
                            name=f'C{cluster} Center',
                            showlegend=True,
                            legendgroup=f'cluster_{cluster}',
                            hovertemplate=(
                                f'<b>Cluster {cluster} Genesis Center</b><br>'
                                f'Avg Position: {avg_lat:.1f}°N, {avg_lon:.1f}°E<br>'
                                f'Storms: {len(cluster_storm_ids)}<br>'
                                f'Avg Intensity: {avg_intensity:.0f} kt<br>'
                                '<extra></extra>'
                            )
                        )
                    )
        
        # Update route map layout with enhanced information and LARGER SIZE
        fig_routes.update_layout(
            title=f"Storm Routes by {method.upper()} Clusters<br><sub>Different line styles = different storms in same cluster | Stars = cluster centers</sub>",
            geo=dict(
                projection_type="natural earth",
                showland=True,
                landcolor="LightGray",
                showocean=True,
                oceancolor="LightBlue",
                showcoastlines=True,
                coastlinecolor="Gray",
                center=dict(lat=20, lon=140),
                projection_scale=2.5  # Larger map
            ),
            height=800,  # Much larger height
            width=1200,  # Wider map
            showlegend=True
        )
        
        # Add cluster info annotation
        cluster_summary = "<br>".join(cluster_info_text)
        fig_routes.add_annotation(
            text=f"<b>Cluster Summary:</b><br>{cluster_summary}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1
        )
        
        # 3. Enhanced pressure evolution plot with cluster identification
        fig_pressure = go.Figure()
        
        for i, cluster in enumerate(unique_clusters):
            cluster_storm_ids = storm_features_viz[storm_features_viz['cluster'] == cluster]['SID'].tolist()
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            
            cluster_pressures = []
            for j, sid in enumerate(cluster_storm_ids[:5]):  # Limit to 5 storms per cluster
                try:
                    storm_track = typhoon_data[typhoon_data['SID'] == sid].sort_values('ISO_TIME')
                    if len(storm_track) > 1 and 'USA_PRES' in storm_track.columns:
                        pressure_values = pd.to_numeric(storm_track['USA_PRES'], errors='coerce').dropna()
                        if len(pressure_values) > 0:
                            storm_name = storm_track['NAME'].iloc[0] if pd.notna(storm_track['NAME'].iloc[0]) else 'UNNAMED'
                            time_hours = range(len(pressure_values))
                            
                            # Normalize time to show relative progression
                            normalized_time = np.linspace(0, 100, len(pressure_values))
                            
                            fig_pressure.add_trace(
                                go.Scatter(
                                    x=normalized_time,
                                    y=pressure_values,
                                    mode='lines',
                                    line=dict(color=color, width=2, dash='solid' if j == 0 else 'dash'),
                                    name=f'C{cluster}: {storm_name}' if j == 0 else None,
                                    showlegend=(j == 0),
                                    legendgroup=f'pressure_cluster_{cluster}',
                                    hovertemplate=(
                                        f'<b>Cluster {cluster}: {storm_name}</b><br>'
                                        'Progress: %{x:.0f}%<br>'
                                        'Pressure: %{y:.0f} hPa<br>'
                                        '<extra></extra>'
                                    ),
                                    opacity=0.8 if j == 0 else 0.5
                                )
                            )
                            cluster_pressures.extend(pressure_values)
                except Exception as e:
                    continue
            
            # Add cluster average line
            if cluster_pressures:
                avg_pressure = np.mean(cluster_pressures)
                fig_pressure.add_hline(
                    y=avg_pressure,
                    line_dash="dot",
                    line_color=color,
                    annotation_text=f"C{cluster} Avg: {avg_pressure:.0f}",
                    annotation_position="right"
                )
        
        fig_pressure.update_layout(
            title=f"Pressure Evolution by {method.upper()} Clusters<br><sub>Normalized timeline (0-100%) | Dotted lines = cluster averages</sub>",
            xaxis_title="Storm Progress (%)",
            yaxis_title="Pressure (hPa)",
            height=500
        )
        
        # 4. Enhanced wind evolution plot
        fig_wind = go.Figure()
        
        for i, cluster in enumerate(unique_clusters):
            cluster_storm_ids = storm_features_viz[storm_features_viz['cluster'] == cluster]['SID'].tolist()
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            
            cluster_winds = []
            for j, sid in enumerate(cluster_storm_ids[:5]):  # Limit to 5 storms per cluster
                try:
                    storm_track = typhoon_data[typhoon_data['SID'] == sid].sort_values('ISO_TIME')
                    if len(storm_track) > 1 and 'USA_WIND' in storm_track.columns:
                        wind_values = pd.to_numeric(storm_track['USA_WIND'], errors='coerce').dropna()
                        if len(wind_values) > 0:
                            storm_name = storm_track['NAME'].iloc[0] if pd.notna(storm_track['NAME'].iloc[0]) else 'UNNAMED'
                            
                            # Normalize time to show relative progression
                            normalized_time = np.linspace(0, 100, len(wind_values))
                            
                            fig_wind.add_trace(
                                go.Scatter(
                                    x=normalized_time,
                                    y=wind_values,
                                    mode='lines',
                                    line=dict(color=color, width=2, dash='solid' if j == 0 else 'dash'),
                                    name=f'C{cluster}: {storm_name}' if j == 0 else None,
                                    showlegend=(j == 0),
                                    legendgroup=f'wind_cluster_{cluster}',
                                    hovertemplate=(
                                        f'<b>Cluster {cluster}: {storm_name}</b><br>'
                                        'Progress: %{x:.0f}%<br>'
                                        'Wind: %{y:.0f} kt<br>'
                                        '<extra></extra>'
                                    ),
                                    opacity=0.8 if j == 0 else 0.5
                                )
                            )
                            cluster_winds.extend(wind_values)
                except Exception as e:
                    continue
            
            # Add cluster average line
            if cluster_winds:
                avg_wind = np.mean(cluster_winds)
                fig_wind.add_hline(
                    y=avg_wind,
                    line_dash="dot",
                    line_color=color,
                    annotation_text=f"C{cluster} Avg: {avg_wind:.0f}",
                    annotation_position="right"
                )
        
        fig_wind.update_layout(
            title=f"Wind Speed Evolution by {method.upper()} Clusters<br><sub>Normalized timeline (0-100%) | Dotted lines = cluster averages</sub>",
            xaxis_title="Storm Progress (%)",
            yaxis_title="Wind Speed (kt)",
            height=500
        )
        
        # Generate enhanced cluster statistics with clear explanations
        try:
            stats_text = f"ENHANCED {method.upper()} CLUSTER ANALYSIS RESULTS\n" + "="*60 + "\n\n"
            stats_text += f"🔍 DIMENSIONALITY REDUCTION: {method.upper()}\n"
            stats_text += f"🎯 CLUSTERING ALGORITHM: DBSCAN (automatic pattern discovery)\n"
            stats_text += f"📊 TOTAL STORMS ANALYZED: {len(storm_features_viz)}\n"
            stats_text += f"🎨 CLUSTERS DISCOVERED: {len(unique_clusters)}\n"
            if noise_count > 0:
                stats_text += f"❌ NOISE POINTS: {noise_count} storms (don't fit clear patterns)\n"
            stats_text += "\n"
            
            for cluster in sorted(storm_features_viz['cluster'].unique()):
                cluster_data = storm_features_viz[storm_features_viz['cluster'] == cluster]
                storm_count = len(cluster_data)
                
                if cluster == -1:
                    stats_text += f"❌ NOISE GROUP: {storm_count} storms\n"
                    stats_text += "   → These storms don't follow the main patterns\n"
                    stats_text += "   → May represent unique or rare storm behaviors\n\n"
                    continue
                
                stats_text += f"🎯 CLUSTER {cluster}: {storm_count} storms\n"
                stats_text += f"   🎨 Color: {CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]}\n"
                
                # Add detailed statistics if available
                if 'USA_WIND_max' in cluster_data.columns:
                    wind_mean = cluster_data['USA_WIND_max'].mean()
                    wind_std = cluster_data['USA_WIND_max'].std()
                    stats_text += f"   💨 Intensity: {wind_mean:.1f} ± {wind_std:.1f} kt\n"
                
                if 'USA_PRES_min' in cluster_data.columns:
                    pres_mean = cluster_data['USA_PRES_min'].mean()
                    pres_std = cluster_data['USA_PRES_min'].std()
                    stats_text += f"   🌡️ Pressure: {pres_mean:.1f} ± {pres_std:.1f} hPa\n"
                
                if 'track_length' in cluster_data.columns:
                    track_mean = cluster_data['track_length'].mean()
                    stats_text += f"   📏 Avg Track Length: {track_mean:.1f} points\n"
                
                if 'genesis_lat' in cluster_data.columns and 'genesis_lon' in cluster_data.columns:
                    lat_mean = cluster_data['genesis_lat'].mean()
                    lon_mean = cluster_data['genesis_lon'].mean()
                    stats_text += f"   🎯 Genesis Region: {lat_mean:.1f}°N, {lon_mean:.1f}°E\n"
                
                # Add interpretation
                if wind_mean < 50:
                    stats_text += "   💡 Pattern: Weaker storm group\n"
                elif wind_mean > 100:
                    stats_text += "   💡 Pattern: Intense storm group\n"
                else:
                    stats_text += "   💡 Pattern: Moderate intensity group\n"
                
                stats_text += "\n"
            
            # Add explanation of the analysis
            stats_text += "📖 INTERPRETATION GUIDE:\n"
            stats_text += f"• {method.upper()} reduces storm characteristics to 2D for visualization\n"
            stats_text += "• DBSCAN finds natural groupings without preset number of clusters\n"
            stats_text += "• Each cluster represents storms with similar behavior patterns\n"
            stats_text += "• Route colors match cluster colors from the similarity plot\n"
            stats_text += "• Stars on map show average genesis locations for each cluster\n"
            stats_text += "• Temporal plots show how each cluster behaves over time\n\n"
            
            stats_text += f"🔧 FEATURES USED FOR CLUSTERING:\n"
            stats_text += f"   Total: {len(feature_cols)} storm characteristics\n"
            stats_text += f"   Including: intensity, pressure, track shape, genesis location\n"
            
        except Exception as stats_error:
            stats_text = f"Error generating enhanced statistics: {str(stats_error)}"
        
        return fig_cluster, fig_routes, fig_pressure, fig_wind, stats_text
        
    except Exception as e:
        logging.error(f"Error in enhanced clustering analysis: {e}")
        import traceback
        traceback.print_exc()
        
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error in clustering analysis: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=16
        )
        return error_fig, error_fig, error_fig, error_fig, f"Error in clustering: {str(e)}"

# -----------------------------
# Regression Functions (Original)
# -----------------------------

def perform_wind_regression(start_year, start_month, end_year, end_month):
    """Perform wind regression analysis"""
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].dropna(subset=['USA_WIND','ONI'])
    data['severe_typhoon'] = (data['USA_WIND']>=64).astype(int)
    X = sm.add_constant(data['ONI'])
    y = data['severe_typhoon']
    try:
        model = sm.Logit(y, X).fit(disp=0)
        beta_1 = model.params['ONI']
        exp_beta_1 = np.exp(beta_1)
        p_value = model.pvalues['ONI']
        return f"Wind Regression: β1={beta_1:.4f}, Odds Ratio={exp_beta_1:.4f}, P-value={p_value:.4f}"
    except Exception as e:
        return f"Wind Regression Error: {e}"

def perform_pressure_regression(start_year, start_month, end_year, end_month):
    """Perform pressure regression analysis"""
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].dropna(subset=['USA_PRES','ONI'])
    data['intense_typhoon'] = (data['USA_PRES']<=950).astype(int)
    X = sm.add_constant(data['ONI'])
    y = data['intense_typhoon']
    try:
        model = sm.Logit(y, X).fit(disp=0)
        beta_1 = model.params['ONI']
        exp_beta_1 = np.exp(beta_1)
        p_value = model.pvalues['ONI']
        return f"Pressure Regression: β1={beta_1:.4f}, Odds Ratio={exp_beta_1:.4f}, P-value={p_value:.4f}"
    except Exception as e:
        return f"Pressure Regression Error: {e}"

def perform_longitude_regression(start_year, start_month, end_year, end_month):
    """Perform longitude regression analysis"""
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].dropna(subset=['LON','ONI'])
    data['western_typhoon'] = (data['LON']<=140).astype(int)
    X = sm.add_constant(data['ONI'])
    y = data['western_typhoon']
    try:
        model = sm.OLS(y, sm.add_constant(X)).fit()
        beta_1 = model.params['ONI']
        exp_beta_1 = np.exp(beta_1)
        p_value = model.pvalues['ONI']
        return f"Longitude Regression: β1={beta_1:.4f}, Odds Ratio={exp_beta_1:.4f}, P-value={p_value:.4f}"
    except Exception as e:
        return f"Longitude Regression Error: {e}"

# -----------------------------
# Visualization Functions (Enhanced)
# -----------------------------

def get_full_tracks(start_year, start_month, end_year, end_month, enso_phase, typhoon_search):
    """Get full typhoon tracks"""
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    filtered_data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].copy()
    filtered_data['ENSO_Phase'] = filtered_data['ONI'].apply(classify_enso_phases)
    if enso_phase != 'all':
        filtered_data = filtered_data[filtered_data['ENSO_Phase'] == enso_phase.capitalize()]
    unique_storms = filtered_data['SID'].unique()
    count = len(unique_storms)
    fig = go.Figure()
    for sid in unique_storms:
        storm_data = typhoon_data[typhoon_data['SID']==sid]
        if storm_data.empty:
            continue
        name = storm_data['NAME'].iloc[0] if pd.notnull(storm_data['NAME'].iloc[0]) else "Unnamed"
        basin = storm_data['SID'].iloc[0][:2]
        storm_oni = filtered_data[filtered_data['SID']==sid]['ONI'].iloc[0]
        color = 'red' if storm_oni>=0.5 else ('blue' if storm_oni<=-0.5 else 'green')
        fig.add_trace(go.Scattergeo(
            lon=storm_data['LON'], lat=storm_data['LAT'], mode='lines',
            name=f"{name} ({basin})",
            line=dict(width=1.5, color=color), hoverinfo="name"
        ))
    if typhoon_search:
        search_mask = typhoon_data['NAME'].str.contains(typhoon_search, case=False, na=False)
        if search_mask.any():
            for sid in typhoon_data[search_mask]['SID'].unique():
                storm_data = typhoon_data[typhoon_data['SID']==sid]
                fig.add_trace(go.Scattergeo(
                    lon=storm_data['LON'], lat=storm_data['LAT'], mode='lines+markers',
                    name=f"MATCHED: {storm_data['NAME'].iloc[0]}",
                    line=dict(width=3, color='yellow'),
                    marker=dict(size=5), hoverinfo="name"
                ))
    fig.update_layout(
        title=f"Typhoon Tracks ({start_year}-{start_month} to {end_year}-{end_month})",
        geo=dict(
            projection_type='natural earth',
            showland=True,
            showcoastlines=True,
            landcolor='rgb(243,243,243)',
            countrycolor='rgb(204,204,204)',
            coastlinecolor='rgb(204,204,204)',
            center=dict(lon=140, lat=20),
            projection_scale=3
        ),
        legend_title="Typhoons by ENSO Phase",
        showlegend=True,
        height=700
    )
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text="Red: El Niño, Blue: La Nina, Green: Neutral",
        showarrow=False, align="left",
        bgcolor="rgba(255,255,255,0.8)"
    )
    return fig, f"Total typhoons displayed: {count}"

def get_wind_analysis(start_year, start_month, end_year, end_month, enso_phase, typhoon_search):
    """Get wind analysis with enhanced categorization"""
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    filtered_data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].copy()
    filtered_data['ENSO_Phase'] = filtered_data['ONI'].apply(classify_enso_phases)
    if enso_phase != 'all':
        filtered_data = filtered_data[filtered_data['ENSO_Phase'] == enso_phase.capitalize()]
    
    fig = px.scatter(filtered_data, x='ONI', y='USA_WIND', color='Category',
                     hover_data=['NAME','Year','Category'],
                     title='Wind Speed vs ONI',
                     labels={'ONI':'ONI Value','USA_WIND':'Max Wind Speed (knots)'},
                     color_discrete_map=enhanced_color_map)
    
    if typhoon_search:
        mask = filtered_data['NAME'].str.contains(typhoon_search, case=False, na=False)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=filtered_data.loc[mask,'ONI'], y=filtered_data.loc[mask,'USA_WIND'],
                mode='markers', marker=dict(size=10, color='red', symbol='star'),
                name=f'Matched: {typhoon_search}',
                text=filtered_data.loc[mask,'NAME']+' ('+filtered_data.loc[mask,'Year'].astype(str)+')'
            ))
    
    regression = perform_wind_regression(start_year, start_month, end_year, end_month)
    return fig, regression

def get_pressure_analysis(start_year, start_month, end_year, end_month, enso_phase, typhoon_search):
    """Get pressure analysis with enhanced categorization"""
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    filtered_data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].copy()
    filtered_data['ENSO_Phase'] = filtered_data['ONI'].apply(classify_enso_phases)
    if enso_phase != 'all':
        filtered_data = filtered_data[filtered_data['ENSO_Phase'] == enso_phase.capitalize()]
    
    fig = px.scatter(filtered_data, x='ONI', y='USA_PRES', color='Category',
                     hover_data=['NAME','Year','Category'],
                     title='Pressure vs ONI',
                     labels={'ONI':'ONI Value','USA_PRES':'Min Pressure (hPa)'},
                     color_discrete_map=enhanced_color_map)
    
    if typhoon_search:
        mask = filtered_data['NAME'].str.contains(typhoon_search, case=False, na=False)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=filtered_data.loc[mask,'ONI'], y=filtered_data.loc[mask,'USA_PRES'],
                mode='markers', marker=dict(size=10, color='red', symbol='star'),
                name=f'Matched: {typhoon_search}',
                text=filtered_data.loc[mask,'NAME']+' ('+filtered_data.loc[mask,'Year'].astype(str)+')'
            ))
    
    regression = perform_pressure_regression(start_year, start_month, end_year, end_month)
    return fig, regression

def get_longitude_analysis(start_year, start_month, end_year, end_month, enso_phase, typhoon_search):
    """Get longitude analysis"""
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 28)
    filtered_data = merged_data[(merged_data['ISO_TIME']>=start_date) & (merged_data['ISO_TIME']<=end_date)].copy()
    filtered_data['ENSO_Phase'] = filtered_data['ONI'].apply(classify_enso_phases)
    if enso_phase != 'all':
        filtered_data = filtered_data[filtered_data['ENSO_Phase'] == enso_phase.capitalize()]
    
    fig = px.scatter(filtered_data, x='LON', y='ONI', hover_data=['NAME'],
                     title='Typhoon Generation Longitude vs ONI (All Years)')
    
    if len(filtered_data) > 1:
        X = np.array(filtered_data['LON']).reshape(-1,1)
        y = filtered_data['ONI']
        try:
            model = sm.OLS(y, sm.add_constant(X)).fit()
            y_pred = model.predict(sm.add_constant(X))
            fig.add_trace(go.Scatter(x=filtered_data['LON'], y=y_pred, mode='lines', name='Regression Line'))
            slope = model.params[1]
            slopes_text = f"All Years Slope: {slope:.4f}"
        except Exception as e:
            slopes_text = f"Regression Error: {e}"
    else:
        slopes_text = "Insufficient data for regression"
    
    regression = perform_longitude_regression(start_year, start_month, end_year, end_month)
    return fig, slopes_text, regression

# -----------------------------
# ENHANCED: Animation Functions with Taiwan Standard Support
# -----------------------------

def get_available_years(typhoon_data):
    """Get all available years including 2025 - with error handling"""
    try:
        if typhoon_data is None or typhoon_data.empty:
            return [str(year) for year in range(2000, 2026)]
            
        if 'ISO_TIME' in typhoon_data.columns:
            years = typhoon_data['ISO_TIME'].dt.year.dropna().unique()
        elif 'SEASON' in typhoon_data.columns:
            years = typhoon_data['SEASON'].dropna().unique()
        else:
            years = range(2000, 2026)  # Default range including 2025
        
        # Convert to strings and sort
        year_strings = sorted([str(int(year)) for year in years if not pd.isna(year)])
        
        # Ensure we have at least some years
        if not year_strings:
            return [str(year) for year in range(2000, 2026)]
            
        return year_strings
        
    except Exception as e:
        print(f"Error in get_available_years: {e}")
        return [str(year) for year in range(2000, 2026)]

def update_typhoon_options_enhanced(year, basin):
    """Enhanced typhoon options with TD support and 2025 data"""
    try:
        year = int(year)
        
        # Filter by year - handle both ISO_TIME and SEASON columns
        if 'ISO_TIME' in typhoon_data.columns:
            year_mask = typhoon_data['ISO_TIME'].dt.year == year
        elif 'SEASON' in typhoon_data.columns:
            year_mask = typhoon_data['SEASON'] == year
        else:
            # Fallback - try to extract year from SID or other fields
            year_mask = typhoon_data.index >= 0  # Include all data as fallback
        
        year_data = typhoon_data[year_mask].copy()
        
        # Filter by basin if specified
        if basin != "All Basins":
            basin_code = basin.split(' - ')[0] if ' - ' in basin else basin[:2]
            if 'SID' in year_data.columns:
                year_data = year_data[year_data['SID'].str.startswith(basin_code, na=False)]
            elif 'BASIN' in year_data.columns:
                year_data = year_data[year_data['BASIN'] == basin_code]
        
        if year_data.empty:
            return gr.update(choices=["No storms found"], value=None)
        
        # Get unique storms - include ALL intensities (including TD)
        storms = year_data.groupby('SID').agg({
            'NAME': 'first',
            'USA_WIND': 'max'
        }).reset_index()
        
        # Enhanced categorization including TD
        storms['category'] = storms['USA_WIND'].apply(categorize_typhoon_enhanced)
        
        # Create options with category information
        options = []
        for _, storm in storms.iterrows():
            name = storm['NAME'] if pd.notna(storm['NAME']) and storm['NAME'] != '' else 'UNNAMED'
            sid = storm['SID']
            category = storm['category']
            max_wind = storm['USA_WIND'] if pd.notna(storm['USA_WIND']) else 0
            
            option = f"{name} ({sid}) - {category} ({max_wind:.0f}kt)"
            options.append(option)
        
        if not options:
            return gr.update(choices=["No storms found"], value=None)
        
        return gr.update(choices=sorted(options), value=options[0])
        
    except Exception as e:
        print(f"Error in update_typhoon_options_enhanced: {e}")
        return gr.update(choices=["Error loading storms"], value=None)

def generate_enhanced_track_video(year, typhoon_selection, standard):
    """Enhanced track video generation with TD support, Taiwan standard, and 2025 compatibility"""
    if not typhoon_selection or typhoon_selection == "No storms found":
        return None
    
    try:
        # Extract SID from selection
        sid = typhoon_selection.split('(')[1].split(')')[0]
        
        # Get storm data
        storm_df = typhoon_data[typhoon_data['SID'] == sid].copy()
        if storm_df.empty:
            print(f"No data found for storm {sid}")
            return None
        
        # Sort by time
        if 'ISO_TIME' in storm_df.columns:
            storm_df = storm_df.sort_values('ISO_TIME')
        
        # Extract data for animation
        lats = storm_df['LAT'].astype(float).values
        lons = storm_df['LON'].astype(float).values
        
        if 'USA_WIND' in storm_df.columns:
            winds = pd.to_numeric(storm_df['USA_WIND'], errors='coerce').fillna(0).values
        else:
            winds = np.full(len(lats), 30)  # Default TD strength
        
        # Enhanced metadata
        storm_name = storm_df['NAME'].iloc[0] if pd.notna(storm_df['NAME'].iloc[0]) else "UNNAMED"
        season = storm_df['SEASON'].iloc[0] if 'SEASON' in storm_df.columns else year
        
        print(f"Generating video for {storm_name} ({sid}) with {len(lats)} track points using {standard} standard")
        
        # Create figure with enhanced map
        fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Enhanced map features
        ax.stock_img()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.5)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
        
        # Set extent based on track
        padding = 5
        ax.set_extent([
            min(lons) - padding, max(lons) + padding,
            min(lats) - padding, max(lats) + padding
        ])
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.3)
        gl.top_labels = gl.right_labels = False
        
        # Title with enhanced info and standard
        ax.set_title(f"{season} {storm_name} ({sid}) Track Animation - {standard.upper()} Standard", 
                    fontsize=18, fontweight='bold')
        
        # Animation elements
        line, = ax.plot([], [], 'b-', linewidth=3, alpha=0.7, label='Track')
        point, = ax.plot([], [], 'o', markersize=15)
        
        # Enhanced info display
        info_box = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                          fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))
        
        # Color legend with both standards - ENHANCED
        legend_elements = []
        
        if standard == 'taiwan':
            categories = ['Tropical Depression', 'Tropical Storm', 'Moderate Typhoon', 'Intense Typhoon']
            for category in categories:
                color = get_taiwan_color(category)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=color, markersize=10, label=category))
        else:
            categories = ['Tropical Depression', 'Tropical Storm', 'C1 Typhoon', 'C2 Typhoon', 
                         'C3 Strong Typhoon', 'C4 Very Strong Typhoon', 'C5 Super Typhoon']
            for category in categories:
                color = get_matplotlib_color(category)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=color, markersize=10, label=category))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        def animate(frame):
            try:
                if frame >= len(lats):
                    return line, point, info_box
                
                # Update track line
                line.set_data(lons[:frame+1], lats[:frame+1])
                
                # Update current position with appropriate categorization
                current_wind = winds[frame]
                
                if standard == 'taiwan':
                    category, color = categorize_typhoon_by_standard(current_wind, 'taiwan')
                else:
                    category, color = categorize_typhoon_by_standard(current_wind, 'atlantic')
                
                # Debug print for first few frames
                if frame < 3:
                    print(f"Frame {frame}: Wind={current_wind:.1f}kt, Category={category}, Color={color}, Standard={standard}")
                
                point.set_data([lons[frame]], [lats[frame]])
                point.set_color(color)
                point.set_markersize(10 + current_wind/8)  # Size based on intensity
                
                # Enhanced info display with standard information
                if 'ISO_TIME' in storm_df.columns and frame < len(storm_df):
                    current_time = storm_df.iloc[frame]['ISO_TIME']
                    time_str = current_time.strftime('%Y-%m-%d %H:%M UTC') if pd.notna(current_time) else 'Unknown'
                else:
                    time_str = f"Step {frame+1}"
                
                # Convert wind speed for Taiwan standard display
                if standard == 'taiwan':
                    wind_ms = current_wind * 0.514444  # Convert to m/s for display
                    wind_display = f"{current_wind:.0f} kt ({wind_ms:.1f} m/s)"
                else:
                    wind_display = f"{current_wind:.0f} kt"
                
                info_text = (
                    f"Storm: {storm_name}\n"
                    f"Time: {time_str}\n"
                    f"Position: {lats[frame]:.1f}°N, {lons[frame]:.1f}°E\n"
                    f"Max Wind: {wind_display}\n"
                    f"Category: {category}\n"
                    f"Standard: {standard.upper()}\n"
                    f"Frame: {frame+1}/{len(lats)}"
                )
                info_box.set_text(info_text)
                
                return line, point, info_box
                
            except Exception as e:
                print(f"Error in animate frame {frame}: {e}")
                return line, point, info_box
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(lats),
            interval=400, blit=False, repeat=True  # Slightly slower for better viewing
        )
        
        # Save animation with graceful fallback if FFmpeg is unavailable
        if shutil.which('ffmpeg'):
            writer = animation.FFMpegWriter(
                fps=3, bitrate=2000, codec='libx264',
                extra_args=['-pix_fmt', 'yuv420p']
            )
            suffix = '.mp4'
        else:
            print("FFmpeg not found - generating GIF instead")
            writer = animation.PillowWriter(fps=3)
            suffix = '.gif'

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix,
                                              dir=tempfile.gettempdir())

        print(f"Saving animation to {temp_file.name}")
        anim.save(temp_file.name, writer=writer, dpi=120)
        plt.close(fig)
        
        print(f"Video generated successfully: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        print(f"Error generating video: {e}")
        import traceback
        traceback.print_exc()
        return None

# Simplified wrapper for backward compatibility - FIXED
def simplified_track_video(year, basin, typhoon, standard):
    """Simplified track video function with fixed color handling"""
    if not typhoon:
        return None
    return generate_enhanced_track_video(year, typhoon, standard)

# -----------------------------
# Load & Process Data
# -----------------------------

# Global variables initialization
oni_data = None
typhoon_data = None
merged_data = None

def initialize_data():
    """Initialize all data safely"""
    global oni_data, typhoon_data, merged_data
    try:
        logging.info("Starting data loading process...")
        update_oni_data()
        oni_data, typhoon_data = load_data_fixed(ONI_DATA_PATH, TYPHOON_DATA_PATH)
        
        if oni_data is not None and typhoon_data is not None:
            oni_long = process_oni_data(oni_data)
            typhoon_max = process_typhoon_data(typhoon_data)
            merged_data = merge_data(oni_long, typhoon_max)
            logging.info("Data loading complete.")
        else:
            logging.error("Failed to load required data")
            # Create minimal fallback data
            oni_data = pd.DataFrame({'Year': [2000], 'Jan': [0], 'Feb': [0], 'Mar': [0], 'Apr': [0], 
                                   'May': [0], 'Jun': [0], 'Jul': [0], 'Aug': [0], 'Sep': [0], 
                                   'Oct': [0], 'Nov': [0], 'Dec': [0]})
            typhoon_data = create_fallback_typhoon_data()
            oni_long = process_oni_data(oni_data)
            typhoon_max = process_typhoon_data(typhoon_data)
            merged_data = merge_data(oni_long, typhoon_max)
    except Exception as e:
        logging.error(f"Error during data initialization: {e}")
        # Create minimal fallback data
        oni_data = pd.DataFrame({'Year': [2000], 'Jan': [0], 'Feb': [0], 'Mar': [0], 'Apr': [0], 
                               'May': [0], 'Jun': [0], 'Jul': [0], 'Aug': [0], 'Sep': [0], 
                               'Oct': [0], 'Nov': [0], 'Dec': [0]})
        typhoon_data = create_fallback_typhoon_data()
        oni_long = process_oni_data(oni_data)
        typhoon_max = process_typhoon_data(typhoon_data)
        merged_data = merge_data(oni_long, typhoon_max)

# Initialize data
initialize_data()

# -----------------------------
# ENHANCED: Gradio Interface with Fixed Route Visualization and Enhanced Features
# -----------------------------

def create_interface():
    """Create the enhanced Gradio interface with robust error handling"""
    try:
        # Ensure data is available
        if oni_data is None or typhoon_data is None or merged_data is None:
            logging.warning("Data not properly loaded, creating minimal interface")
            return create_minimal_fallback_interface()
            
        # Get safe data statistics
        try:
            total_storms = len(typhoon_data['SID'].unique()) if 'SID' in typhoon_data.columns else 0
            total_records = len(typhoon_data)
            available_years = get_available_years(typhoon_data)
            year_range_display = f"{available_years[0]} - {available_years[-1]}" if available_years else "Unknown"
        except Exception as e:
            logging.error(f"Error getting data statistics: {e}")
            total_storms = 0
            total_records = 0
            year_range_display = "Unknown"
            available_years = [str(year) for year in range(2000, 2026)]

        with gr.Blocks(title="Enhanced Typhoon Analysis Platform", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🌪️ Enhanced Typhoon Analysis Platform")
            gr.Markdown("**Advanced ML clustering, route predictions, and comprehensive tropical cyclone analysis including Tropical Depressions**")
            
            with gr.Tab("🏠 Overview"):
                overview_text = f"""
                ## Welcome to the Enhanced Typhoon Analysis Dashboard

                This dashboard provides comprehensive analysis of typhoon data in relation to ENSO phases with advanced machine learning capabilities.

                ### 🚀 Enhanced Features:
                - **Advanced ML Clustering**: UMAP/t-SNE storm pattern analysis with separate visualizations
                - **Predictive Routing**: Advanced storm track and intensity forecasting with uncertainty quantification
                - **Complete TD Support**: Now includes Tropical Depressions (< 34 kt)
                - **Taiwan Standard**: Full support for Taiwan meteorological classification system
                - **2025 Data Ready**: Real-time compatibility with current year data
                - **Enhanced Animations**: High-quality storm track visualizations with both standards
                
                ### 📊 Data Status:
                - **ONI Data**: {len(oni_data)} years loaded
                - **Typhoon Data**: {total_records:,} records loaded
                - **Merged Data**: {len(merged_data):,} typhoons with ONI values
                - **Available Years**: {year_range_display}
                
                ### 🔧 Technical Capabilities:
                - **UMAP Clustering**: {"✅ Available" if UMAP_AVAILABLE else "⚠️ Limited to t-SNE/PCA"}
                - **AI Predictions**: {"🧠 Deep Learning" if CNN_AVAILABLE else "🔬 Physics-based"}
                - **Enhanced Categorization**: Tropical Depression to Super Typhoon
                - **Platform**: Optimized for Hugging Face Spaces
                
                ### 📈 Research Applications:
                - Climate change impact studies
                - Seasonal forecasting research
                - Storm pattern classification
                - ENSO-typhoon relationship analysis
                - Intensity prediction model development
                """
                gr.Markdown(overview_text)

            with gr.Tab("🌊 Monthly Typhoon Genesis Prediction"):
                gr.Markdown("## 🌊 Monthly Typhoon Genesis Prediction")
                gr.Markdown("**Enter month (1-12) and ONI value to see realistic typhoon development throughout the month using Genesis Potential Index**")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        genesis_month = gr.Slider(
                            1, 12, 
                            label="Month", 
                            value=9,
                            step=1,
                            info="1=Jan, 2=Feb, ..., 12=Dec"
                        )
                        genesis_oni = gr.Number(
                            label="ONI Value", 
                            value=0.0,
                            info="El Niño (+) / La Niña (-) / Neutral (0)"
                        )
                        enable_genesis_animation = gr.Checkbox(
                            label="Enable Animation", 
                            value=True,
                            info="Watch daily genesis potential evolution"
                        )
                        generate_genesis_btn = gr.Button("🌊 Generate Monthly Genesis Prediction", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 🌊 What You'll Get:")
                        gr.Markdown("""
                        - **Daily GPI Evolution**: See genesis potential change day-by-day throughout the month
                        - **Genesis Event Detection**: Automatic identification of likely cyclogenesis times/locations
                        - **Storm Track Development**: Physics-based tracks from each genesis point
                        - **Real-time Animation**: Watch storms develop and move with uncertainty visualization
                        - **Environmental Analysis**: SST, humidity, wind shear, and vorticity effects
                        - **ENSO Modulation**: How El Niño/La Niña affects monthly patterns
                        """)
                
                with gr.Row():
                    genesis_animation = gr.HTML(label="🗺️ Daily Genesis Potential & Storm Development")
                
                with gr.Row():
                    genesis_summary = gr.Textbox(label="📋 Monthly Genesis Analysis Summary", lines=25)
                
                def run_genesis_prediction(month, oni, animation):
                    try:
                        # Generate monthly prediction using GPI
                        prediction_data = generate_genesis_prediction_monthly(month, oni, year=2025)
                        logging.info(
                            f"Genesis prediction run for month={month}, oni={oni}: {len(prediction_data.get('genesis_events', []))} events"
                        )

                        # Create animation figure
                        genesis_fig = create_genesis_animation(prediction_data, animation)

                        # Convert to HTML with inline Plotly JS for HF Spaces without CDN access
                        genesis_html = pio.to_html(
                            genesis_fig,
                            include_plotlyjs='inline',
                            full_html=False
                        )

                        # Generate summary
                        summary_text = create_prediction_summary(prediction_data)

                        return genesis_html, summary_text

                    except Exception as e:
                        import traceback
                        error_msg = f"Genesis prediction failed: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
                        logging.error(error_msg)
                        err_fig = create_error_plot(error_msg)
                        err_html = pio.to_html(err_fig, include_plotlyjs='inline', full_html=False)
                        return err_html, error_msg
                
                generate_genesis_btn.click(
                    fn=run_genesis_prediction,
                    inputs=[genesis_month, genesis_oni, enable_genesis_animation],
                    outputs=[genesis_animation, genesis_summary]
                )

            with gr.Tab("🔬 Advanced ML Clustering"):
                gr.Markdown("## 🎯 Storm Pattern Analysis with Separate Visualizations")
                gr.Markdown("**Four separate plots: Clustering, Routes, Pressure Evolution, and Wind Evolution**")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        reduction_method = gr.Dropdown(
                            choices=['UMAP', 't-SNE', 'PCA'], 
                            value='UMAP' if UMAP_AVAILABLE else 't-SNE',
                            label="🔍 Dimensionality Reduction Method",
                            info="UMAP provides better global structure preservation"
                        )
                    with gr.Column(scale=1):
                        analyze_clusters_btn = gr.Button("🚀 Generate All Cluster Analyses", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        cluster_plot = gr.Plot(label="📊 Storm Clustering Analysis")
                    with gr.Column():
                        routes_plot = gr.Plot(label="🗺️ Clustered Storm Routes")
                
                with gr.Row():
                    with gr.Column():
                        pressure_plot = gr.Plot(label="🌡️ Pressure Evolution by Cluster")
                    with gr.Column():
                        wind_plot = gr.Plot(label="💨 Wind Speed Evolution by Cluster")
                
                with gr.Row():
                    cluster_stats = gr.Textbox(label="📈 Detailed Cluster Statistics", lines=15, max_lines=20)
                
                def run_separate_clustering_analysis(method):
                    try:
                        # Extract features for clustering
                        storm_features = extract_storm_features(typhoon_data)
                        if storm_features is None:
                            return None, None, None, None, "Error: Could not extract storm features"
                        
                        fig_cluster, fig_routes, fig_pressure, fig_wind, stats = create_separate_clustering_plots(
                            storm_features, typhoon_data, method.lower()
                        )
                        return fig_cluster, fig_routes, fig_pressure, fig_wind, stats
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        error_msg = f"Error: {str(e)}\n\nDetails:\n{error_details}"
                        return None, None, None, None, error_msg
                
                analyze_clusters_btn.click(
                    fn=run_separate_clustering_analysis,
                    inputs=[reduction_method],
                    outputs=[cluster_plot, routes_plot, pressure_plot, wind_plot, cluster_stats]
                )

            with gr.Tab("🗺️ Track Visualization"):
                with gr.Row():
                    start_year = gr.Number(label="Start Year", value=2020)
                    start_month = gr.Dropdown(label="Start Month", choices=list(range(1, 13)), value=1)
                    end_year = gr.Number(label="End Year", value=2025)
                    end_month = gr.Dropdown(label="End Month", choices=list(range(1, 13)), value=6)
                    enso_phase = gr.Dropdown(label="ENSO Phase", choices=['all', 'El Nino', 'La Nina', 'Neutral'], value='all')
                    typhoon_search = gr.Textbox(label="Typhoon Search")
                analyze_btn = gr.Button("Generate Tracks")
                tracks_plot = gr.Plot()
                typhoon_count = gr.Textbox(label="Number of Typhoons Displayed")
                analyze_btn.click(
                    fn=get_full_tracks,
                    inputs=[start_year, start_month, end_year, end_month, enso_phase, typhoon_search],
                    outputs=[tracks_plot, typhoon_count]
                )
            
            with gr.Tab("💨 Wind Analysis"):
                with gr.Row():
                    wind_start_year = gr.Number(label="Start Year", value=2020)
                    wind_start_month = gr.Dropdown(label="Start Month", choices=list(range(1, 13)), value=1)
                    wind_end_year = gr.Number(label="End Year", value=2024)
                    wind_end_month = gr.Dropdown(label="End Month", choices=list(range(1, 13)), value=6)
                    wind_enso_phase = gr.Dropdown(label="ENSO Phase", choices=['all', 'El Nino', 'La Nina', 'Neutral'], value='all')
                    wind_typhoon_search = gr.Textbox(label="Typhoon Search")
                wind_analyze_btn = gr.Button("Generate Wind Analysis")
                wind_scatter = gr.Plot()
                wind_regression_results = gr.Textbox(label="Wind Regression Results")
                wind_analyze_btn.click(
                    fn=get_wind_analysis,
                    inputs=[wind_start_year, wind_start_month, wind_end_year, wind_end_month, wind_enso_phase, wind_typhoon_search],
                    outputs=[wind_scatter, wind_regression_results]
                )
            
            with gr.Tab("🌡️ Pressure Analysis"):
                with gr.Row():
                    pressure_start_year = gr.Number(label="Start Year", value=2020)
                    pressure_start_month = gr.Dropdown(label="Start Month", choices=list(range(1, 13)), value=1)
                    pressure_end_year = gr.Number(label="End Year", value=2024)
                    pressure_end_month = gr.Dropdown(label="End Month", choices=list(range(1, 13)), value=6)
                    pressure_enso_phase = gr.Dropdown(label="ENSO Phase", choices=['all', 'El Nino', 'La Nina', 'Neutral'], value='all')
                    pressure_typhoon_search = gr.Textbox(label="Typhoon Search")
                pressure_analyze_btn = gr.Button("Generate Pressure Analysis")
                pressure_scatter = gr.Plot()
                pressure_regression_results = gr.Textbox(label="Pressure Regression Results")
                pressure_analyze_btn.click(
                    fn=get_pressure_analysis,
                    inputs=[pressure_start_year, pressure_start_month, pressure_end_year, pressure_end_month, pressure_enso_phase, pressure_typhoon_search],
                    outputs=[pressure_scatter, pressure_regression_results]
                )
            
            with gr.Tab("🌏 Longitude Analysis"):
                with gr.Row():
                    lon_start_year = gr.Number(label="Start Year", value=2020)
                    lon_start_month = gr.Dropdown(label="Start Month", choices=list(range(1, 13)), value=1)
                    lon_end_year = gr.Number(label="End Year", value=2020)
                    lon_end_month = gr.Dropdown(label="End Month", choices=list(range(1, 13)), value=6)
                    lon_enso_phase = gr.Dropdown(label="ENSO Phase", choices=['all', 'El Nino', 'La Nina', 'Neutral'], value='all')
                    lon_typhoon_search = gr.Textbox(label="Typhoon Search (Optional)")
                lon_analyze_btn = gr.Button("Generate Longitude Analysis")
                regression_plot = gr.Plot()
                slopes_text = gr.Textbox(label="Regression Slopes")
                lon_regression_results = gr.Textbox(label="Longitude Regression Results")
                lon_analyze_btn.click(
                    fn=get_longitude_analysis,
                    inputs=[lon_start_year, lon_start_month, lon_end_year, lon_end_month, lon_enso_phase, lon_typhoon_search],
                    outputs=[regression_plot, slopes_text, lon_regression_results]
                )
            
            with gr.Tab("🎬 Enhanced Track Animation"):
                gr.Markdown("## 🎥 High-Quality Storm Track Visualization (Atlantic & Taiwan Standards)")
                
                with gr.Row():
                    year_dropdown = gr.Dropdown(
                        label="Year",
                        choices=available_years,
                        value=available_years[-1] if available_years else "2024"
                    )
                    basin_dropdown = gr.Dropdown(
                        label="Basin",
                        choices=["All Basins", "WP - Western Pacific", "EP - Eastern Pacific", "NA - North Atlantic"],
                        value="All Basins"
                    )
                
                with gr.Row():
                    typhoon_dropdown = gr.Dropdown(label="Storm Selection (All Categories Including TD)")
                    standard_dropdown = gr.Dropdown(
                        label="🎌 Classification Standard",
                        choices=['atlantic', 'taiwan'], 
                        value='atlantic',
                        info="Atlantic: International standard | Taiwan: Local meteorological standard"
                    )
                
                generate_video_btn = gr.Button("🎬 Generate Enhanced Animation", variant="primary")
                video_output = gr.Video(label="Storm Track Animation")
                
                # Update storm options when year or basin changes
                for input_comp in [year_dropdown, basin_dropdown]:
                    input_comp.change(
                        fn=update_typhoon_options_enhanced,
                        inputs=[year_dropdown, basin_dropdown],
                        outputs=[typhoon_dropdown]
                    )
                
                # Generate video
                generate_video_btn.click(
                    fn=generate_enhanced_track_video,
                    inputs=[year_dropdown, typhoon_dropdown, standard_dropdown],
                    outputs=[video_output]
                )

            with gr.Tab("📊 Data Statistics & Insights"):
                gr.Markdown("## 📈 Comprehensive Dataset Analysis")
                
                # Create enhanced data summary
                try:
                    if len(typhoon_data) > 0:
                        # Storm category distribution
                        storm_cats = typhoon_data.groupby('SID')['USA_WIND'].max().apply(categorize_typhoon_enhanced)
                        cat_counts = storm_cats.value_counts()
                        
                        # Create distribution chart with enhanced colors
                        fig_dist = px.bar(
                            x=cat_counts.index,
                            y=cat_counts.values,
                            title="Storm Intensity Distribution (Including Tropical Depressions)",
                            labels={'x': 'Category', 'y': 'Number of Storms'},
                            color=cat_counts.index,
                            color_discrete_map=enhanced_color_map
                        )
                        
                        # Seasonal distribution
                        if 'ISO_TIME' in typhoon_data.columns:
                            seasonal_data = typhoon_data.copy()
                            seasonal_data['Month'] = seasonal_data['ISO_TIME'].dt.month
                            monthly_counts = seasonal_data.groupby(['Month', 'SID']).size().groupby('Month').size()
                            
                            fig_seasonal = px.bar(
                                x=monthly_counts.index,
                                y=monthly_counts.values,
                                title="Seasonal Storm Distribution",
                                labels={'x': 'Month', 'y': 'Number of Storms'},
                                color=monthly_counts.values,
                                color_continuous_scale='Viridis'
                            )
                        else:
                            fig_seasonal = None
                        
                        # Basin distribution
                        if 'SID' in typhoon_data.columns:
                            basin_data = typhoon_data['SID'].str[:2].value_counts()
                            fig_basin = px.pie(
                                values=basin_data.values,
                                names=basin_data.index,
                                title="Distribution by Basin"
                            )
                        else:
                            fig_basin = None
                        
                        with gr.Row():
                            gr.Plot(value=fig_dist)
                        
                        if fig_seasonal:
                            with gr.Row():
                                gr.Plot(value=fig_seasonal)
                        
                        if fig_basin:
                            with gr.Row():
                                gr.Plot(value=fig_basin)
                                
                except Exception as e:
                    gr.Markdown(f"Visualization error: {str(e)}")
                
                # Enhanced statistics - FIXED formatting
                total_storms = len(typhoon_data['SID'].unique()) if 'SID' in typhoon_data.columns else 0
                total_records = len(typhoon_data)
                
                if 'SEASON' in typhoon_data.columns:
                    try:
                        min_year = int(typhoon_data['SEASON'].min())
                        max_year = int(typhoon_data['SEASON'].max())
                        year_range = f"{min_year}-{max_year}"
                        years_covered = typhoon_data['SEASON'].nunique()
                    except (ValueError, TypeError):
                        year_range = "Unknown"
                        years_covered = 0
                else:
                    year_range = "Unknown"
                    years_covered = 0
                
                if 'SID' in typhoon_data.columns:
                    try:
                        basins_available = ', '.join(sorted(typhoon_data['SID'].str[:2].unique()))
                        avg_storms_per_year = total_storms / max(years_covered, 1)
                    except Exception:
                        basins_available = "Unknown"
                        avg_storms_per_year = 0
                else:
                    basins_available = "Unknown"
                    avg_storms_per_year = 0
                
                # TD specific statistics
                try:
                    if 'USA_WIND' in typhoon_data.columns:
                        td_storms = len(typhoon_data[typhoon_data['USA_WIND'] < 34]['SID'].unique())
                        ts_storms = len(typhoon_data[(typhoon_data['USA_WIND'] >= 34) & (typhoon_data['USA_WIND'] < 64)]['SID'].unique())
                        typhoon_storms = len(typhoon_data[typhoon_data['USA_WIND'] >= 64]['SID'].unique())
                        td_percentage = (td_storms / max(total_storms, 1)) * 100
                    else:
                        td_storms = ts_storms = typhoon_storms = 0
                        td_percentage = 0
                except Exception as e:
                    print(f"Error calculating TD statistics: {e}")
                    td_storms = ts_storms = typhoon_storms = 0
                    td_percentage = 0
                
                # Create statistics text safely
                stats_text = f"""
                ### 📊 Enhanced Dataset Summary:
                - **Total Unique Storms**: {total_storms:,}
                - **Total Track Records**: {total_records:,}  
                - **Year Range**: {year_range} ({years_covered} years)
                - **Basins Available**: {basins_available}
                - **Average Storms/Year**: {avg_storms_per_year:.1f}
                
                ### 🌪️ Storm Category Breakdown:
                - **Tropical Depressions**: {td_storms:,} storms ({td_percentage:.1f}%)
                - **Tropical Storms**: {ts_storms:,} storms
                - **Typhoons (C1-C5)**: {typhoon_storms:,} storms
                
                ### 🚀 Platform Capabilities:
                - **Complete TD Analysis** - First platform to include comprehensive TD tracking
                - **Dual Classification Systems** - Both Atlantic and Taiwan standards supported
                - **Advanced ML Clustering** - DBSCAN pattern recognition with separate visualizations
                - **Real-time Predictions** - Physics-based and optional CNN intensity forecasting
                - **2025 Data Ready** - Full compatibility with current season data
                - **Enhanced Animations** - Professional-quality storm track videos
                - **Multi-basin Analysis** - Comprehensive Pacific and Atlantic coverage
                
                ### 🔬 Research Applications:
                - Climate change impact studies
                - Seasonal forecasting research
                - Storm pattern classification
                - ENSO-typhoon relationship analysis
                - Intensity prediction model development
                - Cross-regional classification comparisons
                """
                gr.Markdown(stats_text)

        return demo
    except Exception as e:
        logging.error(f"Error creating Gradio interface: {e}")
        import traceback
        traceback.print_exc()
        # Create a minimal fallback interface
        return create_minimal_fallback_interface()

def create_minimal_fallback_interface():
    """Create a minimal fallback interface when main interface fails"""
    with gr.Blocks() as demo:
        gr.Markdown("# Enhanced Typhoon Analysis Platform")
        gr.Markdown("**Notice**: Loading with minimal interface due to data issues.")
        
        with gr.Tab("Status"):
            gr.Markdown("""
            ## Platform Status
            
            The application is running but encountered issues loading the full interface.
            This could be due to:
            - Data loading problems
            - Missing dependencies
            - Configuration issues
            
            ### Available Features:
            - Basic interface is functional
            - Error logs are being generated
            - System is ready for debugging
            
            ### Next Steps:
            1. Check the console logs for detailed error information
            2. Verify all required data files are accessible
            3. Ensure all dependencies are properly installed
            4. Try restarting the application
            """)
        
        with gr.Tab("Debug"):
            gr.Markdown("## Debug Information")
            
            def get_debug_info():
                debug_text = f"""
                Python Environment:
                - Working Directory: {os.getcwd()}
                - Data Path: {DATA_PATH}
                - UMAP Available: {UMAP_AVAILABLE}
                - CNN Available: {CNN_AVAILABLE}
                
                Data Status:
                - ONI Data: {'Loaded' if oni_data is not None else 'Failed'}
                - Typhoon Data: {'Loaded' if typhoon_data is not None else 'Failed'}
                - Merged Data: {'Loaded' if merged_data is not None else 'Failed'}
                
                File Checks:
                - ONI Path Exists: {os.path.exists(ONI_DATA_PATH)}
                - Typhoon Path Exists: {os.path.exists(TYPHOON_DATA_PATH)}
                """
                return debug_text
            
            debug_btn = gr.Button("Get Debug Info")
            debug_output = gr.Textbox(label="Debug Information", lines=15)
            debug_btn.click(fn=get_debug_info, outputs=debug_output)
    
    return demo

# Create and launch the interface
demo = create_interface()

if __name__ == "__main__":
    demo.launch(share=True)  # Enable sharing with public link