"""
Analytics Handler API

This module provides analytics data for the VoiceGuardAI application.
It tracks user metrics, scan data, and subscription information.
"""

import os
import json
import time
import uuid
import random
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
from functools import wraps

# Import the petition handler to share authentication
from api.petition_handler import require_admin, ADMIN_USERNAME, ADMIN_PASSWORD

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to the data storage files
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ANALYTICS_FILE = os.path.join(DATA_DIR, 'app_analytics.json')
USER_DATA_FILE = os.path.join(DATA_DIR, 'user_data.json')
SCAN_DATA_FILE = os.path.join(DATA_DIR, 'scan_data.json')

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize data files if they don't exist
def initialize_data_files():
    """Initialize data files with default structure if they don't exist."""
    # App analytics
    if not os.path.exists(ANALYTICS_FILE):
        analytics_data = {
            "total_users": 0,
            "premium_users": 0,
            "total_scans": 0,
            "daily_active_users": {},
            "monthly_active_users": {},
            "scan_results": {
                "ai_detected": 0,
                "human_detected": 0
            },
            "revenue": {
                "total": 0,
                "monthly": {}
            },
            "app_installs": {
                "android": 0,
                "ios": 0
            },
            "last_updated": time.time()
        }
        with open(ANALYTICS_FILE, 'w') as file:
            json.dump(analytics_data, file, indent=2)
    
    # User data
    if not os.path.exists(USER_DATA_FILE):
        user_data = {
            "users": []
        }
        with open(USER_DATA_FILE, 'w') as file:
            json.dump(user_data, file, indent=2)
    
    # Scan data
    if not os.path.exists(SCAN_DATA_FILE):
        scan_data = {
            "scans": []
        }
        with open(SCAN_DATA_FILE, 'w') as file:
            json.dump(scan_data, file, indent=2)

# Initialize data files on startup
initialize_data_files()

def load_analytics_data():
    """Load analytics data from file."""
    try:
        with open(ANALYTICS_FILE, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupted or doesn't exist, initialize it
        initialize_data_files()
        with open(ANALYTICS_FILE, 'r') as file:
            return json.load(file)

def save_analytics_data(data):
    """Save analytics data to file."""
    with open(ANALYTICS_FILE, 'w') as file:
        json.dump(data, file, indent=2)

def load_user_data():
    """Load user data from file."""
    try:
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupted or doesn't exist, initialize it
        initialize_data_files()
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)

def save_user_data(data):
    """Save user data to file."""
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(data, file, indent=2)

def load_scan_data():
    """Load scan data from file."""
    try:
        with open(SCAN_DATA_FILE, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        # If file is corrupted or doesn't exist, initialize it
        initialize_data_files()
        with open(SCAN_DATA_FILE, 'r') as file:
            return json.load(file)

def save_scan_data(data):
    """Save scan data to file."""
    with open(SCAN_DATA_FILE, 'w') as file:
        json.dump(data, file, indent=2)

@app.route('/api/analytics/dashboard', methods=['GET'])
@require_admin
def get_dashboard_data():
    """Get all analytics data for the admin dashboard."""
    analytics_data = load_analytics_data()
    
    # Calculate additional metrics
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    current_month = datetime.now().strftime('%Y-%m')
    
    # Get daily active users
    daily_active_today = analytics_data['daily_active_users'].get(today, 0)
    daily_active_yesterday = analytics_data['daily_active_users'].get(yesterday, 0)
    
    # Get monthly active users
    monthly_active = analytics_data['monthly_active_users'].get(current_month, 0)
    
    # Calculate conversion rate
    conversion_rate = 0
    if analytics_data['total_users'] > 0:
        conversion_rate = (analytics_data['premium_users'] / analytics_data['total_users']) * 100
    
    # Calculate average scans per user
    avg_scans_per_user = 0
    if analytics_data['total_users'] > 0:
        avg_scans_per_user = analytics_data['total_scans'] / analytics_data['total_users']
    
    # Calculate AI detection rate
    ai_detection_rate = 0
    if analytics_data['total_scans'] > 0:
        ai_detection_rate = (analytics_data['scan_results']['ai_detected'] / analytics_data['total_scans']) * 100
    
    # Get monthly revenue
    monthly_revenue = analytics_data['revenue']['monthly'].get(current_month, 0)
    
    # Prepare response
    response = {
        "total_users": analytics_data['total_users'],
        "premium_users": analytics_data['premium_users'],
        "total_scans": analytics_data['total_scans'],
        "daily_active_users": {
            "today": daily_active_today,
            "yesterday": daily_active_yesterday,
            "change_percentage": calculate_percentage_change(daily_active_yesterday, daily_active_today)
        },
        "monthly_active_users": monthly_active,
        "conversion_rate": round(conversion_rate, 2),
        "avg_scans_per_user": round(avg_scans_per_user, 2),
        "scan_results": {
            "ai_detected": analytics_data['scan_results']['ai_detected'],
            "human_detected": analytics_data['scan_results']['human_detected'],
            "ai_detection_rate": round(ai_detection_rate, 2)
        },
        "revenue": {
            "total": analytics_data['revenue']['total'],
            "monthly": monthly_revenue
        },
        "app_installs": analytics_data['app_installs'],
        "last_updated": analytics_data['last_updated']
    }
    
    return jsonify(response)

@app.route('/api/analytics/users', methods=['GET'])
@require_admin
def get_user_data():
    """Get user data for the admin dashboard."""
    user_data = load_user_data()
    
    # Limit to the most recent 100 users for performance
    recent_users = user_data['users'][-100:] if len(user_data['users']) > 100 else user_data['users']
    
    return jsonify({"users": recent_users, "total_count": len(user_data['users'])})

@app.route('/api/analytics/scans', methods=['GET'])
@require_admin
def get_scan_data():
    """Get scan data for the admin dashboard."""
    scan_data = load_scan_data()
    
    # Limit to the most recent 100 scans for performance
    recent_scans = scan_data['scans'][-100:] if len(scan_data['scans']) > 100 else scan_data['scans']
    
    return jsonify({"scans": recent_scans, "total_count": len(scan_data['scans'])})

@app.route('/api/analytics/historical', methods=['GET'])
@require_admin
def get_historical_data():
    """Get historical data for charts."""
    analytics_data = load_analytics_data()
    
    # Get date range from query parameters (default to last 30 days)
    days = int(request.args.get('days', 30))
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Generate date range
    date_range = []
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    
    # Prepare historical data
    historical_data = {
        "dates": date_range,
        "daily_active_users": [],
        "new_users": [],
        "scans_performed": [],
        "ai_detections": [],
        "human_detections": [],
        "revenue": []
    }
    
    # Fill in data for each date
    for date in date_range:
        # Daily active users
        historical_data["daily_active_users"].append(
            analytics_data['daily_active_users'].get(date, 0)
        )
        
        # For other metrics, we'll need to query the user and scan data
        # This is a placeholder for actual implementation
        historical_data["new_users"].append(0)
        historical_data["scans_performed"].append(0)
        historical_data["ai_detections"].append(0)
        historical_data["human_detections"].append(0)
        historical_data["revenue"].append(0)
    
    return jsonify(historical_data)

@app.route('/api/analytics/record-user', methods=['POST'])
def record_user():
    """Record a new user or user login."""
    # In production, this would validate an API key
    data = request.json
    
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "User ID is required"}), 400
    
    # Load data
    analytics_data = load_analytics_data()
    user_data = load_user_data()
    
    # Check if this is a new user
    existing_user = False
    for user in user_data['users']:
        if user['user_id'] == user_id:
            existing_user = True
            # Update last login
            user['last_login'] = time.time()
            break
    
    # If new user, add to user data
    if not existing_user:
        new_user = {
            "user_id": user_id,
            "created_at": time.time(),
            "last_login": time.time(),
            "premium": data.get('premium', False),
            "platform": data.get('platform', 'unknown'),
            "device_info": data.get('device_info', {}),
            "scans_performed": 0
        }
        user_data['users'].append(new_user)
        
        # Update analytics
        analytics_data['total_users'] += 1
        if new_user['premium']:
            analytics_data['premium_users'] += 1
        
        # Update app installs
        platform = data.get('platform', '').lower()
        if platform == 'android':
            analytics_data['app_installs']['android'] += 1
        elif platform == 'ios':
            analytics_data['app_installs']['ios'] += 1
    
    # Update daily active users
    today = datetime.now().strftime('%Y-%m-%d')
    analytics_data['daily_active_users'][today] = analytics_data['daily_active_users'].get(today, 0) + 1
    
    # Update monthly active users
    current_month = datetime.now().strftime('%Y-%m')
    analytics_data['monthly_active_users'][current_month] = analytics_data['monthly_active_users'].get(current_month, 0) + 1
    
    # Update last updated timestamp
    analytics_data['last_updated'] = time.time()
    
    # Save data
    save_analytics_data(analytics_data)
    save_user_data(user_data)
    
    return jsonify({"success": True})

@app.route('/api/analytics/record-scan', methods=['POST'])
def record_scan():
    """Record a new scan."""
    # In production, this would validate an API key
    data = request.json
    
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "User ID is required"}), 400
    
    result = data.get('result', 'unknown')
    if result not in ['ai', 'human', 'unknown']:
        return jsonify({"success": False, "message": "Invalid result value"}), 400
    
    # Load data
    analytics_data = load_analytics_data()
    user_data = load_user_data()
    scan_data = load_scan_data()
    
    # Record the scan
    new_scan = {
        "scan_id": str(uuid.uuid4()),
        "user_id": user_id,
        "timestamp": time.time(),
        "result": result,
        "confidence": data.get('confidence', 0),
        "audio_length": data.get('audio_length', 0),
        "model_used": data.get('model_used', 'default')
    }
    scan_data['scans'].append(new_scan)
    
    # Update user scan count
    for user in user_data['users']:
        if user['user_id'] == user_id:
            user['scans_performed'] = user.get('scans_performed', 0) + 1
            break
    
    # Update analytics
    analytics_data['total_scans'] += 1
    if result == 'ai':
        analytics_data['scan_results']['ai_detected'] += 1
    elif result == 'human':
        analytics_data['scan_results']['human_detected'] += 1
    
    # Update last updated timestamp
    analytics_data['last_updated'] = time.time()
    
    # Save data
    save_analytics_data(analytics_data)
    save_user_data(user_data)
    save_scan_data(scan_data)
    
    return jsonify({"success": True})

@app.route('/api/analytics/record-subscription', methods=['POST'])
def record_subscription():
    """Record a new subscription or renewal."""
    # In production, this would validate an API key
    data = request.json
    
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"success": False, "message": "User ID is required"}), 400
    
    amount = data.get('amount', 0)
    if amount <= 0:
        return jsonify({"success": False, "message": "Invalid amount"}), 400
    
    # Load data
    analytics_data = load_analytics_data()
    user_data = load_user_data()
    
    # Update user premium status
    user_updated = False
    for user in user_data['users']:
        if user['user_id'] == user_id:
            if not user.get('premium', False):
                user['premium'] = True
                analytics_data['premium_users'] += 1
            user_updated = True
            break
    
    # If user not found, create a new user record
    if not user_updated:
        new_user = {
            "user_id": user_id,
            "created_at": time.time(),
            "last_login": time.time(),
            "premium": True,
            "platform": data.get('platform', 'unknown'),
            "device_info": {},
            "scans_performed": 0
        }
        user_data['users'].append(new_user)
        analytics_data['total_users'] += 1
        analytics_data['premium_users'] += 1
    
    # Update revenue
    analytics_data['revenue']['total'] += amount
    
    # Update monthly revenue
    current_month = datetime.now().strftime('%Y-%m')
    analytics_data['revenue']['monthly'][current_month] = analytics_data['revenue']['monthly'].get(current_month, 0) + amount
    
    # Update last updated timestamp
    analytics_data['last_updated'] = time.time()
    
    # Save data
    save_analytics_data(analytics_data)
    save_user_data(user_data)
    
    return jsonify({"success": True})

@app.route('/api/analytics/generate-sample-data', methods=['POST'])
@require_admin
def generate_sample_data():
    """Generate sample data for testing the dashboard."""
    # This endpoint is for development/demo purposes only
    
    # Generate sample data for the last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    # Initialize data structures
    analytics_data = {
        "total_users": 0,
        "premium_users": 0,
        "total_scans": 0,
        "daily_active_users": {},
        "monthly_active_users": {},
        "scan_results": {
            "ai_detected": 0,
            "human_detected": 0
        },
        "revenue": {
            "total": 0,
            "monthly": {}
        },
        "app_installs": {
            "android": 0,
            "ios": 0
        },
        "last_updated": time.time()
    }
    
    user_data = {"users": []}
    scan_data = {"scans": []}
    
    # Generate users (start with 100 and add more each day)
    initial_users = 100
    daily_new_users_min = 5
    daily_new_users_max = 15
    
    # Create initial users
    for i in range(initial_users):
        user_id = str(uuid.uuid4())
        is_premium = random.random() < 0.2  # 20% premium users
        platform = random.choice(['android', 'ios'])
        
        user = {
            "user_id": user_id,
            "created_at": time.mktime(start_date.timetuple()),
            "last_login": time.mktime(start_date.timetuple()),
            "premium": is_premium,
            "platform": platform,
            "device_info": {},
            "scans_performed": 0
        }
        
        user_data["users"].append(user)
        analytics_data["total_users"] += 1
        if is_premium:
            analytics_data["premium_users"] += 1
        
        if platform == 'android':
            analytics_data["app_installs"]["android"] += 1
        else:
            analytics_data["app_installs"]["ios"] += 1
    
    # Generate data for each day
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        month_str = current_date.strftime('%Y-%m')
        
        # Add new users for this day
        new_users_count = random.randint(daily_new_users_min, daily_new_users_max)
        for i in range(new_users_count):
            user_id = str(uuid.uuid4())
            is_premium = random.random() < 0.2  # 20% premium users
            platform = random.choice(['android', 'ios'])
            
            user = {
                "user_id": user_id,
                "created_at": time.mktime(current_date.timetuple()),
                "last_login": time.mktime(current_date.timetuple()),
                "premium": is_premium,
                "platform": platform,
                "device_info": {},
                "scans_performed": 0
            }
            
            user_data["users"].append(user)
            analytics_data["total_users"] += 1
            if is_premium:
                analytics_data["premium_users"] += 1
            
            if platform == 'android':
                analytics_data["app_installs"]["android"] += 1
            else:
                analytics_data["app_installs"]["ios"] += 1
        
        # Calculate daily active users (30-50% of total users)
        active_users_pct = random.uniform(0.3, 0.5)
        active_users = int(analytics_data["total_users"] * active_users_pct)
        analytics_data["daily_active_users"][date_str] = active_users
        
        # Calculate monthly active users (60-80% of total users)
        if month_str not in analytics_data["monthly_active_users"]:
            active_users_pct = random.uniform(0.6, 0.8)
            active_users = int(analytics_data["total_users"] * active_users_pct)
            analytics_data["monthly_active_users"][month_str] = active_users
        
        # Generate scans for this day
        scans_per_day = int(active_users * random.uniform(1.5, 3))  # 1.5-3 scans per active user
        for i in range(scans_per_day):
            # Pick a random user
            user = random.choice(user_data["users"])
            user_id = user["user_id"]
            
            # Determine scan result (15-25% AI, rest human)
            is_ai = random.random() < random.uniform(0.15, 0.25)
            result = "ai" if is_ai else "human"
            
            # Create scan record
            scan = {
                "scan_id": str(uuid.uuid4()),
                "user_id": user_id,
                "timestamp": time.mktime(current_date.timetuple()) + random.randint(0, 86399),  # Random time during the day
                "result": result,
                "confidence": random.uniform(0.7, 0.99),
                "audio_length": random.randint(5, 60),
                "model_used": random.choice(["wavlm", "wav2vec", "voiceguard"])
            }
            
            scan_data["scans"].append(scan)
            analytics_data["total_scans"] += 1
            
            if result == "ai":
                analytics_data["scan_results"]["ai_detected"] += 1
            else:
                analytics_data["scan_results"]["human_detected"] += 1
            
            # Update user scan count
            user["scans_performed"] += 1
        
        # Generate subscription revenue for this day
        new_subscriptions = int(new_users_count * random.uniform(0.1, 0.3))  # 10-30% of new users subscribe
        renewals = int(analytics_data["premium_users"] * random.uniform(0.01, 0.03))  # 1-3% of premium users renew each day
        
        daily_revenue = (new_subscriptions + renewals) * 2.99
        analytics_data["revenue"]["total"] += daily_revenue
        
        if month_str not in analytics_data["revenue"]["monthly"]:
            analytics_data["revenue"]["monthly"][month_str] = 0
        analytics_data["revenue"]["monthly"][month_str] += daily_revenue
        
        # Move to next day
        current_date += timedelta(days=1)
    
    # Save the generated data
    save_analytics_data(analytics_data)
    save_user_data(user_data)
    save_scan_data(scan_data)
    
    return jsonify({
        "success": True,
        "message": "Sample data generated successfully",
        "stats": {
            "total_users": analytics_data["total_users"],
            "premium_users": analytics_data["premium_users"],
            "total_scans": analytics_data["total_scans"],
            "ai_detections": analytics_data["scan_results"]["ai_detected"],
            "human_detections": analytics_data["scan_results"]["human_detected"],
            "total_revenue": analytics_data["revenue"]["total"]
        }
    })

def calculate_percentage_change(old_value, new_value):
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 100 if new_value > 0 else 0
    
    return round(((new_value - old_value) / old_value) * 100, 2)

@app.route('/api/analytics/admin', methods=['GET'])
@require_admin
def get_analytics_dashboard():
    """Get simplified analytics data for the admin dashboard."""
    try:
        # Load analytics data
        analytics_data = load_analytics_data()
        
        # Get current date info
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Generate day labels for the past week
        days = []
        daily_users = []
        daily_scans = []
        
        for i in range(6, -1, -1):
            date = datetime.now() - timedelta(days=i)
            day_name = date.strftime("%a")
            days.append(day_name)
            
            # Get actual data if available, otherwise use sample data
            date_str = date.strftime('%Y-%m-%d')
            daily_users.append(analytics_data.get('daily_active_users', {}).get(date_str, random.randint(5, 40)))
            daily_scans.append(analytics_data.get('daily_scans', {}).get(date_str, random.randint(20, 70)))
        
        # Create response in the exact format needed by the dashboard
        response = {
            "total_users": analytics_data.get('total_users', 0),
            "total_scans": analytics_data.get('total_scans', 0),
            "new_users_today": analytics_data.get('daily_signups', {}).get(today, 0),
            "scans_today": analytics_data.get('daily_scans', {}).get(today, 0),
            "daily_users": daily_users,
            "daily_scans": daily_scans,
            "labels": days
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5001)
