"""
Analytics Handler API

This module provides analytics data for the VoiceGuardAI application.
It tracks user metrics, scan data, and subscription information.
"""

import os
import json
import time
import random
from datetime import datetime, timedelta
from fastapi import APIRouter, Request, Depends, Body, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from .utils import parse_date
from .petition_handler import load_signatures, ADMIN_USERNAME, ADMIN_PASSWORD

router = APIRouter()

# Auth dependency
def verify_admin(username: str, password: str):
    if username != ADMIN_USERNAME or password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return True

# Path to the data storage files
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /VoiceGuardAI
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')  # /VoiceGuardAI/data

# Models
class UpdateUserRequest(BaseModel):
    device_id: str
    scan_count: int
    is_premium: bool
    subscription_expiry: Optional[str] = None

# Core data files
USER_DATA_FILE = os.path.join(PROJECT_ROOT, 'users.json')  # /VoiceGuardAI/users.json
ANALYTICS_FILE = os.path.join(DATA_DIR, 'analytics.json')  # /VoiceGuardAI/data/analytics.json
SCAN_DATA_FILE = os.path.join(DATA_DIR, 'scan_data.json')  # Legacy file, may not exist

# Helper function to calculate percentage change
def calculate_percentage_change(old_value, new_value):
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 100 if new_value > 0 else 0
    return round(((new_value - old_value) / old_value) * 100, 2)

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
        user_data = {}
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

@router.get("/admin")
def get_admin_analytics(username: str = Query(...), password: str = Query(...)):
    """Return all pre-processed analytics from analytics.json for admin dashboard."""
    verify_admin(username, password)

    try:
        with open(ANALYTICS_FILE, 'r') as f:
            analytics_data = json.load(f)

        petition_signatures = 0
        petition_completion = 0

        # Optional: add petition metrics if needed in Overview tab
        try:
            signatures = load_signatures()
            petition_signatures = signatures.get("count", 0)
            petition_completion = round(min(100, max(1, (petition_signatures / 50000) * 100)))
        except Exception as e:
            print(f"⚠️ Could not load petition data: {e}")

        # Calculate scan results from analytics data
        scan_results = {
            "ai_detected": analytics_data.get("total_ai_detections", 0),
            "human_detected": analytics_data.get("total_human_detections", 0)
        }

        # Format response
        return {
            "total_users": analytics_data.get("total_users", 0),
            "premium_users": analytics_data.get("premium_users", 0),
            "total_scans": analytics_data.get("total_scans", 0),
            "scan_results": scan_results,
            "revenue": analytics_data.get("revenue", {}),
            "daily_active_users": analytics_data.get("hourly_volumes", {}),  # Using hourly_volumes for now
            "monthly_active_users": {},  # To be implemented if needed
            "last_updated": time.time(),
            "petition_signatures": petition_signatures,
            "petition_completion": petition_completion
        }

    except Exception as e:
        print(f"Error loading analytics.json: {e}")
        raise HTTPException(status_code=500, detail="Failed to load analytics")

def load_analytics_data():
    """Load analytics data from file."""
    try:
        if not os.path.exists(ANALYTICS_FILE):
            return {
                'total_users': 0,
                'premium_users': 0,
                'total_scans': 0,
                'daily_active_users': {},
                'monthly_active_users': {},
                'scan_results': {
                    'ai_detected': 0,
                    'human_detected': 0
                },
                'revenue': {
                    'total': 0,
                    'monthly': {}
                },
                'app_installs': {
                    'android': 0,
                    'ios': 0
                },
                'last_updated': time.time()
            }
        with open(ANALYTICS_FILE, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f'Error loading analytics data: {e}')
        return {}

def save_analytics_data(data):
    """Save analytics data to file."""
    with open(ANALYTICS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def load_user_data():
    """Load user data from file."""
    try:
        if not os.path.exists(USER_DATA_FILE):
            return {}
        with open(USER_DATA_FILE, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f'Error loading user data: {e}')
        return {}

def save_user_data(data):
    """Save user data to file."""
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def load_scan_data():
    """Load scan data from file."""
    try:
        if not os.path.exists(SCAN_DATA_FILE):
            return {'scans': []}
        with open(SCAN_DATA_FILE, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f'Error loading scan data: {e}')
        return {'scans': []}

def save_scan_data(data):
    """Save scan data to file."""
    with open(SCAN_DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

@router.get('/dashboard')
def get_dashboard_data(username: str = Query(...), password: str = Query(...)):
    """Get all analytics data for the admin dashboard."""
    try:
        # Verify admin credentials
        verify_admin(username, password)
        
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
        
        # Format response
        return {
            "users": {
                "total": analytics_data['total_users'],
                "premium": analytics_data['premium_users'],
                "daily_active": daily_active_today,
                "daily_active_change": calculate_percentage_change(daily_active_yesterday, daily_active_today),
                "monthly_active": monthly_active,
                "conversion_rate": conversion_rate
            },
            "scans": {
                "total": analytics_data['total_scans'],
                "average_per_user": avg_scans_per_user,
                "ai_detection_rate": ai_detection_rate
            },
            "revenue": {
                "total": analytics_data['revenue']['total'],
                "monthly": monthly_revenue
            },
            "last_updated": analytics_data['last_updated']
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f'Error in get_dashboard_data: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Failed to get dashboard data: {str(e)}'
        )

@router.get('/users')
def get_user_data(username: str = Query(...), password: str = Query(...)):
    """Get user data for the admin dashboard."""
    verify_admin(username, password)
    try:
        user_data = load_user_data()
        # Convert dict to list format for frontend
        users_list = [{
            'device_id': device_id,
            'scan_count': data.get('scan_count', 0),
            'is_premium': data.get('is_premium', False),
            'subscription_expiry': data.get('subscription_expiry'),
            'last_scan': data.get('last_scan'),
            'platform': data.get('platform', 'unknown')
        } for device_id, data in user_data.items()]
        return {'users': users_list}
    except Exception as e:
        print(f'Error getting user data: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Failed to get user data: {str(e)}'
        )

@router.post('/users/update')
def update_user(data: UpdateUserRequest, username: str = Query(...), password: str = Query(...)):
    """Update user data in users.json"""
    verify_admin(username, password)

    try:
        user_data = load_user_data()
        if data.device_id not in user_data:
            raise HTTPException(status_code=404, detail="User not found")

        # Update values
        user_data[data.device_id]['scan_count'] = data.scan_count
        user_data[data.device_id]['is_premium'] = data.is_premium
        if data.subscription_expiry:
            try:
                # Validate date format
                datetime.strptime(data.subscription_expiry, '%Y-%m-%d')
                user_data[data.device_id]['subscription_expiry'] = data.subscription_expiry
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

        save_user_data(user_data)
        return {"status": "success", "user": user_data[data.device_id]}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f'Error updating user data: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Failed to update user data: {str(e)}'
        )

@router.get('/scans')
async def get_scan_data(username: str = Query(...), password: str = Query(...)):
    """Get scan data for the admin dashboard."""
    try:
        # Verify admin credentials
        verify_admin(username, password)
        
        scan_data = load_scan_data()
        
        # Limit to the most recent 100 scans for performance
        recent_scans = scan_data['scans'][-100:] if len(scan_data['scans']) > 100 else scan_data['scans']
        
        return {"scans": recent_scans, "total_count": len(scan_data['scans'])}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f'Error in get_scan_data: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Failed to get scan data: {str(e)}'
        )

@router.get('/daily-stats')
def get_daily_stats(username: str = Query(None), password: str = Query(None)):
    """Get daily statistics for public dashboard."""
    try:
        # Only verify admin if credentials provided
        if username and password:
            verify_admin(username, password)
        
        analytics_data = load_analytics_data()
        
        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Calculate daily stats
        daily_stats = {
            'total_scans_today': analytics_data.get('hourly_volumes', {}).get(today, 0),
            'ai_detected_today': analytics_data.get('total_ai_detections', 0),
            'human_detected_today': analytics_data.get('total_human_detections', 0),
            'active_users_today': len(analytics_data.get('daily_active_users', {})),
            'date': today
        }
        
        return daily_stats
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f'Error getting daily stats: {e}')
        raise HTTPException(status_code=500, detail=f'Failed to get daily stats: {str(e)}')

@router.get('/historical')
def get_historical_data(username: str, password: str, days: int = Query(30)):
    """Get historical data for charts."""
    try:
        # Verify admin credentials
        verify_admin(username, password)
        
        analytics_data = load_analytics_data()
        
        # Initialize fields if they don't exist
        if 'daily_ai_detections' not in analytics_data:
            analytics_data['daily_ai_detections'] = {}
        if 'daily_human_detections' not in analytics_data:
            analytics_data['daily_human_detections'] = {}
        if 'revenue' not in analytics_data:
            analytics_data['revenue'] = {'daily': {}}
        elif 'daily' not in analytics_data['revenue']:
            analytics_data['revenue']['daily'] = {}
        
        # Get date range from query parameters (default to last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate date range
        date_range = []
        current_date = start_date
        while current_date <= end_date:
            date_range.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        # Initialize data arrays
        historical_data = {
            "dates": date_range,
            "daily_active_users": [],
            "ai_detections": [],
            "human_detections": [],
            "revenue": []
        }
        
        # Fill data for each date
        for date in date_range:
            historical_data["daily_active_users"].append(analytics_data.get('daily_active_users', {}).get(date, 0))
            historical_data["ai_detections"].append(analytics_data.get('daily_ai_detections', {}).get(date, 0))
            historical_data["human_detections"].append(analytics_data.get('daily_human_detections', {}).get(date, 0))
            historical_data["revenue"].append(analytics_data.get('revenue', {}).get('daily', {}).get(date, 0))
        
        return historical_data
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f'Error in get_historical_data: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Failed to get historical data: {str(e)}'
        )

class UserData(BaseModel):
    device_id: str
    platform: Optional[str] = 'unknown'

@router.post('/record-user')
async def record_user(data: UserData):
    """Record a new user or user login."""
    try:
        device_id = data.device_id

        user_data = load_user_data()
        analytics_data = load_analytics_data()
        now = datetime.now()
        today = now.strftime('%Y-%m-%d')
        current_month = now.strftime('%Y-%m')

        # Create new user or update existing
        if device_id not in user_data:
            user_data[device_id] = {
                'is_premium': False,
                'scan_count': 0,
                'last_reset': current_month,
                'subscription_expiry': None,
                'last_active': now.isoformat(),
                'created_at': now.isoformat(),
                'platform': data.get('platform', 'unknown')
            }
            analytics_data['total_users'] += 1
            platform = data.get('platform', 'unknown')
            if platform in ['android', 'ios']:
                analytics_data['app_installs'][platform] = analytics_data['app_installs'].get(platform, 0) + 1
        else:
            user_data[device_id]['last_active'] = now.isoformat()

        # Count DAU and MAU
        analytics_data['daily_active_users'][today] = analytics_data['daily_active_users'].get(today, 0) + 1
        analytics_data['monthly_active_users'][current_month] = analytics_data['monthly_active_users'].get(current_month, 0) + 1
        analytics_data['last_updated'] = time.time()

        save_user_data(user_data)
        save_analytics_data(analytics_data)
        
        return {
            'status': 'success', 
            'user': user_data[device_id]
        }

    except Exception as e:
        print(f'Error recording user: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Failed to record user: {str(e)}'
        )

class ScanData(BaseModel):
    device_id: str
    scan_type: str
    result: Optional[str] = None

@router.post('/record-scan')
async def record_scan(data: ScanData):
    """Record a new scan."""
    try:
        device_id = data.device_id
        scan_type = data.scan_type
        
        result = data.get('result', 'unknown')
        if result not in ['ai', 'human', 'unknown']:
            return {'error': 'Invalid result value'}
        
        # Load all data
        analytics_data = load_analytics_data()
        user_data = load_user_data()
        scan_data = load_scan_data()
        now = time.time()
        
        # Record the scan
        new_scan = {
            'timestamp': now,
            'device_id': device_id,
            'result': result,
            'confidence': data.get('confidence', 0),
            'audio_length': data.get('audio_length', 0),
            'model_version': data.get('model_version', 'default')
        }
        
        # Initialize scans list if needed
        if 'scans' not in scan_data:
            scan_data['scans'] = []
        scan_data['scans'].append(new_scan)
        
        # Update user scan count and last active
        if device_id in user_data:
            user_data[device_id]['scan_count'] = user_data[device_id].get('scan_count', 0) + 1
            user_data[device_id]['last_active'] = datetime.fromtimestamp(now).isoformat()
        
        # Update scan results in analytics
        if result == 'ai':
            analytics_data['scan_results']['ai_detected'] = analytics_data['scan_results'].get('ai_detected', 0) + 1
        elif result == 'human':
            analytics_data['scan_results']['human_detected'] = analytics_data['scan_results'].get('human_detected', 0) + 1
        
        # Update last updated timestamp
        analytics_data['last_updated'] = now
        
        # Save all data
        save_analytics_data(analytics_data)
        save_user_data(user_data)
        save_scan_data(scan_data)
        
        return {
            'status': 'success',
            'scan_count': user_data[device_id]['scan_count'],
            'is_premium': user_data[device_id]['is_premium']
        }

    except Exception as e:
        print(f'Error recording scan: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Failed to record scan: {str(e)}'
        )

class SubscriptionData(BaseModel):
    device_id: str
    subscription_type: str = 'monthly'
    amount: float = 2.99

@router.post('/record-subscription')
async def record_subscription(data: SubscriptionData):
    """Record a new subscription or renewal."""
    try:
        user_id = data.device_id
        
        subscription_type = data.subscription_type
        if subscription_type not in ['monthly', 'yearly']:
            return {'error': 'Invalid subscription type'}
        
        # Load data
        analytics_data = load_analytics_data()
        user_data = load_user_data()
        now = datetime.now()
        
        # Update user subscription
        if user_id not in user_data:
            return {'error': 'User not found'}
        
        user_data[user_id].update({
            'is_premium': True,
            'subscription_type': subscription_type,
            'subscription_expiry': data.get('expiry'),
            'last_active': now.isoformat()
        })
        
        # Update analytics
        analytics_data['premium_users'] = sum(1 for u in user_data.values() if u.get('is_premium'))
        
        # Update revenue
        amount = data.amount
        analytics_data['revenue']['total'] = analytics_data['revenue'].get('total', 0) + amount
        
        current_month = now.strftime('%Y-%m')
        if 'monthly' not in analytics_data['revenue']:
            analytics_data['revenue']['monthly'] = {}
        analytics_data['revenue']['monthly'][current_month] = (
            analytics_data['revenue']['monthly'].get(current_month, 0) + amount
        )
        
        # Update last updated timestamp
        analytics_data['last_updated'] = time.time()
        
        # Save data
        save_analytics_data(analytics_data)
        save_user_data(user_data)
        
        return {
            'status': 'success',
            'user': user_data[user_id]
        }

    except Exception as e:
        print(f'Error recording subscription: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Failed to record subscription: {str(e)}'
        )

@router.post('/generate-sample-data')
async def generate_sample_data(username: str, password: str):
    """Generate sample data for testing the dashboard."""
    try:
        # Verify admin credentials
        verify_admin(username, password)
        
        # Generate sample data for the last 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        # Initialize data structures
        analytics_data = {
            'total_users': 0,
            'premium_users': 0,
            'total_scans': 0,
            'daily_active_users': {},
            'monthly_active_users': {},
            'scan_results': {
                'ai_detected': 0,
                'human_detected': 0
            },
            'revenue': {
                'total': 0,
                'monthly': {}
            },
            'app_installs': {
                'android': 0,
                'ios': 0
            },
            'last_updated': time.time()
        }
        
        # Generate random device IDs
        device_ids = [f'device_{i}' for i in range(100)]
        
        # Generate user data
        user_data = {}
        for device_id in device_ids:
            is_premium = random.random() < 0.2  # 20% premium users
            platform = random.choice(['android', 'ios'])
            created_time = time.mktime(start_date.timetuple())
            
            user_data[device_id] = {
                'is_premium': is_premium,
                'scan_count': 0,
                'last_reset': start_date.strftime('%Y-%m'),
                'subscription_expiry': None,
                'last_active': datetime.fromtimestamp(created_time).isoformat(),
                'created_at': datetime.fromtimestamp(created_time).isoformat(),
                'platform': platform
            }
            
            analytics_data['total_users'] += 1
            if is_premium:
                analytics_data['premium_users'] += 1
                user_data[device_id]['subscription_expiry'] = (end_date + timedelta(days=30)).isoformat()
            
            analytics_data['app_installs'][platform] += 1
    
        # Generate data for each day
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            month_str = current_date.strftime('%Y-%m')
            
            # Add new users for this day
            new_users_count = random.randint(5, 15)
            for _ in range(new_users_count):
                device_id = f'device_{len(user_data)}'
                is_premium = random.random() < 0.2  # 20% premium users
                platform = random.choice(['android', 'ios'])
                created_time = time.mktime(current_date.timetuple())
                
                user_data[device_id] = {
                    'is_premium': is_premium,
                    'scan_count': 0,
                    'last_reset': current_date.strftime('%Y-%m'),
                    'subscription_expiry': None,
                    'last_active': datetime.fromtimestamp(created_time).isoformat(),
                    'created_at': datetime.fromtimestamp(created_time).isoformat(),
                    'platform': platform
                }
                
                analytics_data['total_users'] += 1
                if is_premium:
                    analytics_data['premium_users'] += 1
                    user_data[device_id]['subscription_expiry'] = (current_date + timedelta(days=30)).isoformat()
                
                analytics_data['app_installs'][platform] += 1
            
            # Generate scans for this day
            active_users = random.sample(list(user_data.keys()), k=min(len(user_data), random.randint(10, 50)))
            for device_id in active_users:
                num_scans = random.randint(1, 5)
                for _ in range(num_scans):
                    scan_time = time.mktime(current_date.timetuple()) + random.randint(0, 86400)
                    result = random.choice(['ai', 'human'])
                    
                    scan = {
                        'device_id': device_id,
                        'timestamp': scan_time,
                        'result': result,
                        'confidence': random.uniform(0.6, 0.99),
                        'audio_length': random.randint(5, 60),
                        'model_version': 'v1.0'
                    }
                    
                    scan_data['scans'].append(scan)
                    analytics_data['total_scans'] += 1
                    if result == 'ai':
                        analytics_data['scan_results']['ai_detected'] += 1
                    else:
                        analytics_data['scan_results']['human_detected'] += 1
                    
                    user_data[device_id]['scan_count'] += 1
                    user_data[device_id]['last_active'] = datetime.fromtimestamp(scan_time).isoformat()
            
            # Update daily active users
            analytics_data['daily_active_users'][date_str] = len(active_users)
            
            # Update monthly active users
            if month_str not in analytics_data['monthly_active_users']:
                analytics_data['monthly_active_users'][month_str] = 0
            analytics_data['monthly_active_users'][month_str] = max(
                analytics_data['monthly_active_users'][month_str],
                len(active_users)
            )
            
            # Generate revenue for premium users
            if random.random() < 0.3:  # 30% chance of new premium user each day
                amount = 2.99  # Monthly subscription fee
                analytics_data['revenue']['total'] += amount
                if month_str not in analytics_data['revenue']['monthly']:
                    analytics_data['revenue']['monthly'][month_str] = 0
                analytics_data['revenue']['monthly'][month_str] += amount
            
            current_date += timedelta(days=1)
        
        # Save all data
        save_analytics_data(analytics_data)
        save_user_data(user_data)
        save_scan_data(scan_data)
        
        return {
            'status': 'success',
            'message': 'Sample data generated successfully'
        }
        
    except Exception as e:
        print(f'Error generating sample data: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Failed to generate sample data: {str(e)}'
        )

@router.get('/admin/summary')
async def get_analytics_admin_summary(username: str, password: str):
    """Return simplified analytics data for the frontend dashboard."""
    try:
        # Verify admin credentials
        verify_admin(username, password)
        
        # Load analytics data for accurate totals
        analytics_data = load_analytics_data()
        
        # Get user stats from analytics
        total_users = analytics_data.get('total_users', 0)
        premium_users = analytics_data.get('premium_users', 0)
        
        # Get active users from analytics
        now = datetime.now()
        daily_active = len(analytics_data.get('daily_active_users', {}))
        monthly_active = len(analytics_data.get('monthly_active_users', {}))
        
        # Get scan stats from analytics
        total_scans = analytics_data.get('total_scans', 0)
        scan_results = analytics_data.get('scan_results', {})
        ai_detected = scan_results.get('ai_detected', 0)
        ai_rate = (ai_detected / total_scans * 100) if total_scans > 0 else 0
        
        # Get revenue from analytics
        revenue_data = analytics_data.get('revenue', {})
        monthly_revenue = revenue_data.get('monthly', {}).get(time.strftime('%Y-%m'), 0)
        
        # Get daily stats for charts
        daily_stats = get_daily_stats()
        
        # Get petition stats from VoiceGuardAPI/api/data/petition_signatures.json
        try:
            from .petition_handler import load_signatures
            petition_data = load_signatures()  # Already returns dict with 'signatures' list
            signatures = petition_data.get('signatures', [])
            total_signatures = len(signatures)
            goal = 10000  # Target signatures
            completion = (total_signatures / goal * 100) if goal > 0 else 0
            today_signatures = sum(1 for s in signatures 
                                if (now - parse_date(s.get('timestamp', ''))).days <= 1)
        except Exception as e:
            print(f'Error loading petition data: {e}')
            total_signatures = 0
            completion = 0
            today_signatures = 0
        
        # Format response data
        response_data = {
            'total_users': total_users,
            'premium_users': premium_users,
            'daily_active_users': daily_active,
            'monthly_active_users': monthly_active,
            'total_scans': total_scans,
            'ai_detection_rate': ai_rate,
            'monthly_revenue': monthly_revenue,
            'daily_users': daily_stats.get('users', []),
            'daily_scans': daily_stats.get('scans', []),
            'labels': daily_stats.get('labels', []),
            'petition': {
                'total_signatures': total_signatures,
                'completion_percentage': completion,
                'goal': goal,
                'today_signatures': today_signatures
            }
        }
        
        return response_data
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f'Error in get_analytics_admin_summary: {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Failed to fetch analytics data: {str(e)}'
        )

# Expose router for import
class AnalyticsHandler:
    """Handler class for analytics functionality."""
    def __init__(self):
        self.initialize_data_files()

    def initialize_data_files(self):
        """Initialize data files with default structure if they don't exist."""
        # Create data directory if it doesn't exist
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        # Initialize analytics.json if it doesn't exist
        if not os.path.exists(ANALYTICS_FILE):
            default_analytics = {
                'total_users': 0,
                'premium_users': 0,
                'total_scans': 0,
                'daily_active_users': {},
                'monthly_active_users': {},
                'scan_results': {
                    'ai_detected': 0,
                    'human_detected': 0
                },
                'revenue': {
                    'total': 0,
                    'monthly': {}
                },
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            save_analytics_data(default_analytics)

        # Initialize users.json if it doesn't exist
        if not os.path.exists(USER_DATA_FILE):
            save_user_data({})

        # Initialize scan_data.json if it doesn't exist
        if not os.path.exists(SCAN_DATA_FILE):
            save_scan_data({'scans': []})