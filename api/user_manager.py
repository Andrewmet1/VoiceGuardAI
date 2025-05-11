import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_DB_PATH = os.path.join(PROJECT_ROOT, "users.json")
FREE_TIER_SCAN_LIMIT = 10

def load_users():
    """Load users from JSON file."""
    if not os.path.exists(USERS_DB_PATH):
        # Create empty users file if it doesn't exist
        try:
            with open(USERS_DB_PATH, "w") as f:
                json.dump({}, f)
        except Exception as e:
            logger.error(f"❌ Failed to create users.json: {str(e)}")
            return {}
    
    try:
        with open(USERS_DB_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load users.json: {str(e)}")
        return {}

def save_users(data):
    """Save users to JSON file."""
    try:
        with open(USERS_DB_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving users data: {str(e)}")
        return False

def should_reset_counter(last_reset):
    """Check if the scan counter should be reset (new month)."""
    if not last_reset:
        return True
        
    last_reset_date = datetime.strptime(last_reset, "%Y-%m")
    current_month = datetime.now().strftime("%Y-%m")
    return last_reset_date.strftime("%Y-%m") != current_month

def check_user_scan_limit(device_id, is_premium=False):
    """
    Check if a user has reached their scan limit.
    
    Args:
        device_id (str): Unique identifier for the user
        is_premium (bool): Whether the user has a premium subscription
        
    Returns:
        tuple: (can_scan, user_data)
            - can_scan (bool): Whether the user can perform a scan
            - user_data (dict): Updated user data
    """
    users = load_users()
    
    # Get or create user record
    if device_id not in users:
        user = {
            "device_id": device_id,
            "tier": "premium" if is_premium else "free",
            "scans_this_month": 0,
            "total_scans": 0,
            "last_reset": datetime.now().strftime("%Y-%m"),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_scan": None
        }
        users[device_id] = user
    else:
        user = users[device_id]
        
        # Update premium status if changed
        if is_premium and user["tier"] != "premium":
            user["tier"] = "premium"
    
    # Check if we need to reset the counter (new month)
    if should_reset_counter(user.get("last_reset")):
        user["scans_this_month"] = 0
        user["last_reset"] = datetime.now().strftime("%Y-%m")
    
    # Premium users have unlimited scans
    if user["tier"] == "premium":
        return True, user
    
    # Free users have a monthly limit
    if user["scans_this_month"] >= FREE_TIER_SCAN_LIMIT:
        return False, user
    
    return True, user

def increment_scan_count(device_id):
    """
    Increment the scan count for a user after a successful scan.
    
    Args:
        device_id (str): Unique identifier for the user
        
    Returns:
        dict: Updated user data
    """
    users = load_users()
    
    if device_id not in users:
        # This shouldn't happen if check_user_scan_limit is called first
        return None
    
    user = users[device_id]
    user["scans_this_month"] += 1
    user["total_scans"] += 1
    user["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    save_users(users)
    return user

def get_analytics():
    """
    Get analytics data for the admin dashboard.
    
    Returns:
        dict: Analytics data
    """
    users = load_users()
    
    total_users = len(users)
    free_users = sum(1 for u in users.values() if u["tier"] == "free")
    premium_users = total_users - free_users
    
    # Users who scanned this month
    current_month = datetime.now().strftime("%Y-%m")
    active_users = sum(1 for u in users.values() 
                      if u.get("last_reset") == current_month and u.get("scans_this_month", 0) > 0)
    
    # Total scans
    total_scans = sum(u.get("total_scans", 0) for u in users.values())
    
    # Conversion rate
    conversion_rate = round((premium_users / total_users * 100), 1) if total_users > 0 else 0
    
    return {
        "total_users": total_users,
        "free_users": free_users,
        "premium_users": premium_users,
        "active_users": active_users,
        "total_scans": total_scans,
        "conversion_rate": conversion_rate
    }
