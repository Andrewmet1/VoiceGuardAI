"""
Petition Handler API

This module handles the petition signature storage and retrieval.
It provides endpoints for:
1. Submitting new petition signatures
2. Retrieving the current count of signatures
3. Admin dashboard data access
4. Email notifications for new signatures
"""

import os
import json
import time
import uuid
import csv
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from functools import wraps
from io import StringIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to the data storage file
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PETITION_FILE = os.path.join(DATA_DIR, 'petition_signatures.json')

# Admin credentials (in a real app, these would be stored securely)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD = 'voiceguard2025'

# You can change the admin credentials by modifying the values above
# After changing, restart the API service for changes to take effect

# Email configuration (replace with your actual email settings)
EMAIL_ENABLED = False  # Set to True to enable email notifications
EMAIL_SERVER = 'smtp.example.com'
EMAIL_PORT = 587
EMAIL_USERNAME = 'notifications@voiceguardai.com'
EMAIL_PASSWORD = 'your-email-password'
ADMIN_EMAIL = 'admin@voiceguardai.com'  # Where to send notifications

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def load_signatures():
    """Load existing signatures from file"""
    if not os.path.exists(PETITION_FILE):
        return {"signatures": [], "count": 0}
    
    try:
        with open(PETITION_FILE, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"signatures": [], "count": 0}

def save_signatures(data):
    """Save signatures to file"""
    with open(PETITION_FILE, 'w') as file:
        json.dump(data, file, indent=2)

def send_email_notification(signature):
    """Send email notification about new signature"""
    if not EMAIL_ENABLED:
        return
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USERNAME
    msg['To'] = ADMIN_EMAIL
    msg['Subject'] = 'New Petition Signature: ' + signature['name']
    
    # Create message body
    body = f"""
    New petition signature received:
    
    Name: {signature['name']}
    Email: {signature['email']}
    Country: {signature.get('country', 'Not provided')}
    Reason: {signature.get('reason', 'Not provided')}
    Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(signature['timestamp']))}
    IP: {signature.get('ip_address', 'Unknown')}
    
    Current signature count: {signature['count']}
    
    View all signatures in the admin dashboard.
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    # Send email in a separate thread to avoid blocking the response
    def send_email_thread():
        try:
            server = smtplib.SMTP(EMAIL_SERVER, EMAIL_PORT)
            server.starttls()
            server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"Email notification sent for {signature['name']}")
        except Exception as e:
            print(f"Failed to send email notification: {e}")
    
    threading.Thread(target=send_email_thread).start()

def require_admin(f):
    """Decorator to require admin authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.username != ADMIN_USERNAME or auth.password != ADMIN_PASSWORD:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/api/petition/count', methods=['GET'])
def get_signature_count():
    """Return the current number of petition signatures"""
    data = load_signatures()
    return jsonify({"count": data["count"]})

@app.route('/api/petition/sign', methods=['POST'])
def sign_petition():
    """Add a new signature to the petition"""
    data = load_signatures()
    
    # Get form data
    signature = {
        "id": str(uuid.uuid4()),  # Generate unique ID
        "name": request.form.get('name', ''),
        "email": request.form.get('email', ''),
        "country": request.form.get('country', ''),
        "reason": request.form.get('reason', ''),
        "timestamp": time.time(),
        "ip_address": request.remote_addr
    }
    
    # Basic validation
    if not signature["name"] or not signature["email"]:
        return jsonify({"success": False, "message": "Name and email are required"}), 400
    
    # Check for duplicate email
    emails = [sig["email"] for sig in data["signatures"]]
    if signature["email"] in emails:
        return jsonify({"success": False, "message": "You have already signed the petition"}), 400
    
    # Add the signature
    data["signatures"].append(signature)
    data["count"] = len(data["signatures"])
    
    # Save to file
    save_signatures(data)
    
    # Send email notification
    signature['count'] = data['count']  # Add count for the email
    send_email_notification(signature)
    
    return jsonify({
        "success": True, 
        "message": "Thank you for signing the petition!",
        "count": data["count"]
    })

@app.route('/api/petition/admin', methods=['GET'])
@require_admin
def get_admin_data():
    """Return all petition data for admin dashboard"""
    data = load_signatures()
    
    # Ensure each signature has the required fields in the expected format
    for sig in data.get('signatures', []):
        # Make sure timestamp is in seconds (Unix timestamp)
        if 'timestamp' in sig and isinstance(sig['timestamp'], (int, float)):
            # Already in the correct format
            pass
        elif 'date' in sig and not 'timestamp' in sig:
            # Convert date string to timestamp
            try:
                date_obj = datetime.strptime(sig['date'], '%Y-%m-%d %H:%M:%S')
                sig['timestamp'] = int(date_obj.timestamp())
            except:
                # If conversion fails, use current time
                sig['timestamp'] = int(time.time())
        else:
            # Default to current time if no timestamp or date
            sig['timestamp'] = int(time.time())
        
        # Ensure all required fields exist
        sig['name'] = sig.get('name', 'Anonymous')
        sig['email'] = sig.get('email', '')
        sig['country'] = sig.get('country', '')
        sig['reason'] = sig.get('reason', 'No reason provided')
    
    return jsonify(data)

@app.route('/api/petition/admin/data', methods=['GET'])
@require_admin
def get_admin_data_legacy():
    """Legacy endpoint for backward compatibility"""
    data = load_signatures()
    return jsonify(data)

@app.route('/api/petition/admin/emails', methods=['GET'])
@require_admin
def export_emails():
    """Export all emails as CSV"""
    data = load_signatures()
    
    # Create CSV in memory
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Email', 'Country', 'Date Signed'])
    
    for sig in data['signatures']:
        date_signed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sig['timestamp']))
        writer.writerow([sig['name'], sig['email'], sig.get('country', ''), date_signed])
    
    # Create response
    response = Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=petition_emails.csv',
            'Content-Type': 'text/csv'
        }
    )
    
    return response

@app.route('/api/petition/admin/delete/<signature_id>', methods=['DELETE'])
@require_admin
def delete_signature(signature_id):
    """Delete a signature from the petition"""
    data = load_signatures()
    
    # Find and remove the signature
    initial_count = len(data['signatures'])
    data['signatures'] = [sig for sig in data['signatures'] if sig.get('id') != signature_id]
    new_count = len(data['signatures'])
    
    if initial_count == new_count:
        return jsonify({"success": False, "message": "Signature not found"}), 404
    
    # Update count
    data['count'] = new_count
    
    # Save to file
    save_signatures(data)
    
    return jsonify({
        "success": True,
        "message": "Signature deleted successfully",
        "count": new_count
    })

@app.route('/api/petition/admin/notify-test', methods=['GET'])
@require_admin
def test_email_notification():
    """Test the email notification system"""
    if not EMAIL_ENABLED:
        return jsonify({
            "success": False,
            "message": "Email notifications are disabled. Set EMAIL_ENABLED to True to enable."
        })
    
    test_signature = {
        "id": "test-id",
        "name": "Test User",
        "email": "test@example.com",
        "country": "Test Country",
        "reason": "This is a test notification",
        "timestamp": time.time(),
        "ip_address": "127.0.0.1",
        "count": 123
    }
    
    send_email_notification(test_signature)
    
    return jsonify({
        "success": True,
        "message": "Test notification sent to " + ADMIN_EMAIL
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
