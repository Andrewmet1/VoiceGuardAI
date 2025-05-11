from waitress import serve
from api.analytics_handler import app

serve(app, host="127.0.0.1", port=5001)
