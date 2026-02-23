"""
Demo tool implementations for EdgeAction Concierge.
Mock executors that simulate real-world tool actions.
"""

import json
from datetime import datetime


def get_weather(location):
    """Get current weather for a location."""
    # Mock weather data
    weather_data = {
        "San Francisco": {"temp": 62, "condition": "Foggy", "humidity": 78},
        "London": {"temp": 54, "condition": "Cloudy", "humidity": 85},
        "Tokyo": {"temp": 71, "condition": "Sunny", "humidity": 55},
        "New York": {"temp": 45, "condition": "Clear", "humidity": 40},
        "Paris": {"temp": 58, "condition": "Partly Cloudy", "humidity": 65},
        "Berlin": {"temp": 50, "condition": "Overcast", "humidity": 72},
        "Miami": {"temp": 82, "condition": "Sunny", "humidity": 70},
        "Chicago": {"temp": 38, "condition": "Windy", "humidity": 45},
        "Seattle": {"temp": 52, "condition": "Rainy", "humidity": 88},
    }
    data = weather_data.get(location, {"temp": 65, "condition": "Clear", "humidity": 50})
    return {
        "location": location,
        "temperature_f": data["temp"],
        "condition": data["condition"],
        "humidity": f"{data['humidity']}%",
        "source": "mock_weather_api",
    }


def set_alarm(hour, minute=0):
    """Set an alarm for a given time."""
    time_str = f"{int(hour):02d}:{int(minute):02d}"
    return {
        "status": "alarm_set",
        "time": time_str,
        "message": f"Alarm set for {time_str}",
    }


def send_message(recipient, message):
    """Send a message to a contact."""
    return {
        "status": "sent",
        "recipient": recipient,
        "message": message,
        "timestamp": datetime.now().isoformat(),
    }


def create_reminder(title, time_str):
    """Create a reminder with a title and time."""
    return {
        "status": "reminder_created",
        "title": title,
        "time": time_str,
        "message": f"Reminder '{title}' set for {time_str}",
    }


def search_contacts(query):
    """Search for a contact by name."""
    contacts = {
        "Bob": {"name": "Bob Smith", "phone": "+1-555-0101", "email": "bob@example.com"},
        "Alice": {"name": "Alice Johnson", "phone": "+1-555-0102", "email": "alice@example.com"},
        "Tom": {"name": "Tom Wilson", "phone": "+1-555-0103", "email": "tom@example.com"},
        "Sarah": {"name": "Sarah Davis", "phone": "+1-555-0104", "email": "sarah@example.com"},
        "Jake": {"name": "Jake Brown", "phone": "+1-555-0105", "email": "jake@example.com"},
        "Lisa": {"name": "Lisa Taylor", "phone": "+1-555-0106", "email": "lisa@example.com"},
        "Dave": {"name": "Dave Anderson", "phone": "+1-555-0107", "email": "dave@example.com"},
        "Emma": {"name": "Emma Martinez", "phone": "+1-555-0108", "email": "emma@example.com"},
        "John": {"name": "John Lee", "phone": "+1-555-0109", "email": "john@example.com"},
    }
    # Fuzzy match
    for key, val in contacts.items():
        if query.lower() in key.lower() or query.lower() in val["name"].lower():
            return {"found": True, "contact": val}
    return {"found": False, "query": query, "message": f"No contact found for '{query}'"}


def play_music(song):
    """Play a song or playlist."""
    return {
        "status": "playing",
        "song": song,
        "message": f"Now playing: {song}",
    }


def set_timer(minutes):
    """Set a countdown timer."""
    return {
        "status": "timer_started",
        "minutes": int(minutes),
        "message": f"Timer set for {int(minutes)} minutes",
    }


# Registry: maps tool name → executor function
TOOL_EXECUTORS = {
    "get_weather": get_weather,
    "set_alarm": set_alarm,
    "send_message": send_message,
    "create_reminder": create_reminder,
    "search_contacts": search_contacts,
    "play_music": play_music,
    "set_timer": set_timer,
}


def execute_tool_call(name, arguments):
    """Execute a tool call by name with given arguments."""
    executor = TOOL_EXECUTORS.get(name)
    if executor is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return executor(**arguments)
    except TypeError as e:
        return {"error": f"Invalid arguments for {name}: {e}"}
