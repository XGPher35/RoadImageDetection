import folium
from geopy import Nominatim
import json
from pathlib import Path

DATA_FILE = Path("severity_points.json")
geolocator=Nominatim(user_agent="road_damage_mapper")

def get_location_name(lat, lon):
    try:
        location = geolocator.reverse(
            (lat, lon),
            language="en",
            exactly_one=True
        )

        if location:
            address = location.raw.get("address", {})

            return (
                address.get("road")
                or address.get("suburb")
                or address.get("village")
                or address.get("town")
                or address.get("city")
                or "Unknown Location"
            )

    except Exception as e:
        print("Reverse geocoding failed:", e)

    return "Unknown Location"

def save_detection(location, severity, label):
    """
    Append a new detection to JSON storage.
    """

    # Ensure file exists
    if not DATA_FILE.exists():
        DATA_FILE.write_text("[]")

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    lat = location["latitude"]
    lon = location["longitude"]

    place_name = get_location_name(lat, lon)

    new_entry = {
        "name": place_name,
        "location": {
            "lat": lat,
            "long": lon
        },
        "severity": severity,
        "label": label
    }
    if new_entry not in data:
        data.append(new_entry)


    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_detections():
    """
    Read all detections from JSON.
    """
    if not DATA_FILE.exists():
        return []

    with open(DATA_FILE, "r") as f:
        return json.load(f)


def generate_map(location=None, severity=None, label=None):
    """
    If location/severity/label provided then store first,
    then generate map using full dataset.
    """

    # store new detection
    if location and severity is not None and label:
        save_detection(location, severity, label)

    #load ALL detections
    detections = load_detections()

    print(f"TOTAL DETECTIONS LOADED: {len(detections)}")

    # Default center 
    if detections:
        latest = detections[-1]

        center_lat = latest["location"]["lat"]
        center_lon = latest["location"]["long"]
    else:
        # fallback if JSON is empty
        center_lat = 27.667
        center_lon = 85.44

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=40
    )

    # MARKERS

    for i, d in enumerate(detections):

        is_latest = (i == len(detections) - 1)

        if d["label"] == "Good":
            marker_color = "green"
        elif d["label"] == "Fair":
            marker_color = "lightgreen"
        elif d["label"] == "Poor":
            marker_color = "orange"
        else:
            marker_color = "red"

        if is_latest:
            icon = "star"
        else:
            icon = "info-sign"

        folium.Marker(
            [d["location"]["lat"], d["location"]["long"]],
            popup=f"""
            <b>{d['name']}</b><br>
            Severity: {d['severity']}<br>
            Category: {d['label']}
            {'<br><b>LATEST DETECTION</b>' if is_latest else ''}
            """,
            icon=folium.Icon(
                color=marker_color,
                icon=icon
            )
        ).add_to(m)

    m.save("map/severity_map.html")
    print("Severity map generated successfully.")