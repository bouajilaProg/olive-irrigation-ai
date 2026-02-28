// Default coordinates (Mahdia, Tunisia)
const defaultLat = 35.48525;
const defaultLon = 11.03083;

// Initialize Map
const map = L.map('map').setView([defaultLat, defaultLon], 16); // Zoomed in more for "land" view

// Add Satellite View (Google Hybrid/Satellite style using Esri)
const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
}).addTo(map);

// Add Label layer on top so users can see city names
const labelsLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20
}).addTo(map);

// Add Marker
let marker = L.marker([defaultLat, defaultLon], { draggable: true }).addTo(map);

// Function to update inputs
function updateCoords(lat, lon) {
    document.getElementById('lat').value = lat.toFixed(6);
    document.getElementById('lon').value = lon.toFixed(6);
}

// Set initial values
updateCoords(defaultLat, defaultLon);

// Update coordinates on map click
map.on('click', function (e) {
    const lat = e.latlng.lat;
    const lon = e.latlng.lng;
    marker.setLatLng([lat, lon]);
    updateCoords(lat, lon);
});

// Update coordinates on marker drag
marker.on('dragend', function (e) {
    const position = marker.getLatLng();
    updateCoords(position.lat, position.lng);
});

async function checkHealth() {
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;
    const overlay = document.getElementById('loader-overlay');
    const btn = document.getElementById('analyzeBtn');

    if (!lat || !lon) {
        alert("Please select a location on the map first.");
        return;
    }

    // UI State
    overlay.style.display = 'flex';
    btn.disabled = true;
    btn.style.opacity = '0.7';

    try {
        const response = await fetch(`/api/predict?lat=${lat}&lon=${lon}`);
        const data = await response.json();

        overlay.style.display = 'none';
        btn.disabled = false;
        btn.style.opacity = '1';

        if (response.ok) {
            showResults(data);
        } else {
            alert(`Error: ${data.detail || 'Location processing failed'}`);
        }
    } catch (err) {
        overlay.style.display = 'none';
        btn.disabled = false;
        btn.style.opacity = '1';
        alert("Error: Could not connect to the API server.");
    }

}

function showResults(data) {
    document.getElementById('map-view').style.display = 'none';
    document.getElementById('results-view').style.display = 'flex';

    // Update Gauge
    const fill = document.getElementById('gaugeFill');
    const text = document.getElementById('gaugeText');
    const badge = document.getElementById('stressLevel');
    const content = document.getElementById('resultContent');

    // Calculate dash offset (circumference = 2 * PI * R ≈ 2 * 3.14159 * 45 ≈ 282.7)
    const circumference = 282.7;
    const offset = circumference - (data.value / 100) * circumference;

    // Apply dynamics
    fill.style.strokeDashoffset = circumference; // Reset first
    setTimeout(() => {
        fill.style.strokeDashoffset = offset;
    }, 100);

    text.textContent = data.quality;

    badge.textContent = data.level;

    // Reset classes
    fill.classList.remove('level-CRITICAL', 'level-WARNING', 'level-STATUS');
    text.classList.remove('level-CRITICAL', 'level-WARNING', 'level-STATUS');
    badge.classList.remove('badge-CRITICAL', 'badge-WARNING', 'badge-STATUS');

    // Add classes
    fill.classList.add(`level-${data.level}`);
    text.classList.add(`level-${data.level}`);
    badge.classList.add(`badge-${data.level}`);

    content.innerHTML = `
        <p><strong>📡 Satellite Health (NDVI):</strong> ${data.ndvi}</p>
        <p><strong>🎯 Recommendation:</strong><br>${data.recommendation}</p>
        <p style="font-size: 0.85rem; color: #666; margin-top: 1rem;">
            Location: ${data.lat}, ${data.lon}
        </p>
    `;
}

function showMap() {
    document.getElementById('results-view').style.display = 'none';
    document.getElementById('map-view').style.display = 'grid';

    // Refresh map size since it was hidden
    setTimeout(() => {
        map.invalidateSize();
    }, 100);
}

