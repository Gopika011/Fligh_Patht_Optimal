flatpickr("#start-date", {
  dateFormat: "Y-m-d",
  minDate: "2024-12-01",
  maxDate: "2025-02-28",
  onChange: function (selectedDates) {
    if (selectedDates.length > 0) {
      startDateDisplay.textContent = formatDate(selectedDates[0]);
    }
  }
});

flatpickr("#end-date", {
  dateFormat: "Y-m-d",
  minDate: "2024-12-01",
  maxDate: "2025-02-28",
  onChange: function (selectedDates) {
    if (selectedDates.length > 0) {
      endDateDisplay.textContent = formatDate(selectedDates[0]);
    }
  }
});



// Function to format date as DD/MM/YYYY
function formatDate(date) {
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    return `${day}/${month}/${year}`;
}

// // Function to format time as HH:MM
// function formatTime(date) {
//     const hours = String(date.getHours()).padStart(2, '0');
//     const minutes = String(date.getMinutes()).padStart(2, '0');
//     return `${hours}:${minutes}`;
// }

// Set current date and time
const startDateInput = document.getElementById('start-date');
const endDateInput = document.getElementById('end-date');
const timeInput = document.getElementById('time');
const startDateDisplay = document.getElementById('current-start-date');
const endDateDisplay = document.getElementById('current-end-date');
const currentTimeDisplay = document.getElementById('current-time');
const flightNumberSelect = document.getElementById('flight-number-select'); 
const selectedFlightNumberDisplay = document.getElementById('selected-flight-number');

const now = new Date();
const formattedDate = formatDate(now);
// const formattedTime = formatTime(now);

startDateInput.value = now.toISOString().split('T')[0]; // Set input value in YYYY-MM-DD format
endDateInput.value = now.toISOString().split('T')[0]; // Set input value in YYYY-MM-DD format
// timeInput.value = formattedTime;


// show current date on inputs
startDateDisplay.textContent = `${formattedDate}`;
endDateDisplay.textContent = `${formattedDate}`;
selectedFlightNumberDisplay.textContent = flightNumberSelect.value;

// currentTimeDisplay.textContent = `${formattedTime}`;

// // Update the displayed time every minute
// setInterval(() => {
//     const now = new Date();
//     const formattedTime = formatTime(now);
//     currentTimeDisplay.textContent = `${formattedTime}`;
// }, 60000); // Update every minute


// timeInput.addEventListener("change", function () {
//     currentTimeDisplay.textContent = this.value;
// });


// inputs
document.querySelectorAll(".flight-section select").forEach(select => {
    select.addEventListener("change", function () {
        this.blur(); // Remove focus from the select input
        this.closest(".flight-section").blur(); // Remove focus from the container
        
        let selectedText = this.options[this.selectedIndex].text;
        // Update the specific display span based on the select's ID
        if (this.id === 'from') {
            document.getElementById('selected-from-airport').textContent = selectedText;
        } else if (this.id === 'to') {
            document.getElementById('selected-to-airport').textContent = selectedText;
        } else if (this.id === 'flight-number-select') {
            document.getElementById('selected-flight-number').textContent = selectedText;
        }

    });
});


document.querySelector(".start-date-selector").addEventListener("click", function () {
    startDateInput.showPicker(); // Opens date picker
});

document.querySelector(".end-date-selector").addEventListener("click", function () {
    endDateInput.showPicker(); // Opens date picker
});


// document.querySelector(".time-selector").addEventListener("click", function () {
//     timeInput.showPicker(); // Opens time picker
// });

Cesium.Ion.defaultAccessToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiIxZmM2YmIxMi1hZTMyLTRkNjQtODI0NC02ODQ2ZDZiM2JkM2QiLCJpZCI6MzEwMDg2LCJpYXQiOjE3NDkyOTc1NTh9.503N3_rd0bfAL-JOVdqm31i-E-pyRwU9FVylJHXCGq8';

    // Initialize the Cesium Viewer in the HTML element with the `cesiumContainer` ID.
    const viewer = new Cesium.Viewer('cesiumContainer', {
      terrain: Cesium.Terrain.fromWorldTerrain(),
    });

    // Fly the camera to San Francisco at the given longitude, latitude, and height.
   viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(-122.4175, 37.655, 400),
      orientation: {
        heading: Cesium.Math.toRadians(0.0),
        pitch: Cesium.Math.toRadians(-15.0),
      }
    });


// map




function displayPath(data){
    console.log(data);
    let route_data = data.route_data
    let optimal = data.optimal_path

    const routePoints = [];

    route_data.forEach(point => {
    const [lat, lon] = point.Position.split(',').map(coord => parseFloat(coord.trim()));
    const alt = parseFloat(point.Altitude);

    // or point.alt, based on your data structure
    routePoints.push(lon, lat, alt);
});
    console.log(routePoints)
    const polylines = new Cesium.PolylineCollection();
    polylines.add({
  positions : Cesium.Cartesian3.fromDegreesArrayHeights(routePoints),
  width : 5
});

    const average_points=[];
    const maxAltitude= 40000;
    const n = optimal.length;
   optimal.forEach((point, i) => {
    const lat = parseFloat(point.latitude);
    const lon = parseFloat(point.longitude);

    // Normalized time (0 to 1)
    const t = i / (n - 1);

    // Parabolic altitude: peaks at the middle
    const alt = maxAltitude * (4 * t * (1 - t));

    average_points.push(lon, lat, alt); // Cesium uses lon, lat, alt
});
    console.log(average_points)


  viewer.entities.add({
  polyline: {
    positions: Cesium.Cartesian3.fromDegreesArrayHeights(routePoints),
    width: 10,
    material: new Cesium.PolylineArrowMaterialProperty(Cesium.Color.GREEN),
    clampToGround: false // Optional: remove if you're clamping to terrain
  }
});
    viewer.entities.add({
  polyline: {
    positions: Cesium.Cartesian3.fromDegreesArrayHeights(average_points),
    width: 5,
    material: new Cesium.PolylineDashMaterialProperty({
      color: Cesium.Color.BLUE.withAlpha(0.6),
      dashLength: 16
    })
  }
});
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(routePoints[0],routePoints[1],40000),
      orientation: {
        heading: Cesium.Math.toRadians(0.0),
        pitch: Cesium.Math.toRadians(-15.0),
      }
    });

}

async function searchPath(){
    const searchButton = document.getElementById('search-button');
    const loadingState = document.getElementById('loading-state');

    const start_date = document.getElementById('start-date').value;
    const end_date = document.getElementById('end-date').value;
    const flight_number = document.getElementById('flight-number-select').value;

    console.log("Searching for:", { start_date, end_date, flight_number });

    const startDate = "2024-12-01"  
    const endDate = "2024-12-31"

    try{
        searchButton.disabled = true;
        loadingState.style.display = 'block';
        const response = await fetch(`/get-flight-path?start_date=${start_date}&end_date=${end_date}&flight_number=${flight_number}`);
        const data = await response.json();
        console.log(data)

        if (!response.ok) {
            throw new Error(data.error || "Unknown error occurred");
        }

        displayPath(data);
        
    }catch(error){
        alert(error.message);
    }finally{
        searchButton.disabled = false;
        loadingState.style.display = 'none';
    }

}