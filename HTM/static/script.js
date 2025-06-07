// Function to format date as DD/MM/YYYY
function formatDate(date) {
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    return `${day}/${month}/${year}`;
}

// Function to format time as HH:MM
function formatTime(date) {
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${hours}:${minutes}`;
}

// Set current date and time
const startDateInput = document.getElementById('start-date');
const endDateInput = document.getElementById('end-date');
const timeInput = document.getElementById('time');
const startDateDisplay = document.getElementById('current-start-date');
const endDateDisplay = document.getElementById('current-end-date');
const currentTimeDisplay = document.getElementById('current-time');

const now = new Date();
const formattedDate = formatDate(now);
const formattedTime = formatTime(now);

startDateInput.value = now.toISOString().split('T')[0]; // Set input value in YYYY-MM-DD format
endDateInput.value = now.toISOString().split('T')[0]; // Set input value in YYYY-MM-DD format
timeInput.value = formattedTime;

startDateDisplay.textContent = `${formattedDate}`;
endDateDisplay.textContent = `${formattedDate}`;
currentTimeDisplay.textContent = `${formattedTime}`;

// Update the displayed time every minute
setInterval(() => {
    const now = new Date();
    const formattedTime = formatTime(now);
    currentTimeDisplay.textContent = `${formattedTime}`;
}, 60000); // Update every minute


startDateInput.addEventListener("change", function () {
    const selectedDate = new Date(this.value);
    startDateDisplay.textContent = formatDate(selectedDate);
});

endDateInput.addEventListener("change", function () {
    const selectedDate = new Date(this.value);
    endDateDisplay.textContent = formatDate(selectedDate);
});


timeInput.addEventListener("change", function () {
    currentTimeDisplay.textContent = this.value;
});


// inputs
document.querySelectorAll(".flight-section select").forEach(select => {
    select.addEventListener("change", function () {
        this.blur(); // Remove focus from the select input
        this.closest(".flight-section").blur(); // Remove focus from the container
        
        let selectedText = this.options[this.selectedIndex].text;
        this.closest(".flight-section").querySelector(".selected-airport").textContent = selectedText;

    });
});


document.querySelector(".start-date-selector").addEventListener("click", function () {
    startDateInput.showPicker(); // Opens date picker
});

document.querySelector(".end-date-selector").addEventListener("click", function () {
    endDateInput.showPicker(); // Opens date picker
});


document.querySelector(".time-selector").addEventListener("click", function () {
    timeInput.showPicker(); // Opens time picker
});



// map
var map = L.map('map').setView([20.5937, 78.9629], 4);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);



function displayPath(data){
    console.log(data);
    let route_data = data.route_data
    let optimal = data.optimal_path
    
    const routePoints = route_data.map(point => {
        const [lat, lon] = point.Position.split(',').map(coord => parseFloat(coord.trim()));
        return [lat, lon];
    });
    const pathLine = L.polyline(routePoints, {color: '#0e52fe'}).addTo(map);
    map.flyToBounds(pathLine.getBounds(), { duration: 0.4 }); 
    // map.fitBounds(pathLine.getBounds());

    // Add markers for each waypoint
    // data.forEach(point => {
    //     L.marker([parseFloat(point.lat), parseFloat(point.lon)])
    //         .bindPopup(`Waypoint ID: ${point.waypoint_ID}<br>Altitude: ${point.altitude}`)
    //         .addTo(map);
    // });

    // Add marker to start and stop
    
    L.marker(routePoints[0])
    .bindPopup("Source")
    .addTo(map);

    L.marker(routePoints[routePoints.length - 1])
    .bindPopup("Destination")
    .addTo(map)

    const routePointsOP = optimal.map(point =>{
        return [parseFloat(point.latitude), parseFloat(point.longitude)];
    })

    const pathLineOP = L.polyline(routePointsOP, {color:'#808080', dashArray: '5, 10'}).addTo(map);
    map.flyToBounds(pathLineOP.getBounds(), { duration: 0.4 }); 

}

async function searchPath(){
    const searchButton = document.getElementById('search-button');
    const loadingState = document.getElementById('loading-state');

    const start_date = document.getElementById('start-date').value;
    const end_date = document.getElementById('end-date').value;

    console.log(start_date, end_date);

    const startDate = "2024-12-01"  
    const endDate = "2024-12-31"

    try{
        searchButton.disabled = true;
        loadingState.style.display = 'block';
        const response = await fetch(`/get-flight-path?start_date=${start_date}&end_date=${end_date}`);
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