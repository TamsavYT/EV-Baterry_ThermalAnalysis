// Configuration
const REFRESH_INTERVAL = 3000; // 3 seconds - adjust based on your needs
const MAX_CHART_DATA_POINTS = 20;

// Global state
let temperatureChart = null;
let chartData = {
    labels: [],
    datasets: [{
        label: 'Average Temperature',
        data: [],
        borderColor: '#00ffff',
        backgroundColor: 'rgba(0, 255, 255, 0.1)',
        tension: 0.4,
        fill: true
    }, {
        label: 'Max Temperature',
        data: [],
        borderColor: '#ff0055',
        backgroundColor: 'rgba(255, 0, 85, 0.1)',
        tension: 0.4,
        fill: true
    }]
};

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    console.log('[Init] DOMContentLoaded - Starting dashboard initialization');
    // Wait for Chart.js to load before initializing chart
    if (typeof Chart !== 'undefined') {
        initializeChart();
    } else {
        console.log('[Init] Chart.js not loaded yet, will initialize when ready');
    }
    initializeBatteryCells();
    startDataUpdates();
});

// Initialize temperature chart
function initializeChart() {
    console.log('[initializeChart] Initializing temperature chart');
    const ctx = document.getElementById('tempChart');
    if (!ctx) {
        console.error('[initializeChart] Chart canvas not found!');
        return;
    }

    // Destroy existing chart if it exists to prevent "Canvas already in use" error
    if (temperatureChart) {
        console.log('[initializeChart] Destroying existing chart');
        temperatureChart.destroy();
    }

    temperatureChart = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#ffffff',
                        font: {
                            family: 'Rajdhani',
                            size: 12
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0aec0',
                        font: {
                            family: 'Rajdhani'
                        }
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#a0aec0',
                        font: {
                            family: 'Rajdhani'
                        },
                        callback: function (value) {
                            return value + '°C';
                        }
                    }
                }
            }
        }
    });
    console.log('[initializeChart] Chart initialized successfully');
}

// Initialize battery cells grid
function initializeBatteryCells() {
    const grid = document.getElementById('batteryCellsGrid');
    if (!grid) return;

    // Create 64 battery cells (8x8 grid)
    for (let i = 1; i <= 64; i++) {
        const cell = document.createElement('div');
        cell.className = 'battery-cell normal';
        cell.id = `cell-${i}`;
        cell.textContent = i;
        cell.title = `Cell ${i}`;
        grid.appendChild(cell);
    }
}

// Start periodic data updates
function startDataUpdates() {
    // Initial update
    updateDashboard();

    // Periodic updates
    setInterval(updateDashboard, REFRESH_INTERVAL);
}

// Main dashboard update function
async function updateDashboard() {
    console.log('[updateDashboard] Starting dashboard update...');
    try {
        // Fetch data from API
        const data = await fetchBatteryData();
        console.log('[updateDashboard] Received data:', data);

        // Update all components
        console.log('[updateDashboard] Updating system status...');
        updateSystemStatus(data.systemStatus);

        console.log('[updateDashboard] Updating risk gauge...');
        updateRiskGauge(data.riskLevel);

        console.log('[updateDashboard] Updating metrics...');
        updateMetrics(data.metrics);

        console.log('[updateDashboard] Updating battery cells...');
        updateBatteryCells(data.cells);

        console.log('[updateDashboard] Updating chart...');
        updateChart(data.temperature);

        console.log('[updateDashboard] Updating predictions...');
        updatePredictions(data.predictions);

        console.log('[updateDashboard] Updating alerts...');
        updateAlerts(data.alerts);

        console.log('[updateDashboard] Dashboard update complete!');
    } catch (error) {
        console.error('[updateDashboard] Error updating dashboard:', error);
        console.log('[updateDashboard] Falling back to mock data...');
        // Use mock data for demonstration
        updateWithMockData();
    }
}

// Fetch battery data from API
async function fetchBatteryData() {
    console.log('========================================');
    console.log('[fetchBatteryData] Starting API call to /api/battery-data');
    try {
        const response = await fetch('/api/battery-data');
        console.log('[fetchBatteryData] Response status:', response.status);
        console.log('[fetchBatteryData] Response OK:', response.ok);

        if (!response.ok) {
            throw new Error('API request failed with status: ' + response.status);
        }

        const data = await response.json();
        console.log('[fetchBatteryData] Raw data from backend:', data);
        console.log('[fetchBatteryData] Data keys:', Object.keys(data));

        // Transform backend data to dashboard format
        const transformed = transformBatteryData(data);
        console.log('[fetchBatteryData] Transformed data:', transformed);
        console.log('========================================');
        return transformed;
    } catch (error) {
        console.error('[fetchBatteryData] ERROR:', error);
        console.warn('[fetchBatteryData] Using mock data - API not available');
        console.log('========================================');
        throw error;
    }
}

// Transform backend data format to dashboard format
function transformBatteryData(backendData) {
    console.log('[transformBatteryData] Transforming data...');
    console.log('[transformBatteryData] Backend risk level:', backendData.riskLevel);
    console.log('[transformBatteryData] Backend avgTemp:', backendData.averageTemperature);
    console.log('[transformBatteryData] Backend cells count:', backendData.cells ? backendData.cells.length : 0);
    console.log('[transformBatteryData] Backend systemStatus:', backendData.systemStatus);
    const transformed = {
        systemStatus: backendData.systemStatus || 'Active',
        riskLevel: backendData.riskLevel || 0,
        metrics: {
            avgTemp: backendData.averageTemperature || 0,
            maxTemp: backendData.maxTemperature || 0,
            voltage: backendData.voltage || 0,
            current: backendData.current || 0,
            soc: backendData.soc || 0,
            soh: backendData.soh || 0
        },
        cells: (backendData.cells || []).map(cell => ({
            id: cell.cellId,
            temp: cell.temperature ? cell.temperature.toFixed(1) : '0',
            status: cell.status || 'normal'
        })),
        temperature: {
            avg: backendData.averageTemperature || 0,
            max: backendData.maxTemperature || 0
        },
        predictions: {
            forecast1h: Math.round((backendData.riskLevel || 0) + Math.random() * 5),
            forecast6h: Math.round((backendData.riskLevel || 0) + Math.random() * 15),
            forecast24h: Math.round((backendData.riskLevel || 0) + Math.random() * 25),
            recommendation: getRiskRecommendation(backendData.riskLevel || 0)
        },
        alerts: []
    };

    console.log('[transformBatteryData] Transform complete. Risk level:', transformed.riskLevel);
    return transformed;
}

// Update system status
function updateSystemStatus(status) {
    const statusEl = document.getElementById('systemStatus');
    const pulseDot = document.querySelector('.pulse-dot');
    const indicatorEl = document.querySelector('.status-indicator');

    if (!statusEl) return;

    // Normalize status string
    const statusText = status ? status.charAt(0).toUpperCase() + status.slice(1) : 'Online';
    statusEl.textContent = `System ${statusText}`;

    let color = '#00ff87'; // Default green
    let shadowColor = 'rgba(0, 255, 135, 0.5)';

    if (status && status.toLowerCase() === 'inactive') {
        color = '#ff0055'; // Red
        shadowColor = 'rgba(255, 0, 85, 0.5)';
    }

    // Apply colors
    statusEl.style.color = color;
    statusEl.style.textShadow = `0 0 10px ${shadowColor}`;

    if (pulseDot) {
        pulseDot.style.background = color;
        pulseDot.style.boxShadow = `0 0 10px ${color}`;
    }

    if (indicatorEl) {
        indicatorEl.style.borderColor = color;
        indicatorEl.style.background = status && status.toLowerCase() === 'inactive'
            ? 'rgba(255, 0, 85, 0.1)'
            : 'rgba(0, 255, 135, 0.1)';
    }
}

// Update risk gauge
function updateRiskGauge(riskLevel) {
    const riskValueEl = document.getElementById('riskLevel');
    const riskStatusEl = document.getElementById('riskStatus');
    const gaugeFill = document.getElementById('gaugeFill');

    if (!riskValueEl || !riskStatusEl || !gaugeFill) return;

    // Animate value
    animateValue(riskValueEl, parseInt(riskValueEl.textContent) || 0, riskLevel, 1000);

    // Update gauge fill (arc from 0% to 100% = 251.2 total length)
    const offset = 251.2 - (riskLevel / 100 * 251.2);
    gaugeFill.style.strokeDashoffset = offset;

    // Update status text and color
    let status, color;
    if (riskLevel < 30) {
        status = 'Low Risk - Optimal';
        color = '#00ff87';
    } else if (riskLevel < 60) {
        status = 'Moderate Risk - Monitor';
        color = '#ffd700';
    } else if (riskLevel < 80) {
        status = 'High Risk - Warning';
        color = '#ff8c00';
    } else {
        status = 'Critical Risk - Alert';
        color = '#ff0055';
    }

    riskStatusEl.textContent = status;
    riskStatusEl.style.color = color;
}

// Update live metrics
function updateMetrics(metrics) {
    const updates = {
        'avgTemp': { value: metrics.avgTemp, unit: '°C' },
        'maxTemp': { value: metrics.maxTemp, unit: '°C' },
        'voltage': { value: metrics.voltage, unit: 'V' },
        'current': { value: metrics.current, unit: 'A' },
        'soc': { value: metrics.soc, unit: '%' },
        'soh': { value: metrics.soh, unit: '%' }
    };

    Object.keys(updates).forEach(id => {
        const element = document.getElementById(id);
        if (element) {
            const valueEl = element.querySelector('.value');
            if (valueEl) {
                valueEl.textContent = updates[id].value.toFixed(1);
            }
        }
    });
}

// Update battery cells
function updateBatteryCells(cells) {
    cells.forEach(cell => {
        const cellEl = document.getElementById(`cell-${cell.id}`);
        if (cellEl) {
            cellEl.className = 'battery-cell ' + cell.status;
            cellEl.title = `Cell ${cell.id}: ${cell.temp}°C`;
        }
    });
}

// Update temperature chart
function updateChart(temperature) {
    if (!temperatureChart) return;

    const now = new Date().toLocaleTimeString();

    chartData.labels.push(now);
    chartData.datasets[0].data.push(temperature.avg);
    chartData.datasets[1].data.push(temperature.max);

    // Keep only last N data points
    if (chartData.labels.length > MAX_CHART_DATA_POINTS) {
        chartData.labels.shift();
        chartData.datasets[0].data.shift();
        chartData.datasets[1].data.shift();
    }

    temperatureChart.update('none');
}

// Update predictions
function updatePredictions(predictions) {
    updatePredictionBar('forecast1h', 'forecast1hValue', predictions.forecast1h);
    updatePredictionBar('forecast6h', 'forecast6hValue', predictions.forecast6h);
    updatePredictionBar('forecast24h', 'forecast24hValue', predictions.forecast24h);

    const recommendationEl = document.getElementById('aiRecommendation');
    if (recommendationEl) {
        recommendationEl.innerHTML = `<strong>AI Recommendation:</strong> ${predictions.recommendation}`;
    }
}

// Update prediction bar
function updatePredictionBar(barId, valueId, value) {
    const bar = document.getElementById(barId);
    const valueEl = document.getElementById(valueId);

    if (bar) {
        bar.style.width = value + '%';
    }

    if (valueEl) {
        valueEl.textContent = value + '% Risk';
    }
}

// Update alerts
function updateAlerts(alerts) {
    const container = document.getElementById('alertsContainer');
    if (!container || !alerts || alerts.length === 0) return;

    // Clear existing alerts except the first one
    while (container.children.length > 1) {
        container.removeChild(container.lastChild);
    }

    // Add new alerts
    alerts.forEach(alert => {
        const alertEl = createAlertElement(alert);
        container.appendChild(alertEl);
    });
}

// Create alert element
function createAlertElement(alert) {
    const div = document.createElement('div');
    div.className = `alert-item ${alert.type}`;

    const icons = {
        success: '✓',
        warning: '⚠',
        danger: '✕'
    };

    div.innerHTML = `
        <div class="alert-icon">${icons[alert.type] || 'ℹ'}</div>
        <div class="alert-content">
            <div class="alert-title">${alert.title}</div>
            <div class="alert-time">${alert.time}</div>
        </div>
    `;

    return div;
}

// Utility: Animate number value
function animateValue(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;

    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

// Mock data generator for demonstration
function updateWithMockData() {
    const mockData = generateMockData();

    updateRiskGauge(mockData.riskLevel);
    updateMetrics(mockData.metrics);
    updateBatteryCells(mockData.cells);
    updateChart(mockData.temperature);
    updatePredictions(mockData.predictions);

    // Add occasional mock alerts
    if (Math.random() > 0.95) {
        const mockAlerts = [
            { type: 'warning', title: 'Temperature spike in Cell 23', time: 'Just now' },
            { type: 'success', title: 'Cooling system activated', time: '1 min ago' }
        ];
        updateAlerts(mockAlerts);
    }
}

// Generate mock data
function generateMockData() {
    const baseTemp = 25;
    const variation = Math.random() * 15;
    const avgTemp = baseTemp + variation;
    const maxTemp = avgTemp + Math.random() * 10;

    const riskLevel = Math.min(100, Math.max(0, (maxTemp - 25) * 2 + Math.random() * 10));

    return {
        riskLevel: Math.round(riskLevel),
        metrics: {
            avgTemp: avgTemp,
            maxTemp: maxTemp,
            voltage: 380 + Math.random() * 20,
            current: 50 + Math.random() * 30,
            soc: 70 + Math.random() * 20,
            soh: 95 + Math.random() * 4
        },
        cells: Array.from({ length: 64 }, (_, i) => {
            const temp = avgTemp + (Math.random() - 0.5) * 15;
            let status = 'normal';
            if (temp > 40) status = 'critical';
            else if (temp > 35) status = 'warning';

            return {
                id: i + 1,
                temp: temp.toFixed(1),
                status: status
            };
        }),
        temperature: {
            avg: avgTemp,
            max: maxTemp
        },
        predictions: {
            forecast1h: Math.round(riskLevel + Math.random() * 5),
            forecast6h: Math.round(riskLevel + Math.random() * 15),
            forecast24h: Math.round(riskLevel + Math.random() * 25),
            recommendation: getRiskRecommendation(riskLevel)
        },
        alerts: []
    };
}

// Get risk recommendation based on level
function getRiskRecommendation(riskLevel) {
    if (riskLevel < 30) {
        return 'All systems operating within normal parameters. Continue monitoring.';
    } else if (riskLevel < 60) {
        return 'Elevated temperature detected in some cells. Recommend reducing charge rate.';
    } else if (riskLevel < 80) {
        return 'High thermal risk detected. Activate enhanced cooling and reduce power output.';
    } else {
        return 'CRITICAL: Immediate action required. Reduce all battery operations and activate emergency cooling.';
    }
}

// Load Chart.js library
(function loadChartJS() {
    if (typeof Chart === 'undefined') {
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js';
        script.onload = () => {
            console.log('[loadChartJS] Chart.js loaded successfully');
            // Only initialize if DOM is already loaded
            if (document.readyState === 'complete' || document.readyState === 'interactive') {
                console.log('[loadChartJS] DOM ready, initializing chart now');
                initializeChart();
            } else {
                console.log('[loadChartJS] Waiting for DOMContentLoaded to initialize chart');
            }
        };
        document.head.appendChild(script);
    }
})();
