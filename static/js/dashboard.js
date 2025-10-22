// static/js/dashboard.js
document.addEventListener('DOMContentLoaded', function() {
    // --- Element Cache ---
    const masterStatusDisplay = document.getElementById('master-status-display');
    const startCycleBtn = document.getElementById('start-cycle-btn');
    const stopCycleBtn = document.getElementById('stop-cycle-btn');
    // Removed references to genesisConsole, singularityLog, and Champion elements
    const dashboard = document.getElementById('bots-dashboard');
    const modelsTableBody = document.getElementById('models-table-body');

    // --- State Management ---
    // Polling intervals remain crucial for updates.
    let masterPollInterval;
    let modelsPollInterval;
    let pricesPollInterval;

    // CRITICAL FIX: Price cache to prevent blinking
    const priceCache = {};

    // --- UI Update Functions ---

    // Removed logToConsole and updateConsole functions.

    function createBotCard(symbol, botData = null) {
        const symbolId = symbol.replace('/', '').replace(':', '');
        const isActive = botData !== null;

        let statusBadge = `<span class="badge bg-secondary">Inactive</span>`;
        if (isActive) {
            // Use bg-success for active, bg-danger for terminated/inactive bots
            statusBadge = `<span class="badge ${botData.is_active ? 'bg-success' : 'bg-danger'}">${botData.is_active ? 'Active' : 'Inactive'}</span>`;
        }

        let positionInfo = 'Flat';
        let capitalInfo = 'N/A';
        // Detailed status message (What the bot is doing)
        let statusDetail = 'Idle';

        if (isActive && botData.is_active) {
            // Prioritize the backend's status message
            statusDetail = botData.status_message || 'Monitoring Market Conditions';

            if (botData.in_position && botData.current_trade) {
                const trade = botData.current_trade;
                const entryPrice = trade.entry_price || 0;
                // Assuming LONG based on original code context, adjust if backend provides 'side'
                const side = 'LONG';
                const sideClass = 'text-success';

                positionInfo = `<span class="fw-bold ${sideClass}">${side}</span> @ $${entryPrice.toFixed(4)}`;

                // Update status detail when in position
                statusDetail = `In Trade: Monitoring PnL, tracking stops, and evaluating exit conditions.`;

                if (trade.capital_deployed) {
                    capitalInfo = `$${trade.capital_deployed.toFixed(2)}`;
                }
            } else {
                 // Enhance the 'searching' message
                 if (statusDetail.toLowerCase().includes('idle') || statusDetail.toLowerCase().includes('monitoring') || statusDetail.toLowerCase().includes('scanning')) {
                     statusDetail = 'Searching: Analyzing microstructure and MTF alignment for high-probability entry signals.';
                 }
            }
        } else if (isActive && !botData.is_active) {
            statusDetail = 'Agent terminated or paused.';
        }

        const modelId = botData?.model_id ? botData.model_id.substring(0, 18) + '...' : 'N/A';
        const balance = botData?.balance ? `$${botData.balance.toFixed(2)}` : 'N/A';

        // CRITICAL FIX: Use cached price if available
        let displayPrice = '$---.--';
        if (priceCache[symbol]) {
            const price = parseFloat(priceCache[symbol]);
            const precision = price > 1000 ? 2 : (price > 1 ? 4 : 6);
            displayPrice = `$${price.toFixed(precision)}`;
        }

        // Redesigned Card Structure
        return `
            <div class="col">
                <div class="card h-100 asset-block" id="bot-card-${symbolId}">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">${symbol}</h5>
                        ${statusBadge}
                    </div>
                    <div class="card-body">
                        <div class="text-center mb-4">
                            <div class="asset-price" id="price-${symbolId}">${displayPrice}</div>
                            <small class="text-muted">Current Price</small>
                        </div>
                        
                        <div class="status-detail mb-3" id="status-detail-${symbolId}">
                             ${statusDetail}
                        </div>

                        <ul class="list-group list-group-flush">
                            <li class="list-group-item px-0">Position: <span class="float-end fw-bold" id="pos-${symbolId}">${positionInfo}</span></li>
                            <li class="list-group-item px-0">Deployed Capital: <span class="float-end" id="capital-${symbolId}">${capitalInfo}</span></li>
                            <li class="list-group-item px-0">Account Balance: <span class="float-end" id="balance-${symbolId}">${balance}</span></li>
                        </ul>
                    </div>
                    <div class="card-footer small text-muted">
                        Model: <code id="model-${symbolId}">${modelId}</code>
                    </div>
                </div>
            </div>
        `;
    }

    // Main UI update function driven by /api/master_status
    function updateUI(data) {
        const isRunning = data.trader.is_running || data.singularity.is_running;
        masterStatusDisplay.innerHTML = isRunning ? '<span class="badge bg-success">SYSTEM ONLINE</span>' : '<span class="badge bg-secondary">ALL SYSTEMS IDLE</span>';
        startCycleBtn.disabled = isRunning;
        stopCycleBtn.disabled = !isRunning;

        // Update Dashboard Grid (data.dashboard structure assumed from original JS)
        if (data.dashboard) {
            dashboard.innerHTML = ''; // Clear existing cards
            const assets = Object.keys(data.dashboard).sort();

            if (assets.length === 0) {
                dashboard.innerHTML = '<div class="col-12"><p class="text-muted">Awaiting connection and asset data...</p></div>';
                return;
            }

            assets.forEach(symbol => {
                const botData = data.dashboard[symbol];
                dashboard.innerHTML += createBotCard(symbol, botData);
            });

            // CRITICAL FIX: Immediately fetch prices after recreating cards
            // This prevents the "$---.--" placeholder from being visible
            fetchPrices();
        }
    }

    // Update Prices
    function updatePrices(prices) {
        Object.keys(prices).forEach(symbol => {
            const symbolId = symbol.replace('/', '').replace(':', '');
            const priceEl = document.getElementById(`price-${symbolId}`);

            const newPrice = parseFloat(prices[symbol]);

            // CRITICAL FIX: Validate price before updating
            // If price is invalid (NaN, null, undefined, 0, negative), keep the old price
            if (isNaN(newPrice) || newPrice === null || newPrice <= 0) {
                console.warn(`Invalid price for ${symbol}: ${prices[symbol]}`);
                return; // Skip this update, keep old price visible
            }

            // CRITICAL FIX: Save valid price to cache immediately
            priceCache[symbol] = newPrice;

            // If element doesn't exist yet (cards not created), cache is enough
            if (!priceEl) {
                return;
            }

            const oldPriceText = priceEl.textContent.replace(/[^0-9.-]+/g,"");
            const oldPrice = parseFloat(oldPriceText);

            // Dynamic precision formatting
            const precision = newPrice > 1000 ? 2 : (newPrice > 1 ? 4 : 6);

            // Only update if price actually changed
            if (!isNaN(oldPrice) && Math.abs(newPrice - oldPrice) < 0.0001) {
                return; // Price hasn't changed meaningfully, skip animation
            }

            priceEl.textContent = `$${newPrice.toFixed(precision)}`;

            // Add smooth animation class with color indication
            if (!isNaN(oldPrice) && newPrice !== oldPrice) {
                // Remove any existing animation classes first
                priceEl.classList.remove('price-up', 'price-down');

                // Force a reflow to restart the animation
                void priceEl.offsetWidth;

                const priceClass = newPrice > oldPrice ? 'price-up' : 'price-down';
                priceEl.classList.add(priceClass);

                // Remove the class after the animation completes
                setTimeout(() => {
                    priceEl.classList.remove('price-up', 'price-down');
                }, 500); // Fast 500ms animation
            }
        });
    }

    // Update Model Archive Table
    function updateModels(models) {
        modelsTableBody.innerHTML = ''; // Clear the table
        // Sort models by timestamp descending (newest first)
        models.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

        models.forEach(model => {
            const row = document.createElement('tr');
            const fitness = model.fitness_score ? model.fitness_score.toFixed(4) : 'N/A';

            // Removed status-based row coloring for cleaner UI consistency

            row.innerHTML = `
                <td>${model.timestamp}</td>
                <td><code class="small">${model.model_id.substring(0, 25)}...</code></td>
                <td>${model.asset_symbol}</td>
                <td>${model.architecture || 'N/A'}</td>
                <td>${fitness}</td>
                <td>${model.status}</td>
            `;
            modelsTableBody.appendChild(row);
        });
    }

    // --- API Fetching Functions (Preserving original endpoints) ---
    async function fetchMasterStatus() {
        try {
            // Using the endpoint from the original dashboard.js provided by the user
            const response = await fetch('/api/master_status');
            if (!response.ok) throw new Error('Network response was not ok.');
            const data = await response.json();
            updateUI(data);
        } catch (error) {
            console.error('Error fetching master status:', error);
            masterStatusDisplay.innerHTML = '<span class="badge bg-danger">CONNECTION ERROR</span>';
        }
    }

    async function fetchPrices() {
        try {
            const response = await fetch('/api/prices');
            if (!response.ok) return;
            const prices = await response.json();
            updatePrices(prices);
        } catch (error) {
            console.error('Error fetching prices:', error);
        }
    }

    async function fetchModels() {
        try {
            // Change this line:
            // const response = await fetch('/api/models');
            // To this:
            const response = await fetch('/api/models/list'); // <-- FIX

            if (!response.ok) return;
            // The /api/models/list endpoint returns {'models': [...]}, adjust if needed
            const data = await response.json();
            updateModels(data.models || []); // <-- Adjust based on API response structure
        } catch (error) {
            console.error('Error fetching models:', error);
        }
    }

    // --- Event Handlers (Preserving original endpoints) ---
    startCycleBtn.addEventListener('click', async () => {
        startCycleBtn.disabled = true;
        try {
            await fetch('/api/start_loop', { method: 'POST' });
            fetchMasterStatus(); // Immediate update
        } catch (error) {
            console.error('Error starting loop:', error);
            startCycleBtn.disabled = false;
        }
    });

    stopCycleBtn.addEventListener('click', async () => {
        stopCycleBtn.disabled = true;
        try {
            await fetch('/api/stop_loop', { method: 'POST' });
            fetchMasterStatus(); // Immediate update
        } catch (error) {
            console.error('Error stopping loop:', error);
            stopCycleBtn.disabled = false;
        }
    });

    // --- Initialization ---
    fetchPrices(); // CRITICAL: Fetch prices FIRST to populate cache
    fetchMasterStatus();
    fetchModels();
    // Set intervals for polling
    masterPollInterval = setInterval(fetchMasterStatus, 5000); // Poll status/grid every 5s
    pricesPollInterval = setInterval(fetchPrices, 2000); // Poll prices every 2s
    modelsPollInterval = setInterval(fetchModels, 15000); // Poll archive every 15s
});