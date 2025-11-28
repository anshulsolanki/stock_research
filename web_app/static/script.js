// Tab switching function
function showTab(tabName) {
    // Hide all tabs
    const tabs = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => tab.classList.remove('active'));

    // Remove active from all buttons
    const buttons = document.querySelectorAll('.tab-button');
    buttons.forEach(btn => btn.classList.remove('active'));

    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');

    // Activate selected button
    event.target.classList.add('active');
}

function toggleConfig(panelId) {
    const panel = document.getElementById(panelId);
    if (panel.style.display === 'none') {
        panel.style.display = 'flex';
    } else {
        panel.style.display = 'none';
    }
}

async function analyzeStock(analysisType = 'all') {
    const ticker = document.getElementById('tickerInput').value.trim();
    const resultsDiv = document.getElementById('results');
    const errorDiv = document.getElementById('error');

    // Determine which button triggered the analysis
    let activeBtn;
    let originalBtnText;

    if (analysisType === 'all') {
        activeBtn = document.getElementById('analyzeBtn');
        originalBtnText = activeBtn.querySelector('.btn-text').textContent;
    } else if (analysisType === 'macd') {
        activeBtn = document.querySelector('#macd-tab .run-analysis-btn');
        originalBtnText = activeBtn.textContent;
    } else if (analysisType === 'supertrend') {
        activeBtn = document.querySelector('#supertrend-tab .run-analysis-btn');
        originalBtnText = activeBtn.textContent;
    } else if (analysisType === 'bollinger') {
        activeBtn = document.querySelector('#bollinger-tab .run-analysis-btn');
        originalBtnText = activeBtn.textContent;
    } else if (analysisType === 'crossover') {
        activeBtn = document.querySelector('#crossover-tab .run-analysis-btn');
        originalBtnText = activeBtn.textContent;
    } else if (analysisType === 'donchian') {
        activeBtn = document.querySelector('#donchian-tab .run-analysis-btn');
        originalBtnText = activeBtn.textContent;
    }

    // Validation
    if (!ticker) {
        showError('Please enter a stock ticker symbol');
        return;
    }

    // Reset Error
    errorDiv.style.display = 'none';

    // Show loading state
    if (activeBtn) {
        activeBtn.disabled = true;
        if (analysisType === 'all') {
            activeBtn.querySelector('.btn-text').textContent = 'Analyzing...';
            activeBtn.querySelector('.loader').style.display = 'inline-block';
        } else {
            activeBtn.textContent = 'Running...';
        }
    }

    // Prepare Request Body
    const requestBody = {
        ticker: ticker,
        analysis_type: analysisType
    };

    if (analysisType === 'all' || analysisType === 'macd') {
        requestBody.macd_config = {
            FAST: parseInt(document.getElementById('macdFast').value) || 12,
            SLOW: parseInt(document.getElementById('macdSlow').value) || 26,
            SIGNAL: parseInt(document.getElementById('macdSignal').value) || 9,
            INTERVAL: document.getElementById('macdInterval').value || '1d',
            LOOKBACK_PERIODS: parseInt(document.getElementById('macdLookback').value) || 730
        };
    }

    if (analysisType === 'all' || analysisType === 'supertrend') {
        requestBody.supertrend_config = {
            PERIOD: parseInt(document.getElementById('stPeriod').value) || 14,
            MULTIPLIER: parseFloat(document.getElementById('stMultiplier').value) || 3.0,
            INTERVAL: document.getElementById('stInterval').value || '1d',
            LOOKBACK_PERIODS: parseInt(document.getElementById('stLookback').value) || 730
        };
    }

    if (analysisType === 'all' || analysisType === 'bollinger') {
        requestBody.bollinger_config = {
            WINDOW: parseInt(document.getElementById('bbWindow').value) || 20,
            NUM_STD: parseFloat(document.getElementById('bbNumStd').value) || 2,
            INTERVAL: document.getElementById('bbInterval').value || '1d',
            LOOKBACK_PERIODS: parseInt(document.getElementById('bbLookback').value) || 730
        };
    }

    if (analysisType === 'all' || analysisType === 'crossover') {
        requestBody.crossover_config = {
            WINDOWS: [
                parseInt(document.getElementById('crossShort').value) || 20,
                parseInt(document.getElementById('crossMedium').value) || 50,
                parseInt(document.getElementById('crossLong').value) || 200
            ],
            INTERVAL: document.getElementById('crossInterval').value || '1d',
            LOOKBACK_PERIODS: parseInt(document.getElementById('crossLookback').value) || 730
        };
    }

    if (analysisType === 'all' || analysisType === 'donchian') {
        requestBody.donchian_config = {
            WINDOW: parseInt(document.getElementById('donchianWindow').value) || 20,
            INTERVAL: document.getElementById('donchianInterval').value || '1d',
            LOOKBACK_PERIODS: parseInt(document.getElementById('donchianLookback').value) || 730
        };
    }

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data, analysisType);
        } else {
            showError(data.error || 'Analysis failed');
        }

    } catch (error) {
        showError('Network error. Please check your connection and try again.');
        console.error('Error:', error);
    } finally {
        // Reset button state
        if (activeBtn) {
            activeBtn.disabled = false;
            if (analysisType === 'all') {
                activeBtn.querySelector('.btn-text').textContent = originalBtnText;
                activeBtn.querySelector('.loader').style.display = 'none';
            } else {
                activeBtn.textContent = originalBtnText;
            }
        }
    }
}

function displayResults(data, analysisType) {
    const resultsDiv = document.getElementById('results');

    // Update header
    document.getElementById('tickerName').textContent = data.ticker;
    document.getElementById('timestamp').textContent = new Date().toLocaleString();

    // ========== MACD Data ==========
    if (data.macd && (analysisType === 'all' || analysisType === 'macd')) {
        const macd = data.macd;

        document.getElementById('macdLine').textContent = macd.macd_line.toFixed(2);
        document.getElementById('signalLine').textContent = macd.signal_line.toFixed(2);
        document.getElementById('histogram').textContent = macd.histogram.toFixed(2);
        document.getElementById('trend').textContent = macd.trend;
        document.getElementById('momentum').textContent = macd.momentum;

        // Color-code trend card
        const trendCard = document.getElementById('trendCard');
        trendCard.classList.remove('bullish', 'bearish');
        trendCard.classList.add(macd.trend.toLowerCase());

        // Color-code momentum card
        const momentumCard = document.getElementById('momentumCard');
        momentumCard.classList.remove('strengthening', 'weakening');
        momentumCard.classList.add(macd.momentum.toLowerCase());

        // Update crossover signal
        const crossoverEl = document.getElementById('crossoverSignal');
        if (macd.crossover_signal) {
            crossoverEl.textContent = macd.crossover_signal;
        } else {
            crossoverEl.textContent = 'No Recent Signal';
        }

        // Handle divergences
        const divergenceSection = document.getElementById('divergenceSection');
        const divergenceList = document.getElementById('divergenceList');

        if (macd.divergences && macd.divergences.length > 0) {
            divergenceSection.style.display = 'block';
            divergenceList.innerHTML = '';

            macd.divergences.forEach(div => {
                const divItem = document.createElement('div');
                divItem.className = 'divergence-item';
                divItem.innerHTML = `
                    <div class="divergence-type">${div.type}</div>
                    <div class="divergence-details">
                        Date: ${div.date} | Price: ${div.price.toFixed(2)}
                        <br>${div.details}
                    </div>
                `;
                divergenceList.appendChild(divItem);
            });
        } else {
            divergenceSection.style.display = 'none';
        }

        // Update MACD chart
        document.getElementById('macdChartImage').src = 'data:image/png;base64,' + macd.chart_image;
    }

    // ========== Supertrend Data ==========
    if (data.supertrend && (analysisType === 'all' || analysisType === 'supertrend')) {
        const supertrend = data.supertrend;

        document.getElementById('supertrendStatus').textContent = supertrend.status;
        document.getElementById('supertrendPrice').textContent = supertrend.last_price.toFixed(2);
        document.getElementById('supertrendValue').textContent = supertrend.supertrend_value.toFixed(2);
        document.getElementById('supertrendDate').textContent = supertrend.last_date;

        // Color-code Supertrend status card
        const statusCard = document.getElementById('supertrendStatusCard');
        statusCard.classList.remove('uptrend', 'downtrend');
        statusCard.classList.add(supertrend.last_trend === 1 ? 'uptrend' : 'downtrend');

        // Update Supertrend chart
        document.getElementById('supertrendChartImage').src = 'data:image/png;base64,' + supertrend.chart_image;
    }

    // ========== Bollinger Band Data ==========
    if (data.bollinger && (analysisType === 'all' || analysisType === 'bollinger')) {
        const bollinger = data.bollinger;

        document.getElementById('bbUpper').textContent = bollinger.bb_upper.toFixed(2);
        document.getElementById('bbLower').textContent = bollinger.bb_lower.toFixed(2);
        document.getElementById('bbMiddle').textContent = bollinger.sma_20.toFixed(2);
        document.getElementById('bbPctB').textContent = bollinger.pct_b.toFixed(2);
        document.getElementById('bbBandWidth').textContent = bollinger.bandwidth.toFixed(4);
        document.getElementById('bbStatus').textContent = bollinger.status;

        // Handle Signals List
        const bbSignalSection = document.getElementById('bbSignalSection');
        const bbSignalList = document.getElementById('bbSignalList');

        if (bollinger.signals && bollinger.signals.length > 0) {
            bbSignalSection.style.display = 'block';
            bbSignalList.innerHTML = '';

            // Show last 5 signals in reverse order
            const recentSignals = bollinger.signals.slice(-5).reverse();

            recentSignals.forEach(sig => {
                const sigItem = document.createElement('div');
                sigItem.className = 'divergence-item'; // Reuse divergence styling

                // Color code signal type
                let typeClass = '';
                if (sig.type.includes('Buy')) typeClass = 'bullish';
                if (sig.type.includes('Sell')) typeClass = 'bearish';

                sigItem.innerHTML = `
                    <div class="divergence-type ${typeClass}">${sig.type}</div>
                    <div class="divergence-details">
                        Date: ${sig.date} | Price: ${sig.price.toFixed(2)}
                        <br>Reason: ${sig.reason}
                    </div>
                `;
                bbSignalList.appendChild(sigItem);
            });
        } else {
            bbSignalSection.style.display = 'none';
        }

        // Update Bollinger chart
        document.getElementById('bollingerChartImage').src = 'data:image/png;base64,' + bollinger.chart_image;
    }

    // ========== Crossover Data ==========
    if (data.crossover && (analysisType === 'all' || analysisType === 'crossover')) {
        const crossover = data.crossover;

        document.getElementById('ema20').textContent = crossover.ema_20 ? crossover.ema_20.toFixed(2) : '--';
        document.getElementById('ema50').textContent = crossover.ema_50 ? crossover.ema_50.toFixed(2) : '--';
        document.getElementById('ema200').textContent = crossover.ema_200 ? crossover.ema_200.toFixed(2) : '--';
        document.getElementById('crossoverTrend').textContent = crossover.trend_status;

        // Color-code Trend Status
        const trendCard = document.getElementById('crossoverTrendCard');
        trendCard.classList.remove('bullish', 'bearish');
        if (crossover.trend_status.includes('Uptrend')) {
            trendCard.classList.add('bullish');
        } else if (crossover.trend_status.includes('Downtrend')) {
            trendCard.classList.add('bearish');
        }

        // Handle Golden Cross Info
        const gcSection = document.getElementById('goldenCrossSection');
        const gcInfo = document.getElementById('goldenCrossInfo');

        if (crossover.gc_date) {
            gcSection.style.display = 'block';
            gcInfo.innerHTML = `
                <div class="divergence-item">
                    <div class="divergence-type bullish">Golden Cross</div>
                    <div class="divergence-details">
                        Date: ${crossover.gc_date} | Price: ${crossover.gc_price.toFixed(2)}
                        <br>50 EMA crossed above 200 EMA
                    </div>
                </div>
            `;
        } else {
            gcSection.style.display = 'none';
        }

        // Update Crossover chart
        document.getElementById('crossoverChartImage').src = 'data:image/png;base64,' + crossover.chart_image;
    }

    // ========== Donchian Channel Data ==========
    if (data.donchian && (analysisType === 'all' || analysisType === 'donchian')) {
        const donchian = data.donchian;

        document.getElementById('donchianUpper').textContent = donchian.dc_upper.toFixed(2);
        document.getElementById('donchianLower').textContent = donchian.dc_lower.toFixed(2);
        document.getElementById('donchianMiddle').textContent = donchian.dc_middle.toFixed(2);
        document.getElementById('donchianPrice').textContent = donchian.last_price.toFixed(2);
        document.getElementById('donchianStatus').textContent = donchian.status;

        // Color-code Status Card
        const statusCard = document.getElementById('donchianStatusCard');
        statusCard.classList.remove('bullish', 'bearish', 'neutral');
        if (donchian.status.includes('Bullish')) {
            statusCard.classList.add('bullish');
        } else if (donchian.status.includes('Bearish')) {
            statusCard.classList.add('bearish');
        } else {
            statusCard.classList.add('neutral');
        }

        // Update signal
        const signalEl = document.getElementById('donchianSignal');
        if (donchian.breakout_signal) {
            signalEl.textContent = donchian.breakout_signal;
        } else {
            signalEl.textContent = 'No Active Signal';
        }

        // Handle Signals List
        const donchianSignalSection = document.getElementById('donchianSignalSection');
        const donchianSignalList = document.getElementById('donchianSignalList');

        if (donchian.signals && donchian.signals.length > 0) {
            donchianSignalSection.style.display = 'block';
            donchianSignalList.innerHTML = '';

            // Show last 5 signals in reverse order
            const recentSignals = donchian.signals.slice(-5).reverse();

            recentSignals.forEach(sig => {
                const sigItem = document.createElement('div');
                sigItem.className = 'divergence-item';

                // Color code signal type
                let typeClass = '';
                if (sig.type.includes('Bullish')) typeClass = 'bullish';
                if (sig.type.includes('Bearish')) typeClass = 'bearish';

                sigItem.innerHTML = `
                    <div class="divergence-type ${typeClass}">${sig.type}</div>
                    <div class="divergence-details">
                        Date: ${sig.date} | Price: ${sig.price.toFixed(2)}
                        <br>Channel Range: ${sig.lower.toFixed(2)} - ${sig.upper.toFixed(2)}
                    </div>
                `;
                donchianSignalList.appendChild(sigItem);
            });
        } else {
            donchianSignalSection.style.display = 'none';
        }

        // Update Donchian chart
        document.getElementById('donchianChartImage').src = 'data:image/png;base64,' + donchian.chart_image;
    }

    // Show results
    resultsDiv.style.display = 'block';
    if (analysisType === 'all') {
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = '❌ ' + message;
    errorDiv.style.display = 'block';
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Allow Enter key to trigger analysis
document.getElementById('tickerInput').addEventListener('keypress', function (event) {
    if (event.key === 'Enter') {
        analyzeStock('all');
    }
});
