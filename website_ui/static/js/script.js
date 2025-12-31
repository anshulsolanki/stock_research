// Cache variables
let sectorAnalysisCache = null;
let stocksInSectorCache = {};
let lastAnalyzedStock = null;
let batchAnalysisCache = null;

// Initialize caches from localStorage on page load
function initializeCaches() {
    try {
        const savedSectorCache = localStorage.getItem('sectorAnalysisCache');
        if (savedSectorCache) {
            sectorAnalysisCache = JSON.parse(savedSectorCache);
        }

        const savedStocksCache = localStorage.getItem('stocksInSectorCache');
        if (savedStocksCache) {
            stocksInSectorCache = JSON.parse(savedStocksCache);
        }

        const savedLastStock = localStorage.getItem('lastAnalyzedStock');
        if (savedLastStock) {
            lastAnalyzedStock = JSON.parse(savedLastStock);
        }

        const savedBatchCache = localStorage.getItem('batchAnalysisCache');
        if (savedBatchCache) {
            batchAnalysisCache = JSON.parse(savedBatchCache);
        }
    } catch (error) {
        console.error('Error loading caches from localStorage:', error);
    }
}

// Call initialization immediately
initializeCaches();

// Tab Switching Logic
function switchMainTab(tabName) {
    // Update buttons
    document.querySelectorAll('.main-tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('onclick').includes(`'${tabName}'`)) {
            btn.classList.add('active');
        }
    });

    // Update content sections
    document.querySelectorAll('.section-content').forEach(section => {
        section.classList.remove('active');
    });

    const targetSection = document.getElementById(`${tabName}-section`);
    if (targetSection) {
        targetSection.classList.add('active');
    }

    // Auto-load data for specific tabs
    if (tabName === 'sector_analysis') {
        loadSectorAnalysis(false); // false = don't force refresh, use cache if available
    }


}

function switchSubTab(section, tabName) {
    // Update buttons within the specific section
    const sectionEl = document.getElementById(`${section}-section`);
    sectionEl.querySelectorAll('.sub-tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('onclick').includes(`'${tabName}'`)) {
            btn.classList.add('active');
        }
    });

    // Update panels within the specific section
    sectionEl.querySelectorAll('.analysis-panel').forEach(panel => {
        panel.classList.remove('active');
    });

    const targetPanel = document.getElementById(`${tabName}-tab`);
    if (targetPanel) {
        targetPanel.classList.add('active');
    }
}

function toggleConfig(panelId) {
    const panel = document.getElementById(panelId);
    if (panel.style.display === 'none') {
        panel.style.display = 'flex';
    } else {
        panel.style.display = 'none';
    }
}

// Stock Selection from Sidebar
function selectStock(ticker) {
    document.getElementById('tickerInput').value = ticker;

    // Highlight sidebar item
    document.querySelectorAll('.stock-item').forEach(item => {
        item.classList.remove('active');
        if (item.querySelector('.stock-symbol').textContent === ticker) {
            item.classList.add('active');
        }
    });

    analyzeStock('all');
}

// Sidebar Search
document.getElementById('stockSearchInput').addEventListener('input', function (e) {
    const searchTerm = e.target.value.toLowerCase();
    document.querySelectorAll('.stock-item').forEach(item => {
        const symbol = item.querySelector('.stock-symbol').textContent.toLowerCase();
        const name = item.querySelector('.stock-name').textContent.toLowerCase();

        if (symbol.includes(searchTerm) || name.includes(searchTerm)) {
            item.style.display = 'flex';
        } else {
            item.style.display = 'none';
        }
    });
});

async function analyzeStock(analysisType = 'all') {
    const ticker = document.getElementById('tickerInput').value.trim();
    const errorDiv = document.getElementById('error');

    // Validation
    if (!ticker) {
        showError('Please enter a stock ticker symbol');
        return;
    }

    // Reset Error
    errorDiv.style.display = 'none';

    // Show loading state (optional: add a global loader or button loader)
    const analyzeBtn = document.getElementById('analyzeBtn');
    const originalBtnText = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    analyzeBtn.disabled = true;

    // Prepare Request Body
    const requestBody = {
        ticker: ticker,
        analysis_type: analysisType
    };

    // Collect Configs (Same as before, IDs are preserved)
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

    if (analysisType === 'all' || analysisType === 'rsi') {
        requestBody.rsi_config = {
            PERIOD: parseInt(document.getElementById('rsiPeriod').value) || 14,
            ORDER: parseInt(document.getElementById('rsiOrder').value) || 5,
            INTERVAL: document.getElementById('rsiInterval').value || '1wk',
            LOOKBACK_PERIODS: parseInt(document.getElementById('rsiLookback').value) || 730,
            RSI_OVERBOUGHT: parseInt(document.getElementById('rsiOverbought').value) || 70,
            RSI_OVERSOLD: parseInt(document.getElementById('rsiOversold').value) || 30
        };
    }

    if (analysisType === 'all' || analysisType === 'rsi_volume') {
        requestBody.rsi_volume_config = {
            RSI_PERIOD: parseInt(document.getElementById('rsiVolumePeriod').value) || 14,
            ORDER: parseInt(document.getElementById('rsiVolumeOrder').value) || 5,
            VOLUME_MA_SHORT: parseInt(document.getElementById('rsiVolumeMAShort').value) || 20,
            VOLUME_MA_LONG: parseInt(document.getElementById('rsiVolumeMALong').value) || 50,
            INTERVAL: document.getElementById('rsiVolumeInterval').value || '1d',
            LOOKBACK_PERIODS: parseInt(document.getElementById('rsiVolumeLookback').value) || 730,
            RSI_OVERBOUGHT: parseInt(document.getElementById('rsiVolumeOverbought').value) || 70,
            RSI_OVERSOLD: parseInt(document.getElementById('rsiVolumeOversold').value) || 30
        };
    }

    if (analysisType === 'all' || analysisType === 'volatility_squeeze') {
        requestBody.volatility_squeeze_config = {
            BB_PERIOD: parseInt(document.getElementById('vsBBPeriod').value) || 20,
            BB_STD: parseFloat(document.getElementById('vsBBStd').value) || 2,
            ATR_PERIOD: parseInt(document.getElementById('vsATRPeriod').value) || 14,
            VOLUME_MA_PERIOD: parseInt(document.getElementById('vsVolumeMA').value) || 20,
            VOLUME_SURGE_MULTIPLIER: parseFloat(document.getElementById('vsVolumeSurge').value) || 1.5,
            LOOKBACK: parseInt(document.getElementById('vsLookback').value) || 126,
            SCAN_DAYS: parseInt(document.getElementById('vsScanDays').value) || 60,
            PERCENTILE_THRESHOLD: parseInt(document.getElementById('vsPercentile').value) || 20,
            INTERVAL: document.getElementById('vsInterval').value || '1d',
            LOOKBACK_PERIODS: parseInt(document.getElementById('vsLookbackPeriods').value) || 365
        };
    }

    if (analysisType === 'all' || analysisType === 'rs') {
        const selectedBenchmark = document.getElementById('rsBenchmarkSelect').value;
        requestBody.rs_config = {
            INTERVAL: document.getElementById('rsInterval').value || '1d',
            LOOKBACK_PERIODS: parseInt(document.getElementById('rsLookbackPeriods').value) || 504,
            BENCHMARK_TICKER: selectedBenchmark || null
        };
        requestBody.use_sector_index = document.getElementById('rsUseSectorIndex').checked;
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

            // Save last analyzed stock to localStorage
            try {
                const stockData = {
                    ticker: ticker,
                    data: data,
                    analysisType: analysisType,
                    timestamp: new Date().toISOString()
                };
                lastAnalyzedStock = stockData;
                localStorage.setItem('lastAnalyzedStock', JSON.stringify(stockData));
            } catch (error) {
                console.error('Error saving last stock to localStorage:', error);
            }
        } else {
            showError(data.error || 'Analysis failed');
        }

    } catch (error) {
        showError('Network error. Please check your connection and try again.');
        console.error('Error:', error);
    } finally {
        analyzeBtn.innerHTML = originalBtnText;
        analyzeBtn.disabled = false;
    }
}

function displayResults(data, analysisType) {
    // Show dashboard content if hidden
    document.getElementById('dashboardContent').style.display = 'block';

    // Update header
    document.getElementById('selectedTicker').textContent = data.ticker;
    document.getElementById('timestamp').textContent = 'Last Updated: ' + new Date().toLocaleString();



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
        crossoverEl.textContent = macd.crossover_signal || 'No Recent Signal';

        // Handle divergences
        const divergenceSection = document.getElementById('divergenceSection');
        const divergenceList = document.getElementById('divergenceList');

        if (macd.divergences && macd.divergences.length > 0) {
            divergenceSection.style.display = 'block';
            divergenceList.innerHTML = '';
            macd.divergences.forEach(div => {
                const divItem = document.createElement('div');
                divItem.className = `divergence-item ${div.type.includes('Bullish') ? 'bullish' : 'bearish'}`;
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

        document.getElementById('macdChartImage').src = 'data:image/png;base64,' + macd.chart_image;
    }

    // ========== Supertrend Data ==========
    if (data.supertrend && (analysisType === 'all' || analysisType === 'supertrend')) {
        const supertrend = data.supertrend;
        document.getElementById('supertrendStatus').textContent = supertrend.status;
        document.getElementById('supertrendPrice').textContent = supertrend.last_price.toFixed(2);
        document.getElementById('supertrendValue').textContent = supertrend.supertrend_value.toFixed(2);
        document.getElementById('supertrendDate').textContent = supertrend.last_date;
        document.getElementById('supertrendSignalDate').textContent = supertrend.signal_date;

        const statusCard = document.getElementById('supertrendStatusCard');
        statusCard.classList.remove('uptrend', 'downtrend');
        statusCard.classList.add(supertrend.last_trend === 1 ? 'uptrend' : 'downtrend');

        const summarySection = document.getElementById('supertrendSummarySection');
        const summaryDiv = document.getElementById('supertrendSummary');
        summaryDiv.textContent = `Status: ${supertrend.status}\nLast Price: ${supertrend.last_price.toFixed(2)}\nSupertrend: ${supertrend.supertrend_value.toFixed(2)}\nSignal Date: ${supertrend.signal_date}\nDate: ${supertrend.last_date}`;
        summarySection.style.display = 'block';

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

        const bbSignalSection = document.getElementById('bbSignalSection');
        const bbSignalList = document.getElementById('bbSignalList');

        if (bollinger.signals && bollinger.signals.length > 0) {
            bbSignalSection.style.display = 'block';
            bbSignalList.innerHTML = '';
            bollinger.signals.slice(-5).reverse().forEach(sig => {
                const sigItem = document.createElement('div');
                let typeClass = '';
                if (sig.type.includes('Buy')) typeClass = 'bullish';
                if (sig.type.includes('Sell')) typeClass = 'bearish';

                sigItem.className = `divergence-item ${typeClass}`;
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

        document.getElementById('bollingerChartImage').src = 'data:image/png;base64,' + bollinger.chart_image;
    }

    // ========== Crossover Data ==========
    if (data.crossover && (analysisType === 'all' || analysisType === 'crossover')) {
        const crossover = data.crossover;
        document.getElementById('ema20').textContent = crossover.ema_20 ? crossover.ema_20.toFixed(2) : '--';
        document.getElementById('ema50').textContent = crossover.ema_50 ? crossover.ema_50.toFixed(2) : '--';
        document.getElementById('ema200').textContent = crossover.ema_200 ? crossover.ema_200.toFixed(2) : '--';
        document.getElementById('crossoverTrend').textContent = crossover.trend_status;

        const trendCard = document.getElementById('crossoverTrendCard');
        trendCard.classList.remove('bullish', 'bearish');
        if (crossover.trend_status.includes('Uptrend')) trendCard.classList.add('bullish');
        if (crossover.trend_status.includes('Downtrend')) trendCard.classList.add('bearish');

        const gcSection = document.getElementById('goldenCrossSection');
        const gcInfo = document.getElementById('goldenCrossInfo');

        if (crossover.gc_date) {
            gcSection.style.display = 'block';
            gcInfo.innerHTML = `
                <div class="divergence-item bullish">
                    <div class="divergence-type">Golden Cross</div>
                    <div class="divergence-details">
                        Date: ${crossover.gc_date} | Price: ${crossover.gc_price.toFixed(2)}
                        <br>50 EMA crossed above 200 EMA
                    </div>
                </div>
            `;
        } else {
            gcSection.style.display = 'none';
        }

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

        const statusCard = document.getElementById('donchianStatusCard');
        statusCard.classList.remove('bullish', 'bearish', 'neutral');
        if (donchian.status.includes('Bullish')) statusCard.classList.add('bullish');
        else if (donchian.status.includes('Bearish')) statusCard.classList.add('bearish');
        else statusCard.classList.add('neutral');

        document.getElementById('donchianSignal').textContent = donchian.breakout_signal || 'No Active Signal';

        const donchianSignalSection = document.getElementById('donchianSignalSection');
        const donchianSignalList = document.getElementById('donchianSignalList');

        if (donchian.signals && donchian.signals.length > 0) {
            donchianSignalSection.style.display = 'block';
            donchianSignalList.innerHTML = '';
            donchian.signals.slice(-5).reverse().forEach(sig => {
                let typeClass = '';
                if (sig.type.includes('Bullish')) typeClass = 'bullish';
                if (sig.type.includes('Bearish')) typeClass = 'bearish';

                const sigItem = document.createElement('div');
                sigItem.className = `divergence-item ${typeClass}`;
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

        document.getElementById('donchianChartImage').src = 'data:image/png;base64,' + donchian.chart_image;
    }

    // ========== RSI Divergence Data ==========
    if (data.rsi && (analysisType === 'all' || analysisType === 'rsi')) {
        const rsi = data.rsi;
        document.getElementById('rsiCurrent').textContent = rsi.current_rsi ? rsi.current_rsi.toFixed(2) : '--';

        if (rsi.chart_image) {
            document.getElementById('rsiChartImage').src = 'data:image/png;base64,' + rsi.chart_image;
        }

        const rsiDivergenceList = document.getElementById('rsiDivergenceList');
        const rsiDivergenceSection = document.getElementById('rsiDivergenceSection');
        rsiDivergenceList.innerHTML = '';

        if (rsi.divergences && rsi.divergences.length > 0) {
            rsiDivergenceSection.style.display = 'block';
            rsi.divergences.slice().reverse().forEach(divData => {
                const div = document.createElement('div');
                div.className = `divergence-item ${divData.type.includes('Bullish') ? 'bullish' : 'bearish'}`;
                div.innerHTML = `
                    <div class="divergence-type">${divData.type}</div>
                    <div class="divergence-details">
                        Date: ${divData.date} | Price: ${divData.price.toFixed(2)}
                        <br>${divData.details}
                    </div>
                `;
                rsiDivergenceList.appendChild(div);
            });
        } else {
            rsiDivergenceSection.style.display = 'none';
        }
    }

    // ========== RSI-Volume Divergence Data ==========
    if (data.rsi_volume && (analysisType === 'all' || analysisType === 'rsi_volume')) {
        const rsiVol = data.rsi_volume;
        document.getElementById('rsiVolumeCurrent').textContent = rsiVol.current_rsi ? rsiVol.current_rsi.toFixed(2) : '--';
        document.getElementById('rsiVolumeCurrentVol').textContent = rsiVol.current_volume ? rsiVol.current_volume.toLocaleString() : '--';
        document.getElementById('rsiVolumeMA20').textContent = rsiVol.volume_ma_20 ? rsiVol.volume_ma_20.toLocaleString() : '--';
        document.getElementById('rsiVolumeMA50').textContent = rsiVol.volume_ma_50 ? rsiVol.volume_ma_50.toLocaleString() : '--';

        if (rsiVol.chart_image) {
            document.getElementById('rsiVolumeChartImage').src = 'data:image/png;base64,' + rsiVol.chart_image;
        }

        // Bullish
        const bullishSection = document.getElementById('rsiVolumeBullishSection');
        const bullishList = document.getElementById('rsiVolumeBullishList');
        bullishList.innerHTML = '';
        if (rsiVol.bullish_divergences && rsiVol.bullish_divergences.length > 0) {
            bullishSection.style.display = 'block';
            rsiVol.bullish_divergences.slice().reverse().forEach(divData => {
                const div = document.createElement('div');
                div.className = 'divergence-item bullish';
                div.innerHTML = `
                    <div class="divergence-type"><strong>${divData.type}</strong></div>
                    <div class="divergence-details">
                        Date: ${divData.date} | Price: ${divData.price.toFixed(2)}<br>
                        RSI: ${divData.rsi.toFixed(2)} | Volume: ${divData.volume.toLocaleString()}<br>
                        ${divData.details}
                    </div>
                `;
                bullishList.appendChild(div);
            });
        } else {
            bullishSection.style.display = 'none';
        }

        // Bearish
        const bearishSection = document.getElementById('rsiVolumeBearishSection');
        const bearishList = document.getElementById('rsiVolumeBearishList');
        bearishList.innerHTML = '';
        if (rsiVol.bearish_divergences && rsiVol.bearish_divergences.length > 0) {
            bearishSection.style.display = 'block';
            rsiVol.bearish_divergences.slice().reverse().forEach(divData => {
                const div = document.createElement('div');
                div.className = 'divergence-item bearish';
                div.innerHTML = `
                    <div class="divergence-type"><strong>${divData.type}</strong></div>
                    <div class="divergence-details">
                        Date: ${divData.date} | Price: ${divData.price.toFixed(2)}<br>
                        RSI: ${divData.rsi.toFixed(2)} | Volume: ${divData.volume.toLocaleString()}<br>
                        ${divData.details}
                    </div>
                `;
                bearishList.appendChild(div);
            });
        } else {
            bearishSection.style.display = 'none';
        }

        // Reversals
        const reversalSection = document.getElementById('rsiVolumeReversalSection');
        const reversalList = document.getElementById('rsiVolumeReversalList');
        reversalList.innerHTML = '';
        if (rsiVol.early_reversals && rsiVol.early_reversals.length > 0) {
            reversalSection.style.display = 'block';
            rsiVol.early_reversals.slice().reverse().forEach(revData => {
                const div = document.createElement('div');
                div.className = `divergence-item ${revData.type.includes('Bullish') ? 'bullish' : 'bearish'}`;
                div.innerHTML = `
                    <div class="divergence-type"><strong>⭐ ${revData.type}</strong></div>
                    <div class="divergence-details">
                        Date: ${revData.date} | Price: ${revData.price.toFixed(2)}<br>
                        RSI: ${revData.rsi.toFixed(2)} | Volume: ${revData.volume.toLocaleString()}<br>
                        ${revData.details}
                    </div>
                `;
                reversalList.appendChild(div);
            });
        } else {
            reversalSection.style.display = 'none';
        }
    }

    // ========== Volatility Squeeze Data ==========
    if (data.volatility_squeeze && (analysisType === 'all' || analysisType === 'volatility_squeeze')) {
        const volSqueeze = data.volatility_squeeze;
        document.getElementById('volatilitySqueezeCurrentBBWidth').textContent = volSqueeze.current_bb_width ? volSqueeze.current_bb_width.toFixed(4) : '--';
        document.getElementById('volatilitySqueezeCurrentATR').textContent = volSqueeze.current_atr ? volSqueeze.current_atr.toFixed(2) : '--';

        if (volSqueeze.chart_image) {
            document.getElementById('volatilitySqueezeChartImage').src = 'data:image/png;base64,' + volSqueeze.chart_image;
        }

        const signalsSection = document.getElementById('volatilitySqueezeSignalsSection');
        const signalsList = document.getElementById('volatilitySqueezeSignalsList');
        signalsList.innerHTML = '';

        if (volSqueeze.signals && volSqueeze.signals.length > 0) {
            signalsSection.style.display = 'block';
            volSqueeze.signals.slice().reverse().forEach(sigData => {
                let typeClass = '';
                if (sigData.type.includes('Bullish Breakout')) typeClass = 'bullish';
                else if (sigData.type.includes('Bearish Breakout')) typeClass = 'bearish';
                else if (sigData.type.includes('BB Squeeze')) typeClass = 'neutral';

                const div = document.createElement('div');
                div.className = `divergence-item ${typeClass}`;
                div.innerHTML = `
                    <div class="divergence-type ${typeClass}"><strong>${sigData.type}</strong></div>
                    <div class="divergence-details">
                        Date: ${sigData.date} | Price: ${sigData.price.toFixed(2)}<br>
                        BB Width: ${sigData.bb_width.toFixed(4)} | ATR: ${sigData.atr.toFixed(2)}
                    </div>
                `;
                signalsList.appendChild(div);
            });
        } else {
            signalsSection.style.display = 'none';
        }
    }

    // ========== RS Analysis Data ==========
    if (data.rs && (analysisType === 'all' || analysisType === 'rs')) {
        const rs = data.rs;
        document.getElementById('rsBenchmark').textContent = rs.benchmark || '--';
        document.getElementById('rsSector').textContent = rs.sector || 'N/A';
        document.getElementById('rsClassification').textContent = rs.classification || '--';
        document.getElementById('rsScore').textContent = rs.rs_score ? rs.rs_score.toFixed(1) : '--';

        document.getElementById('rs1M').textContent = rs.rs_ratios['1M'] ? rs.rs_ratios['1M'].toFixed(3) : '--';
        document.getElementById('rs3M').textContent = rs.rs_ratios['3M'] ? rs.rs_ratios['3M'].toFixed(3) : '--';
        document.getElementById('rs6M').textContent = rs.rs_ratios['6M'] ? rs.rs_ratios['6M'].toFixed(3) : '--';
        document.getElementById('rs1Y').textContent = rs.rs_ratios['1Y'] ? rs.rs_ratios['1Y'].toFixed(3) : '--';

        const classificationCard = document.getElementById('rsClassificationCard');
        classificationCard.classList.remove('bullish', 'bearish', 'neutral');
        if (rs.classification.includes('Leader') && !rs.classification.includes('Weakening')) classificationCard.classList.add('bullish');
        else if (rs.classification.includes('Laggard')) classificationCard.classList.add('bearish');
        else classificationCard.classList.add('neutral');

        const tradingSummarySection = document.getElementById('rsTradingSummarySection');
        const tradingSummary = document.getElementById('rsTradingSummary');
        if (rs.trading_summary) {
            tradingSummarySection.style.display = 'block';
            tradingSummary.textContent = rs.trading_summary;
        } else {
            tradingSummarySection.style.display = 'none';
        }

        const signalsSection = document.getElementById('rsSignalsSection');
        const signalsList = document.getElementById('rsSignalsList');
        signalsList.innerHTML = '';

        if (rs.signals && rs.signals.length > 0) {
            signalsSection.style.display = 'block';
            rs.signals.slice().reverse().forEach(sigData => {
                let typeClass = 'neutral';
                if (sigData.type.includes('Leader')) typeClass = 'bullish';
                else if (sigData.type.includes('Laggard')) typeClass = 'bearish';

                const div = document.createElement('div');
                div.className = `divergence-item ${typeClass}`;
                div.innerHTML = `
                    <div class="divergence-type ${typeClass}"><strong>${sigData.type}</strong></div>
                    <div class="divergence-details">
                        Date: ${sigData.date}<br>
                        ${sigData.description}
                    </div>
                `;
                signalsList.appendChild(div);
            });
        } else {
            signalsSection.style.display = 'none';
        }

        if (rs.chart_image) {
            document.getElementById('rsChartImage').src = 'data:image/png;base64,' + rs.chart_image;
        }
    }
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = '❌ ' + message;
    errorDiv.style.display = 'block';

    // Hide after 5 seconds
    setTimeout(() => {
        errorDiv.style.display = 'none';
    }, 5000);
}

// Allow Enter key to trigger analysis
const tickerInput = document.getElementById('tickerInput');
if (tickerInput) {
    tickerInput.addEventListener('keypress', function (event) {
        if (event.key === 'Enter') {
            analyzeStock('all');
        }
    });
}

// Initialize application
function initApp() {
    fetchBenchmarks();
    loadStockData();
    loadSectors();

    // Check if Sector Analysis is active and load it
    const sectorSection = document.getElementById('sector_analysis-section');
    if (sectorSection && sectorSection.classList.contains('active')) {
        loadSectorAnalysis();
    }

    // Restore last analyzed stock on Dashboard
    if (window.location.pathname === '/' || window.location.pathname === '/index.html') {
        restoreLastAnalyzedStock();
    }

    // Auto-run batch analysis on Batch Analysis page
    if (window.location.pathname === '/batch_analysis' || window.location.pathname === '/batch' || window.location.pathname.includes('batch')) {
        // Only run if no cache exists
        if (!batchAnalysisCache) {
            runBatchAnalysis();
        } else {
            // Restore from cache
            console.log('Restoring batch analysis from cache');
            renderBatchTable(batchAnalysisCache);
            const updatedEl = document.getElementById('lastUpdated');
            if (updatedEl) updatedEl.textContent = 'Last Updated: ' + new Date().toLocaleString();
        }
    }
}

// Restore last analyzed stock from cache
function restoreLastAnalyzedStock() {
    if (lastAnalyzedStock && lastAnalyzedStock.data) {
        console.log('Restoring last analyzed stock:', lastAnalyzedStock.ticker);

        // Set the ticker input
        const tickerInput = document.getElementById('tickerInput');
        if (tickerInput) {
            tickerInput.value = lastAnalyzedStock.ticker;
        }

        // Display the results
        displayResults(lastAnalyzedStock.data, lastAnalyzedStock.analysisType);

        // Restore fundamental analysis if available in cache for this ticker
        if (typeof fundamentalsCache !== 'undefined' && fundamentalsCache[lastAnalyzedStock.ticker]) {
            displayFundamentalResults(fundamentalsCache[lastAnalyzedStock.ticker]);
        }

        // Update timestamp
        const timestampEl = document.getElementById('timestamp');
        if (timestampEl && lastAnalyzedStock.timestamp) {
            const date = new Date(lastAnalyzedStock.timestamp);
            timestampEl.textContent = 'Last Updated: ' + date.toLocaleString();
        }
    }
}

// Check if DOM is already loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
} else {
    initApp();
}

async function fetchBenchmarks() {
    try {
        const response = await fetch('/api/benchmarks');
        const data = await response.json();

        if (data.success && data.benchmarks) {
            const select = document.getElementById('rsBenchmarkSelect');
            if (select) {
                data.benchmarks.forEach(bench => {
                    const option = document.createElement('option');
                    option.value = bench.symbol;
                    option.textContent = `${bench.name} (${bench.symbol})`;
                    select.appendChild(option);
                });
            }
        }
    } catch (error) {
        console.error('Failed to fetch benchmarks:', error);
    }
}

async function loadStockData() {
    try {
        // 1. Fetch Watchlist
        const watchlistResponse = await fetch('/api/watchlist');
        const watchlistData = await watchlistResponse.json();

        if (watchlistData.success && watchlistData.watchlist) {
            const watchlist = watchlistData.watchlist;

            // Populate Sidebar Watchlist
            const sidebarList = document.getElementById('stockList');
            if (sidebarList) {
                sidebarList.innerHTML = ''; // Clear existing
                for (const [name, symbol] of Object.entries(watchlist)) {
                    const li = document.createElement('li');
                    li.className = 'stock-item';
                    li.onclick = () => selectStock(symbol);
                    li.innerHTML = `
                        <span class="stock-symbol">${symbol}</span>
                        <span class="stock-name">${name}</span>
                    `;
                    sidebarList.appendChild(li);
                }
            }

            // Populate Settings Watchlist Listbox
            const settingsWatchlist = document.getElementById('watchlistSettings');
            if (settingsWatchlist) {
                settingsWatchlist.innerHTML = '';
                for (const [name, symbol] of Object.entries(watchlist)) {
                    const option = document.createElement('option');
                    option.value = symbol;
                    option.textContent = `${name} (${symbol})`;
                    settingsWatchlist.appendChild(option);
                }
            }
        } else {
            console.error('Watchlist API failed:', watchlistData.error);
            showError('Failed to load watchlist: ' + (watchlistData.error || 'Unknown error'));
        }

        // 2. Fetch Tickers List (for Settings)
        const tickersResponse = await fetch('/api/tickers');
        const tickersData = await tickersResponse.json();

        if (tickersData.success && tickersData.tickers) {
            const tickers = tickersData.tickers;

            // Populate Settings Tickers Listbox
            const settingsTickers = document.getElementById('tickersList');
            if (settingsTickers) {
                settingsTickers.innerHTML = '';
                for (const [name, symbol] of Object.entries(tickers)) {
                    const option = document.createElement('option');
                    option.value = symbol;
                    option.textContent = `${name} (${symbol})`;
                    settingsTickers.appendChild(option);
                }
            }
        } else {
            console.error('Tickers API failed:', tickersData.error);
            showError('Failed to load tickers: ' + (tickersData.error || 'Unknown error'));
        }

    } catch (error) {
        console.error('Error loading stock data:', error);
        showError('Failed to load stock lists');
    }
}

// ========== Sector Analysis Logic ==========
async function loadSectorAnalysis(forceRefresh = false) {
    console.log('loadSectorAnalysis called, forceRefresh:', forceRefresh);
    const loadingEl = document.getElementById('sectorLoading');
    const contentEl = document.getElementById('sectorContent');

    // If we have cached data and not forcing refresh, use cache
    if (sectorAnalysisCache && !forceRefresh) {
        console.log('Using cached sector analysis data');
        displaySectorAnalysis(sectorAnalysisCache);
        return;
    }

    if (loadingEl) loadingEl.style.display = 'flex';
    if (contentEl) contentEl.style.display = 'none';

    try {
        console.log('Fetching sector analysis data from API...');
        const response = await fetch('/api/market_analysis/sector');
        const data = await response.json();
        console.log('Sector analysis data received:', data);

        if (data.success) {
            // Cache the results
            sectorAnalysisCache = data;

            // Save to localStorage
            try {
                localStorage.setItem('sectorAnalysisCache', JSON.stringify(data));
            } catch (error) {
                console.error('Error saving sector cache to localStorage:', error);
            }

            // Display the data
            displaySectorAnalysis(data);
        } else {
            console.error('Sector analysis failed:', data.error);
            showError('Sector analysis failed: ' + data.error);
        }
    } catch (error) {
        console.error('Error loading sector analysis:', error);
        showError('Error loading sector analysis');
    } finally {
        if (loadingEl) loadingEl.style.display = 'none';
    }
}

// Helper function to display sector analysis data
function displaySectorAnalysis(data) {
    const contentEl = document.getElementById('sectorContent');

    // Display Chart
    const chartImg = document.getElementById('sectorChartImage');
    if (chartImg) {
        chartImg.src = 'data:image/png;base64,' + data.chart_image;
    }

    // Populate Table
    const tableBody = document.querySelector('#sectorTable tbody');
    if (tableBody) {
        tableBody.innerHTML = '';

        data.data.forEach(row => {
            const tr = document.createElement('tr');

            // Helper for boolean icons
            const getIcon = (val) => val ? '<i class="fas fa-check-circle text-success"></i>' : '<span class="text-muted">-</span>';

            tr.innerHTML = `
                <td class="fw-bold">${row.Sector}</td>
                <td><span class="badge bg-primary">${row.Score}</span></td>
                <td>${row['1M'] || '-'}</td>
                <td>${row['3M'] || '-'}</td>
                <td>${row['6M'] || '-'}</td>
                <td>${row['1Y'] || '-'}</td>
                <td class="text-center">${getIcon(row.Consistent)}</td>
                <td class="text-center">${getIcon(row.Emerging)}</td>
                <td class="text-center">${getIcon(row.Early_Turnaround)}</td>
                <td class="text-center">${getIcon(row.MA_Breakout)}</td>
                <td class="text-center">${getIcon(row.Volume_Surge)}</td>
            `;
            tableBody.appendChild(tr);
        });
    }

    if (contentEl) contentEl.style.display = 'block';
}

// ========== Stocks in Sector Analysis Logic ==========

// Load available sectors into dropdown
async function loadSectors() {
    try {
        const response = await fetch('/api/sectors');
        const data = await response.json();

        if (data.success && data.sectors) {
            const select = document.getElementById('sectorSelect');
            if (select) {
                // Clear existing options except the first one
                select.innerHTML = '<option value="">Select a Sector...</option>';

                // Add sector options
                data.sectors.forEach(sector => {
                    const option = document.createElement('option');
                    option.value = sector.name;
                    option.textContent = `${sector.name} (${sector.index_symbol})`;
                    select.appendChild(option);
                });
            }
        } else {
            console.error('Failed to load sectors:', data.error);
        }
    } catch (error) {
        console.error('Error loading sectors:', error);
    }
}

// Analyze stocks in selected sector
async function analyzeStocksInSector(forceRefresh = false) {
    const select = document.getElementById('sectorSelect');
    const sectorName = select.value;

    if (!sectorName) {
        showError('Please select a sector first');
        return;
    }

    console.log('analyzeStocksInSector called, sector:', sectorName, 'forceRefresh:', forceRefresh);

    const loadingEl = document.getElementById('stocksInSectorLoading');
    const contentEl = document.getElementById('stocksInSectorContent');

    // If we have cached data for this sector and not forcing refresh, use cache
    if (stocksInSectorCache[sectorName] && !forceRefresh) {
        console.log('Using cached stocks in sector data for', sectorName);
        displayStocksInSector(stocksInSectorCache[sectorName]);
        return;
    }

    if (loadingEl) loadingEl.style.display = 'flex';
    if (contentEl) contentEl.style.display = 'none';

    try {
        console.log('Fetching stocks in sector data from API for', sectorName);
        const response = await fetch('/api/market_analysis/stocks_in_sector', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ sector: sectorName })
        });
        const data = await response.json();
        console.log('Stocks in sector data received:', data);

        if (data.success) {
            // Cache the results for this sector
            stocksInSectorCache[sectorName] = data;

            // Save to localStorage
            try {
                localStorage.setItem('stocksInSectorCache', JSON.stringify(stocksInSectorCache));
            } catch (error) {
                console.error('Error saving stocks cache to localStorage:', error);
            }

            // Display the data
            displayStocksInSector(data);
        } else {
            console.error('Stocks in sector analysis failed:', data.error);
            showError('Analysis failed: ' + data.error);
        }
    } catch (error) {
        console.error('Error loading stocks in sector analysis:', error);
        showError('Error loading analysis');
    } finally {
        if (loadingEl) loadingEl.style.display = 'none';
    }
}

// Helper function to display stocks in sector analysis data
function displayStocksInSector(data) {
    const contentEl = document.getElementById('stocksInSectorContent');

    // Update title and index info
    const titleEl = document.getElementById('stockSectorTitle');
    const indexEl = document.getElementById('stockSectorIndex');
    if (titleEl) titleEl.textContent = `${data.sector_name} Sector Analysis`;
    if (indexEl) indexEl.textContent = `Benchmark: ${data.sector_index}`;

    // Display Chart
    const chartImg = document.getElementById('stocksInSectorChartImage');
    if (chartImg) {
        chartImg.src = 'data:image/png;base64,' + data.chart_image;
    }

    // Populate Table
    const tableBody = document.querySelector('#stocksInSectorTable tbody');
    if (tableBody) {
        tableBody.innerHTML = '';

        data.data.forEach(row => {
            const tr = document.createElement('tr');

            // Helper for boolean icons
            const getIcon = (val) => val ? '<i class="fas fa-check-circle text-success"></i>' : '<span class="text-muted">-</span>';

            tr.innerHTML = `
                <td class="fw-bold">${row.Stock}</td>
                <td><span class="badge bg-primary">${row.Score}</span></td>
                <td>${row['1M'] || '-'}</td>
                <td>${row['3M'] || '-'}</td>
                <td>${row['6M'] || '-'}</td>
                <td>${row['1Y'] || '-'}</td>
                <td class="text-center">${getIcon(row.Consistent)}</td>
                <td class="text-center">${getIcon(row.Emerging)}</td>
                <td class="text-center">${getIcon(row.Early_Turnaround)}</td>
                <td class="text-center">${getIcon(row.MA_Breakout)}</td>
                <td class="text-center">${getIcon(row.Volume_Surge)}</td>
            `;
            tableBody.appendChild(tr);
        });
    }

    if (contentEl) contentEl.style.display = 'block';
}

// ========== Batch Analysis Logic ==========

let batchResultsCache = [];

async function runBatchAnalysis(forceRefresh = false) {
    // Read interval from selector
    const intervalSelect = document.getElementById('intervalSelect');
    const selectedInterval = intervalSelect ? intervalSelect.value : '1d';

    // Check if interval changed - if so, invalidate cache
    const cachedInterval = localStorage.getItem('batchAnalysisInterval');
    if (cachedInterval && cachedInterval !== selectedInterval) {
        console.log(`Interval changed from ${cachedInterval} to ${selectedInterval}, clearing cache`);
        batchAnalysisCache = null;
        localStorage.removeItem('batchAnalysisCache');
    }

    // Store current interval
    localStorage.setItem('batchAnalysisInterval', selectedInterval);

    // Check cache first
    if (batchAnalysisCache && !forceRefresh) {
        console.log('Using cached batch analysis data');
        renderBatchTable(batchAnalysisCache);
        const updatedEl = document.getElementById('lastUpdated');
        if (updatedEl) updatedEl.textContent = `Last Updated: ${new Date().toLocaleString()} (${selectedInterval})`;
        return;
    }

    // Simplified: No sector selection needed
    const progressEl = document.getElementById('batchProgress');
    const tableBody = document.getElementById('batchTableBody');
    const runBtn = document.querySelector('.action-btn');

    if (progressEl) progressEl.style.display = 'block';
    if (runBtn) runBtn.disabled = true;
    if (tableBody) tableBody.innerHTML = '<tr><td colspan="9" class="text-center"><i class="fas fa-spinner fa-spin"></i> Processing batch... This may take 10-30 seconds.</td></tr>';

    try {
        const response = await fetch('/api/batch_analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ interval: selectedInterval })
        });

        const data = await response.json();

        if (data.success) {
            batchResultsCache = data.data;
            batchAnalysisCache = data.data; // Cache for persistence

            // Save to localStorage
            try {
                localStorage.setItem('batchAnalysisCache', JSON.stringify(data.data));
            } catch (error) {
                console.error('Error saving batch cache to localStorage:', error);
            }

            renderBatchTable(batchResultsCache);

            // Update last updated with interval info
            const updatedEl = document.getElementById('lastUpdated');
            if (updatedEl) updatedEl.textContent = `Last Updated: ${new Date().toLocaleString()} (${data.interval || selectedInterval})`;
        } else {
            console.error('Batch analysis failed:', data.error);
            if (tableBody) tableBody.innerHTML = `<tr><td colspan="9" class="text-center text-danger">Error: ${data.error}</td></tr>`;
        }
    } catch (error) {
        console.error('Error running batch analysis:', error);
        if (tableBody) tableBody.innerHTML = `<tr><td colspan="9" class="text-center text-danger">Network Error</td></tr>`;
    } finally {
        if (progressEl) progressEl.style.display = 'none';
        if (runBtn) runBtn.disabled = false;
    }
}

function renderBatchTable(data) {
    const tableBody = document.getElementById('batchTableBody');
    if (!tableBody) return;

    tableBody.innerHTML = '';

    if (!data || data.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="9" class="text-center">No results found.</td></tr>';
        return;
    }

    data.forEach(item => {
        if (!item.success) {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="fw-bold">${item.ticker}</td>
                <td colspan="8" class="text-danger">Error: ${item.error || 'Analysis failed'}</td>
            `;
            tableBody.appendChild(tr);
            return;
        }

        const tr = document.createElement('tr');
        const columns = item.columns;
        const score = item.score;

        // Determine score class based on strategy interpretation
        let scoreClass = 'score-low';
        if (score >= 80) scoreClass = 'score-high';      // Strong Buy
        else if (score >= 60) scoreClass = 'score-med';  // Buy/Accumulate
        else if (score >= 40) scoreClass = 'score-low';  // Hold/Neutral
        else scoreClass = 'score-low';                   // Avoid/Sell

        // Format Trend Direction with details
        let trendHtml = columns.trend_direction;
        if (columns.trend_direction_detail) {
            trendHtml += `<br><small class="text-muted">MACD: ${columns.trend_direction_detail.macd}</small>`;
            trendHtml += `<br><small class="text-muted">ST: ${columns.trend_direction_detail.supertrend}</small>`;
        }

        // Format Trend Signal with details
        let signalHtml = columns.trend_signal;
        if (columns.trend_signal_detail) {
            signalHtml += `<br><small class="text-muted">Cross: ${columns.trend_signal_detail.crossover}</small>`;
            signalHtml += `<br><small class="text-muted">ST: ${columns.trend_signal_detail.supertrend}</small>`;
        }

        // Format RSI with value and state
        let rsiHtml = `<strong>${columns.rsi_value}</strong>`;
        rsiHtml += `<br><small>${columns.rsi_state}</small>`;

        // Format Divergence with date
        let divHtml = columns.divergence;
        if (columns.divergence_date && columns.divergence !== '➖ None') {
            const divDate = new Date(columns.divergence_date);
            divHtml += `<br><small class="text-muted">${divDate.toLocaleDateString()}</small>`;
        }

        // Format RS Score with classification
        let rsHtml = `<strong>${columns.rs_score}</strong>`;
        if (columns.rs_classification) {
            rsHtml += `<br><small class="text-muted">${columns.rs_classification}</small>`;
        }

        tr.innerHTML = `
            <td class="fw-bold">${item.ticker}</td>
            <td>₹${item.price ? item.price.toFixed(2) : '-'}</td>
            <td>${trendHtml}</td>
            <td>${signalHtml}</td>
            <td>${rsiHtml}</td>
            <td>${divHtml}</td>
            <td>${columns.squeeze}</td>
            <td>${rsHtml}</td>
            <td><span class="score-badge ${scoreClass}">${score}</span></td>
        `;

        tableBody.appendChild(tr);
    });
}

function exportBatchCSV() {
    if (!batchResultsCache || batchResultsCache.length === 0) {
        alert('No data to export');
        return;
    }

    const headers = ['Ticker', 'Price', 'Trend', 'Momentum', 'RSI', 'RS_Score', 'Score'];
    const rows = batchResultsCache.map(d => [
        d.ticker,
        d.price,
        d.metrics.trend,
        d.metrics.momentum,
        d.metrics.rsi,
        d.metrics.rs_score,
        d.score
    ]);

    let csvContent = "data:text/csv;charset=utf-8,"
        + headers.join(",") + "\n"
        + rows.map(e => e.join(",")).join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "batch_analysis_results.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function sortTable(n) {
    const table = document.getElementById("batchTable");
    let rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
    switching = true;
    dir = "asc";

    while (switching) {
        switching = false;
        rows = table.rows;

        for (i = 1; i < (rows.length - 1); i++) {
            shouldSwitch = false;
            x = rows[i].getElementsByTagName("TD")[n];
            y = rows[i + 1].getElementsByTagName("TD")[n];

            let xVal = x.textContent || x.innerText;
            let yVal = y.textContent || y.innerText;

            // Numeric sort for Price (1), RSI (3), RS (5), Score (6)
            if ([1, 3, 5, 6].includes(n)) {
                xVal = parseFloat(xVal) || 0;
                yVal = parseFloat(yVal) || 0;
            }

            if (dir == "asc") {
                if (xVal > yVal) { shouldSwitch = true; break; }
            } else if (dir == "desc") {
                if (xVal < yVal) { shouldSwitch = true; break; }
            }
        }

        if (shouldSwitch) {
            rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
            switching = true;
            switchcount++;
        } else {
            if (switchcount == 0 && dir == "asc") {
                dir = "desc";
                switching = true;
            }
        }
    }
}
