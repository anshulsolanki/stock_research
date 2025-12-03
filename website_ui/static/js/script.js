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

        const statusCard = document.getElementById('supertrendStatusCard');
        statusCard.classList.remove('uptrend', 'downtrend');
        statusCard.classList.add(supertrend.last_trend === 1 ? 'uptrend' : 'downtrend');

        const summarySection = document.getElementById('supertrendSummarySection');
        const summaryDiv = document.getElementById('supertrendSummary');
        summaryDiv.textContent = `Status: ${supertrend.status}\nLast Price: ${supertrend.last_price.toFixed(2)}\nSupertrend: ${supertrend.supertrend_value.toFixed(2)}\nDate: ${supertrend.last_date}`;
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
document.getElementById('tickerInput').addEventListener('keypress', function (event) {
    if (event.key === 'Enter') {
        analyzeStock('all');
    }
});

// Fetch benchmarks on load
document.addEventListener('DOMContentLoaded', async function () {
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
});
