/**
 * FUNDAMENTAL ANALYSIS JAVASCRIPT
 * Add this to the existing script.js or include inline in dashboard.html
 */

// Cache for fundamental analysis results
const fundamentalsCache = {};

/**
 * Analyze fundamentals for a ticker
 */
async function analyzeFundamentals(ticker) {
    // Check cache first
    if (fundamentalsCache[ticker]) {
        console.log(`Using cached fundamentals for ${ticker}`);
        displayFundamentalResults(fundamentalsCache[ticker]);
        return;
    }

    // Show loading
    document.getElementById('fundamentals-loading').style.display = 'block';
    document.getElementById('fundamentals-results').style.display = 'none';
    document.getElementById('fundamentals-error').style.display = 'none';

    try {
        const response = await fetch('/api/fundamental-analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker })
        });

        const data = await response.json();

        if (data.success) {
            // Cache the results
            fundamentalsCache[ticker] = data;
            displayFundamentalResults(data);
        } else {
            showFundamentalsError(data.error || 'Analysis failed');
        }
    } catch (error) {
        showFundamentalsError('Network error: ' + error.message);
    } finally {
        document.getElementById('fundamentals-loading').style.display = 'none';
    }
}

/**
 * Display fundamental analysis results
 */
function displayFundamentalResults(data) {
    document.getElementById('fundamentals-results').style.display = 'block';
    document.getElementById('fundamentals-error').style.display = 'none';

    const lt = data.long_term;
    const st = data.short_term;

    // Long-term Revenue (5Y)
    displayLongTermRow('revenue', lt.revenue_4y);

    // Long-term Profit (5Y)
    displayLongTermRow('profit', lt.profit_4y);

    // Long-term ROE (5Y)
    displayLongTermRow('roe', lt.roe_4y);

    // Long-term EPS (5Y)
    displayLongTermRow('eps', lt.eps_4y);

    // PE Ratio
    displayPERow(lt.pe_ratio);

    // Short-term Revenue (6Q)
    displayShortTermRow('revenue', st.revenue_6q);

    // Short-term Profit (6Q)
    displayShortTermRow('profit', st.profit_6q);

    // Short-term ROE (6Q)
    displayShortTermRow('roe', st.roe_6q);

    // Short-term EPS (6Q)
    displayShortTermRow('eps', st.eps_6q);

    // Timestamp
    document.getElementById('fundamentals-timestamp').textContent =
        `Analysis Date: ${data.analysis_date}`;
}

/**
 * Display long-term row (5Y)
 */
function displayLongTermRow(metric, data) {
    if (!data.success) {
        // Handle error case
        document.getElementById(`${metric}-5y-growing`).innerHTML =
            `<span class="badge badge-neutral">N/A</span>`;
        document.getElementById(`${metric}-5y-accelerating`).innerHTML =
            `<span class="badge badge-neutral">N/A</span>`;
        document.getElementById(`${metric}-5y-1y`).textContent = 'N/A';
        document.getElementById(`${metric}-5y-3y`).textContent = 'N/A';
        document.getElementById(`${metric}-5y-status`).innerHTML =
            `<span class="badge badge-neutral">${data.error}</span>`;
        return;
    }

    // Is Growing?
    document.getElementById(`${metric}-5y-growing`).innerHTML =
        data.is_growing ?
            '<span class="badge badge-success">✓ Yes</span>' :
            '<span class="badge badge-danger">✗ No</span>';

    // Accelerating?
    document.getElementById(`${metric}-5y-accelerating`).innerHTML =
        data.has_accelerating_trend ?
            '<span class="badge badge-success">✓ Yes</span>' :
            '<span class="badge badge-warning">→ No</span>';

    // Growth rates
    document.getElementById(`${metric}-5y-1y`).textContent = `${data.growth_1y}%`;
    document.getElementById(`${metric}-5y-3y`).textContent = `${data.growth_3y_cagr}%`;

    // Status
    let statusClass = 'badge-neutral';
    let statusText = 'Stable';

    if (data.is_growing && data.has_accelerating_trend) {
        statusClass = 'badge-success';
        statusText = 'Strong Growth';
    } else if (data.is_growing) {
        statusClass = 'badge-success';
        statusText = 'Growing';
    } else if (data.growth_3y_cagr < -5) {
        statusClass = 'badge-danger';
        statusText = 'Declining';
    } else {
        statusClass = 'badge-warning';
        statusText = 'Flat';
    }

    document.getElementById(`${metric}-5y-status`).innerHTML =
        `<span class="badge ${statusClass}">${statusText}</span>`;
}

/**
 * Display PE ratio row
 */
function displayPERow(data) {
    if (!data.success) {
        document.getElementById('pe-value').textContent = 'N/A';
        document.getElementById('pe-industry').textContent = 'N/A';
        document.getElementById('pe-vs-industry').innerHTML =
            `<span class="badge badge-neutral">${data.error}</span>`;
        return;
    }

    document.getElementById('pe-value').textContent =
        `Current: ${data.current_pe}`;

    if (data.industry_pe) {
        document.getElementById('pe-industry').textContent =
            `Industry: ${data.industry_pe}`;

        let statusClass = 'badge-neutral';
        if (data.is_overvalued) {
            statusClass = 'badge-danger';
        } else if (data.is_undervalued) {
            statusClass = 'badge-success';
        }

        document.getElementById('pe-vs-industry').innerHTML =
            `<span class="badge ${statusClass}">${data.vs_industry}</span>`;
    } else {
        document.getElementById('pe-industry').textContent =
            `${data.industry} | ${data.sector}`;
        document.getElementById('pe-vs-industry').innerHTML =
            '<span class="badge badge-neutral">Industry PE N/A</span>';
    }
}

/**
 * Display short-term row (6Q)
 */
function displayShortTermRow(metric, data) {
    if (!data.success) {
        document.getElementById(`${metric}-6q-growing`).innerHTML =
            `<span class="badge badge-neutral">N/A</span>`;
        document.getElementById(`${metric}-6q-recent`).textContent = 'N/A';
        document.getElementById(`${metric}-6q-avg`).textContent = 'N/A';
        document.getElementById(`${metric}-6q-status`).innerHTML =
            `<span class="badge badge-neutral">${data.error}</span>`;
        return;
    }

    // Is Growing?
    document.getElementById(`${metric}-6q-growing`).innerHTML =
        data.is_growing ?
            '<span class="badge badge-success">✓ Yes</span>' :
            `<span class="badge badge-danger">✗ No</span>`;

    // Recent QoQ
    const recentGrowth = data.recent_quarter_growth || 0;
    document.getElementById(`${metric}-6q-recent`).textContent = `${recentGrowth}%`;

    // Avg QoQ
    const avgGrowth = data.average_qoq_growth || 0;
    document.getElementById(`${metric}-6q-avg`).textContent = `${avgGrowth}%`;

    // Status
    let statusClass = 'badge-neutral';
    let statusText = 'Stable';

    if (data.is_growing && avgGrowth > 5) {
        statusClass = 'badge-success';
        statusText = 'Strong Growth';
    } else if (data.is_growing) {
        statusClass = 'badge-success';
        statusText = 'Growing';
    } else if (avgGrowth < -5) {
        statusClass = 'badge-danger';
        statusText = 'Declining';
    } else {
        statusClass = 'badge-warning';
        statusText = 'Flat';
    }

    document.getElementById(`${metric}-6q-status`).innerHTML =
        `<span class="badge ${statusClass}">${statusText}</span>`;
}

/**
 * Show fundamentals error
 */
function showFundamentalsError(message) {
    document.getElementById('fundamentals-error').style.display = 'block';
    document.getElementById('fundamentals-results').style.display = 'none';
    document.getElementById('fundamentals-error-message').textContent = message;
}
