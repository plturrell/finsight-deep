// Research Dashboard JavaScript
class ResearchDashboard {
    constructor() {
        this.currentPage = 'overview';
        this.researchData = [];
        this.filters = new Set(['all']);
        this.charts = {};
        
        this.initialize();
    }

    async initialize() {
        // Setup event listeners
        this.setupEventListeners();
        
        // Initialize charts
        this.initializeCharts();
        
        // Load initial data
        await this.loadResearchData();
        
        // Start real-time updates
        this.startRealtimeUpdates();
    }

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = e.currentTarget.dataset.page;
                this.navigateTo(page);
            });
        });

        // Search
        const searchInput = document.getElementById('searchInput');
        searchInput.addEventListener('input', (e) => {
            this.searchResearch(e.target.value);
        });

        // Filters
        document.querySelectorAll('.chip').forEach(chip => {
            chip.addEventListener('click', (e) => {
                const filter = e.target.dataset.filter;
                this.toggleFilter(filter);
            });
        });

        // Refresh button
        document.getElementById('refreshBtn').addEventListener('click', () => {
            this.refreshData();
        });

        // New research button
        document.getElementById('newResearchBtn').addEventListener('click', () => {
            this.showNewResearchModal();
        });

        // Modal close
        document.getElementById('closeModal').addEventListener('click', () => {
            this.closeModal();
        });
    }

    initializeCharts() {
        // Market Trend Chart
        const ctx = document.createElement('canvas');
        ctx.id = 'marketTrendChart';
        
        this.charts.marketTrend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Market Index',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#94a3b8'
                        }
                    }
                }
            }
        });
    }

    async loadResearchData() {
        // Show loading state
        this.showLoading();

        try {
            // In production, this would fetch from API
            const response = await this.mockFetchResearch();
            this.researchData = response.data;
            
            // Render research cards
            this.renderResearchCards(this.researchData);
            
            // Update metrics
            this.updateMetrics(response.metrics);
            
            // Update AI analysis
            this.updateAIAnalysis(response.aiAnalysis);
            
        } catch (error) {
            console.error('Error loading research data:', error);
            this.showError('Failed to load research data');
        } finally {
            this.hideLoading();
        }
    }

    renderResearchCards(data) {
        const grid = document.getElementById('researchGrid');
        grid.innerHTML = '';

        data.forEach(item => {
            const card = this.createResearchCard(item);
            grid.appendChild(card);
        });

        // Add fade-in animation
        grid.querySelectorAll('.research-card').forEach((card, index) => {
            card.style.animationDelay = `${index * 0.05}s`;
            card.classList.add('fade-in');
        });
    }

    createResearchCard(item) {
        const card = document.createElement('div');
        card.className = 'research-card';
        card.onclick = () => this.showResearchDetails(item);

        card.innerHTML = `
            <div class="research-header">
                <h3 class="research-title">${item.title}</h3>
                <div class="research-meta">
                    <span>${item.date}</span>
                    <span>${item.author}</span>
                    <span>${item.readTime} min read</span>
                </div>
            </div>
            <div class="research-body">
                <p class="research-summary">${item.summary}</p>
                <div class="research-tags">
                    ${item.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                </div>
            </div>
        `;

        return card;
    }

    showResearchDetails(item) {
        const modal = document.getElementById('researchModal');
        const modalTitle = document.getElementById('modalTitle');
        const modalBody = document.getElementById('modalBody');

        modalTitle.textContent = item.title;
        modalBody.innerHTML = `
            <div class="research-details">
                <div class="research-meta-detailed">
                    <span>By ${item.author}</span>
                    <span>${item.date}</span>
                    <span>${item.readTime} min read</span>
                </div>
                
                <div class="research-content">
                    ${item.content}
                </div>
                
                <div class="research-charts">
                    <canvas id="detailChart"></canvas>
                </div>
                
                <div class="research-recommendations">
                    <h4>Key Recommendations</h4>
                    ${item.recommendations.map(rec => `
                        <div class="recommendation">
                            <strong>${rec.title}</strong>
                            <p>${rec.description}</p>
                        </div>
                    `).join('')}
                </div>
                
                <div class="research-actions">
                    <button class="btn btn-primary" onclick="dashboard.applyResearch('${item.id}')">
                        Apply to Portfolio
                    </button>
                    <button class="btn btn-secondary" onclick="dashboard.shareResearch('${item.id}')">
                        Share Research
                    </button>
                </div>
            </div>
        `;

        // Create detail chart
        this.createDetailChart(item);

        modal.classList.add('open');
    }

    createDetailChart(item) {
        setTimeout(() => {
            const ctx = document.getElementById('detailChart');
            if (!ctx) return;

            new Chart(ctx, {
                type: item.chartType || 'line',
                data: item.chartData || this.generateMockChartData(),
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#94a3b8'
                            }
                        }
                    },
                    scales: {
                        y: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.05)'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.05)'
                            },
                            ticks: {
                                color: '#94a3b8'
                            }
                        }
                    }
                }
            });
        }, 100);
    }

    generateMockChartData() {
        const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
        return {
            labels: labels,
            datasets: [{
                label: 'Performance',
                data: labels.map(() => Math.random() * 100),
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                tension: 0.4
            }, {
                label: 'Benchmark',
                data: labels.map(() => Math.random() * 100),
                borderColor: '#4ade80',
                backgroundColor: 'rgba(74, 222, 128, 0.1)',
                tension: 0.4
            }]
        };
    }

    updateMetrics(metrics) {
        // Update metric cards
        document.querySelectorAll('.metric-card').forEach((card, index) => {
            const valueElement = card.querySelector('.metric-value');
            const changeElement = card.querySelector('.metric-change');
            
            if (index === 0) { // Research Items
                valueElement.textContent = metrics.researchCount;
                changeElement.textContent = `+${metrics.newThisWeek} this week`;
            } else if (index === 1) { // Active Signals
                valueElement.textContent = metrics.activeSignals;
                changeElement.textContent = `${metrics.signalAccuracy}% accuracy`;
            } else if (index === 2) { // Portfolio Return
                valueElement.textContent = `+${metrics.portfolioReturn}%`;
                changeElement.textContent = 'YTD Performance';
            } else if (index === 3) { // Risk Score
                valueElement.textContent = metrics.riskScore;
                changeElement.textContent = this.getRiskLabel(metrics.riskScore);
            }
        });
    }

    updateAIAnalysis(analysis) {
        const aiContent = document.getElementById('aiAnalysis');
        aiContent.textContent = analysis.summary;

        // Add typing animation
        this.typewriterEffect(aiContent, analysis.summary);
    }

    typewriterEffect(element, text) {
        element.textContent = '';
        let i = 0;
        
        const type = () => {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, 20);
            }
        };
        
        type();
    }

    searchResearch(query) {
        if (!query) {
            this.renderResearchCards(this.researchData);
            return;
        }

        const filtered = this.researchData.filter(item => {
            const searchStr = `${item.title} ${item.summary} ${item.tags.join(' ')}`.toLowerCase();
            return searchStr.includes(query.toLowerCase());
        });

        this.renderResearchCards(filtered);
    }

    toggleFilter(filter) {
        const chip = document.querySelector(`[data-filter="${filter}"]`);
        
        if (filter === 'all') {
            // Clear all filters except 'all'
            document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            this.filters.clear();
            this.filters.add('all');
        } else {
            // Toggle specific filter
            if (this.filters.has(filter)) {
                this.filters.delete(filter);
                chip.classList.remove('active');
            } else {
                this.filters.add(filter);
                chip.classList.add('active');
                // Remove 'all' when specific filter is selected
                this.filters.delete('all');
                document.querySelector('[data-filter="all"]').classList.remove('active');
            }
        }

        // Apply filters
        this.applyFilters();
    }

    applyFilters() {
        if (this.filters.has('all') || this.filters.size === 0) {
            this.renderResearchCards(this.researchData);
            return;
        }

        const filtered = this.researchData.filter(item => {
            return Array.from(this.filters).some(filter => 
                item.category === filter || item.tags.includes(filter)
            );
        });

        this.renderResearchCards(filtered);
    }

    navigateTo(page) {
        // Update active nav
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        document.querySelector(`[data-page="${page}"]`).classList.add('active');

        // Update page title
        const titles = {
            overview: 'AI Research Dashboard',
            research: 'Research Library',
            markets: 'Market Analysis',
            portfolio: 'Portfolio Insights',
            signals: 'Trading Signals',
            settings: 'Settings'
        };
        
        document.querySelector('.page-title').textContent = titles[page] || 'Dashboard';
        
        // Load page-specific content
        this.loadPageContent(page);
    }

    loadPageContent(page) {
        // In production, this would load different content for each page
        console.log(`Loading content for page: ${page}`);
        
        // For demo, just refresh the current data
        if (page === 'research') {
            this.loadResearchData();
        }
    }

    async refreshData() {
        const refreshBtn = document.getElementById('refreshBtn');
        refreshBtn.disabled = true;
        refreshBtn.innerHTML = '<span>Refreshing...</span>';

        await this.loadResearchData();

        refreshBtn.disabled = false;
        refreshBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M1 4V10H7M23 20V14H17M20.49 9C19.9828 7.56678 19.1209 6.28841 17.9845 5.27542C16.8482 4.26243 15.4745 3.54629 13.9917 3.18979C12.5089 2.83329 10.9652 2.84697 9.48891 3.22959C8.01259 3.61221 6.65227 4.35239 5.53 5.38M3.51 15C4.01719 16.4332 4.87907 17.7116 6.01547 18.7246C7.15187 19.7376 8.52547 20.4537 10.0083 20.8102C11.4911 21.1667 13.0348 21.153 14.5111 20.7704C15.9874 20.3878 17.3477 19.6476 18.47 18.62" 
                      stroke="currentColor" stroke-width="2"/>
            </svg>
            Refresh
        `;
    }

    showNewResearchModal() {
        // In production, this would show a form to create new research
        alert('New research creation - coming soon!');
    }

    closeModal() {
        const modal = document.getElementById('researchModal');
        modal.classList.remove('open');
    }

    applyResearch(researchId) {
        // In production, this would apply research recommendations to portfolio
        console.log(`Applying research ${researchId}`);
        alert('Research recommendations applied to portfolio!');
        this.closeModal();
    }

    shareResearch(researchId) {
        // In production, this would share research
        console.log(`Sharing research ${researchId}`);
        alert('Research sharing link copied to clipboard!');
    }

    getRiskLabel(score) {
        if (score < 3) return 'Low';
        if (score < 7) return 'Moderate';
        return 'High';
    }

    showLoading() {
        document.getElementById('loadingState').style.display = 'block';
        document.getElementById('researchGrid').style.display = 'none';
    }

    hideLoading() {
        document.getElementById('loadingState').style.display = 'none';
        document.getElementById('researchGrid').style.display = 'grid';
    }

    showError(message) {
        // In production, show proper error UI
        console.error(message);
    }

    startRealtimeUpdates() {
        // Update AI analysis every 30 seconds
        setInterval(() => {
            this.updateAIAnalysis({
                summary: this.generateAIInsight()
            });
        }, 30000);

        // Update metrics every minute
        setInterval(() => {
            this.updateMetrics(this.generateMockMetrics());
        }, 60000);
    }

    generateAIInsight() {
        const insights = [
            "Current market conditions show strong momentum in technology stocks, with particular strength in AI and semiconductor sectors.",
            "Risk indicators suggest increasing volatility in emerging markets. Consider defensive positioning in utilities and consumer staples.",
            "Technical analysis reveals bullish patterns forming in major indices. Key resistance levels approaching - monitor for breakout opportunities.",
            "Sentiment indicators are reaching extreme levels in crypto markets. Historical patterns suggest a cooling period may be imminent.",
            "Macro environment remains supportive for growth assets. Federal Reserve policy stance continues to favor risk-on positioning."
        ];
        
        return insights[Math.floor(Math.random() * insights.length)];
    }

    generateMockMetrics() {
        return {
            researchCount: Math.floor(Math.random() * 50) + 200,
            newThisWeek: Math.floor(Math.random() * 20) + 5,
            activeSignals: Math.floor(Math.random() * 10) + 15,
            signalAccuracy: Math.floor(Math.random() * 10) + 80,
            portfolioReturn: (Math.random() * 20 + 5).toFixed(1),
            riskScore: (Math.random() * 10).toFixed(1)
        };
    }

    async mockFetchResearch() {
        // Mock data for demonstration
        return {
            data: [
                {
                    id: '1',
                    title: 'AI Revolution in Financial Markets',
                    summary: 'Deep dive into how artificial intelligence is transforming trading strategies and market analysis.',
                    author: 'Dr. Sarah Chen',
                    date: '2024-01-15',
                    readTime: 8,
                    category: 'tech',
                    tags: ['AI', 'Technology', 'Trading'],
                    content: '<p>Detailed analysis of AI impact on markets...</p>',
                    recommendations: [
                        {
                            title: 'Increase AI/Tech Exposure',
                            description: 'Allocate 10-15% of portfolio to AI-focused ETFs'
                        }
                    ],
                    chartType: 'line',
                    chartData: this.generateMockChartData()
                },
                {
                    id: '2',
                    title: 'Emerging Markets Opportunity Analysis',
                    summary: 'Comprehensive review of investment opportunities in developing economies.',
                    author: 'Michael Rodriguez',
                    date: '2024-01-14',
                    readTime: 12,
                    category: 'markets',
                    tags: ['Emerging Markets', 'Global', 'Growth'],
                    content: '<p>Analysis of emerging market opportunities...</p>',
                    recommendations: [
                        {
                            title: 'Diversify into EM Bonds',
                            description: 'Consider 5-7% allocation to emerging market debt'
                        }
                    ]
                },
                {
                    id: '3',
                    title: 'Cryptocurrency Market Cycles',
                    summary: 'Technical analysis of Bitcoin and altcoin market patterns.',
                    author: 'Alex Thompson',
                    date: '2024-01-13',
                    readTime: 6,
                    category: 'crypto',
                    tags: ['Bitcoin', 'Crypto', 'Technical Analysis'],
                    content: '<p>Crypto market cycle analysis...</p>',
                    recommendations: [
                        {
                            title: 'DCA into Bitcoin',
                            description: 'Implement dollar-cost averaging strategy for BTC'
                        }
                    ]
                }
            ],
            metrics: this.generateMockMetrics(),
            aiAnalysis: {
                summary: this.generateAIInsight()
            }
        };
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new ResearchDashboard();
});