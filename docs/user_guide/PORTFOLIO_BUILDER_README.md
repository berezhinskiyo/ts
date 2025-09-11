# Portfolio Builder Module

## Overview

The Portfolio Builder module is a comprehensive tool for constructing portfolios of trading strategies/projects based on risk tolerance and investment volume. It uses advanced optimization algorithms to create optimal allocations that balance return and risk according to your preferences.

## Features

### 🎯 Risk-Based Portfolio Construction
- **4 Risk Levels**: Conservative, Moderate, Aggressive, Very Aggressive
- **Custom Constraints**: Set your own risk parameters and investment limits
- **Multiple Objectives**: Optimize for utility, Sharpe ratio, Sortino ratio, or risk parity

### 📊 Advanced Risk Metrics
- **Value at Risk (VaR)**: 95% confidence level risk measurement
- **Conditional VaR (CVaR)**: Expected loss beyond VaR
- **Diversification Ratio**: Measure of portfolio diversification effectiveness
- **Correlation Analysis**: Automatic correlation matrix calculation

### 💼 Project Type Classification
- **Arbitrage**: Spread trading, pairs trading
- **Momentum**: Trend following, breakout strategies
- **Mean Reversion**: Oscillator-based strategies
- **ML Predictive**: Machine learning price prediction
- **Technical**: Traditional technical analysis
- **Hedge**: Risk hedging strategies
- **Scalping**: High-frequency trading

### 🔧 Flexible Configuration
- **Investment Limits**: Set minimum and maximum allocations per project
- **Liquidity Requirements**: Filter projects by liquidity score
- **Complexity Limits**: Control portfolio complexity
- **Custom Constraints**: Target returns, volatility limits, drawdown limits

## Installation

1. Ensure you have the required dependencies:
```bash
pip install -r requirements.txt
```

2. The module requires `scipy` for optimization algorithms:
```bash
pip install scipy==1.11.1
```

## Quick Start

### 1. Using the CLI (Recommended)

#### Create a portfolio from strategy results:
```bash
python portfolio_cli.py build --strategy-file advanced_strategy_results.json --capital 1000000 --risk moderate
```

#### Create a sample portfolio:
```bash
python portfolio_cli.py sample --capital 500000 --risk aggressive --detailed
```

#### Compare all risk levels:
```bash
python portfolio_cli.py compare --capital 1000000 --strategy-file advanced_strategy_results.json
```

### 2. Using Python Code

```python
from portfolio_builder import PortfolioBuilder, PortfolioConstraints, RiskLevel

# Initialize portfolio builder
builder = PortfolioBuilder()

# Load projects from strategy results
builder.load_projects_from_strategies('advanced_strategy_results.json')

# Create constraints
constraints = PortfolioConstraints(
    total_capital=1000000,  # 1M rubles
    risk_level=RiskLevel.MODERATE
)

# Optimize portfolio
result = builder.optimize_portfolio(constraints, 'utility')

# Generate report
if result.get('success', False):
    report = builder.generate_allocation_report(result)
    print(report)
```

## Risk Levels

### Conservative
- **Target Return**: 8% annually
- **Max Volatility**: 15%
- **Max Drawdown**: 10%
- **Best for**: Capital preservation, low-risk investors

### Moderate
- **Target Return**: 12% annually
- **Max Volatility**: 25%
- **Max Drawdown**: 20%
- **Best for**: Balanced growth, typical investors

### Aggressive
- **Target Return**: 18% annually
- **Max Volatility**: 40%
- **Max Drawdown**: 30%
- **Best for**: Growth-oriented investors

### Very Aggressive
- **Target Return**: 25% annually
- **Max Volatility**: 60%
- **Max Drawdown**: 45%
- **Best for**: High-risk tolerance, experienced traders

## Project Types and Characteristics

### Arbitrage Strategies
- **Risk Level**: Low to Medium
- **Expected Return**: 8-15%
- **Volatility**: 5-15%
- **Correlation**: Low with market
- **Best for**: Conservative portfolios

### Momentum Strategies
- **Risk Level**: Medium to High
- **Expected Return**: 15-30%
- **Volatility**: 20-40%
- **Correlation**: High with market
- **Best for**: Growth portfolios

### ML Predictive Strategies
- **Risk Level**: High
- **Expected Return**: 20-40%
- **Volatility**: 25-50%
- **Correlation**: Medium with market
- **Best for**: Aggressive portfolios

### Technical Strategies
- **Risk Level**: Medium
- **Expected Return**: 10-25%
- **Volatility**: 15-30%
- **Correlation**: Medium with market
- **Best for**: Balanced portfolios

## Optimization Objectives

### Utility Function (Default)
Maximizes expected utility considering risk aversion:
```
Utility = Expected Return - 0.5 × Risk Aversion × Volatility²
```

### Sharpe Ratio
Maximizes risk-adjusted returns:
```
Sharpe = (Return - Risk Free Rate) / Volatility
```

### Sortino Ratio
Maximizes downside risk-adjusted returns:
```
Sortino = (Return - Risk Free Rate) / Downside Deviation
```

### Risk Parity
Equalizes risk contribution from each project.

## Output Files

The module generates several output files:

### 1. Portfolio Analysis JSON
Contains detailed optimization results:
```json
{
  "optimization_result": {
    "success": true,
    "allocation_plan": {...},
    "portfolio_metrics": {...},
    "constraints": {...}
  },
  "projects_data": {...},
  "analysis_date": "2024-01-15T10:30:00"
}
```

### 2. Detailed Report TXT
Human-readable comprehensive report with:
- Portfolio performance metrics
- Capital allocation breakdown
- Risk analysis by project type
- Growth projections
- Management recommendations

### 3. Comparison Results
When comparing risk levels:
- Side-by-side performance comparison
- Risk-return trade-off analysis
- Project count comparison

## Advanced Usage

### Custom Project Definition

```python
from portfolio_builder import ProjectProfile, ProjectType

# Create custom project
custom_project = ProjectProfile(
    name="My_Custom_Strategy",
    project_type=ProjectType.TECHNICAL,
    expected_return=0.20,  # 20% annual
    volatility=0.25,       # 25% annual volatility
    max_drawdown=0.15,     # 15% max drawdown
    liquidity_score=0.8,   # 80% liquidity
    complexity_score=0.6,  # 60% complexity
    success_rate=0.7       # 70% success rate
)

builder.add_project(custom_project)
```

### Custom Constraints

```python
constraints = PortfolioConstraints(
    total_capital=2000000,
    risk_level=RiskLevel.MODERATE,
    max_projects=8,
    min_projects=4,
    max_single_allocation=0.30,  # 30% max per project
    min_single_allocation=0.05,  # 5% min per project
    target_return=0.15,          # 15% target return
    max_volatility=0.25,         # 25% max volatility
    liquidity_requirement=0.8,   # 80% minimum liquidity
    complexity_limit=0.7         # 70% maximum complexity
)
```

### Multiple Optimization Runs

```python
# Test different objectives
objectives = ['utility', 'sharpe', 'sortino', 'risk_parity']
results = {}

for objective in objectives:
    result = builder.optimize_portfolio(constraints, objective)
    results[objective] = result

# Compare results
for obj, res in results.items():
    if res.get('success', False):
        metrics = res['portfolio_metrics']
        print(f"{obj}: Return={metrics['expected_return']:.2%}, Sharpe={metrics['sharpe_ratio']:.3f}")
```

## Risk Management Features

### 1. Position Sizing
- Automatic calculation based on risk per trade
- Portfolio-level risk limits
- Correlation-based position adjustments

### 2. Diversification
- Minimum number of projects required
- Maximum allocation per project
- Correlation matrix analysis
- Project type diversification

### 3. Risk Monitoring
- Real-time drawdown tracking
- VaR and CVaR calculations
- Volatility monitoring
- Performance attribution

## Best Practices

### 1. Portfolio Construction
- Start with conservative risk levels
- Gradually increase risk as you gain experience
- Diversify across different project types
- Consider market conditions when selecting strategies

### 2. Risk Management
- Set appropriate stop-loss levels
- Monitor correlation between projects
- Rebalance portfolio regularly
- Keep track of performance metrics

### 3. Implementation
- Test strategies on paper first
- Start with small capital allocations
- Monitor and adjust based on performance
- Document all decisions and rationale

## Troubleshooting

### Common Issues

1. **"Not enough projects available"**
   - Solution: Reduce minimum project requirements or add more strategies

2. **"Optimization failed"**
   - Solution: Relax constraints or check project data quality

3. **"High correlation detected"**
   - Solution: Add more diverse project types or reduce allocation limits

### Performance Tips

1. **Large Portfolios**: Use `max_projects` limit to avoid over-diversification
2. **Complex Strategies**: Set appropriate `complexity_limit` to manage operational risk
3. **Liquidity**: Ensure `liquidity_requirement` matches your trading frequency

## Integration with Existing System

The Portfolio Builder integrates seamlessly with the existing trading system:

1. **Strategy Results**: Loads from `advanced_strategy_results.json`
2. **Risk Manager**: Uses existing risk management parameters
3. **Configuration**: Respects settings from `config.py`
4. **Reporting**: Generates compatible output formats

## Examples

### Example 1: Conservative Portfolio for 500K Rubles

```bash
python portfolio_cli.py build --strategy-file advanced_strategy_results.json --capital 500000 --risk conservative --detailed
```

Expected output:
- 4-6 projects
- 8-12% expected return
- 10-15% volatility
- Low correlation projects

### Example 2: Aggressive Portfolio for 2M Rubles

```bash
python portfolio_cli.py build --strategy-file advanced_strategy_results.json --capital 2000000 --risk aggressive --detailed
```

Expected output:
- 6-8 projects
- 18-25% expected return
- 30-40% volatility
- Mix of momentum and ML strategies

### Example 3: Risk Level Comparison

```bash
python portfolio_cli.py compare --capital 1000000 --strategy-file advanced_strategy_results.json
```

This will show you the trade-offs between different risk levels for the same capital amount.

## Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the example outputs in the test files
3. Examine the detailed error messages in the logs
4. Test with sample data first using the `sample` command

## Future Enhancements

Planned features for future versions:

1. **Real-time Optimization**: Dynamic portfolio rebalancing
2. **Machine Learning**: AI-powered project selection
3. **Market Regime Detection**: Adaptive risk management
4. **Multi-Asset Support**: Bonds, commodities, alternatives
5. **Tax Optimization**: Tax-efficient portfolio construction
6. **ESG Integration**: Environmental, social, governance factors

---

**Note**: This module is designed for educational and research purposes. Always perform thorough testing before using with real capital. Past performance does not guarantee future results.

