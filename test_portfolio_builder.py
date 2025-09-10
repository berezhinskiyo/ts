#!/usr/bin/env python3
"""
Test script for Portfolio Builder module
Demonstrates portfolio construction for different risk levels and investment amounts
"""

import logging
import json
from datetime import datetime
from portfolio_builder import (
    PortfolioBuilder, 
    ProjectProfile, 
    PortfolioConstraints, 
    RiskLevel, 
    ProjectType
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_projects():
    """Create sample projects for testing"""
    projects = [
        ProjectProfile(
            name="Arbitrage_Spread_Trading",
            project_type=ProjectType.ARBITRAGE,
            expected_return=0.15,  # 15% annual
            volatility=0.12,  # 12% annual volatility
            max_drawdown=0.08,
            liquidity_score=0.9,
            complexity_score=0.7,
            success_rate=0.75,
            min_investment=10000,
            max_investment=1000000
        ),
        ProjectProfile(
            name="Momentum_Breakout_Strategy",
            project_type=ProjectType.MOMENTUM,
            expected_return=0.25,
            volatility=0.28,
            max_drawdown=0.20,
            liquidity_score=0.8,
            complexity_score=0.6,
            success_rate=0.65,
            min_investment=10000,
            max_investment=1000000
        ),
        ProjectProfile(
            name="ML_Price_Prediction",
            project_type=ProjectType.ML_PREDICTIVE,
            expected_return=0.30,
            volatility=0.35,
            max_drawdown=0.25,
            liquidity_score=0.7,
            complexity_score=0.9,
            success_rate=0.70,
            min_investment=10000,
            max_investment=1000000
        ),
        ProjectProfile(
            name="RSI_Mean_Reversion",
            project_type=ProjectType.MEAN_REVERSION,
            expected_return=0.18,
            volatility=0.20,
            max_drawdown=0.15,
            liquidity_score=0.8,
            complexity_score=0.5,
            success_rate=0.68,
            min_investment=10000,
            max_investment=1000000
        ),
        ProjectProfile(
            name="Technical_MACD_Strategy",
            project_type=ProjectType.TECHNICAL,
            expected_return=0.20,
            volatility=0.22,
            max_drawdown=0.18,
            liquidity_score=0.8,
            complexity_score=0.4,
            success_rate=0.62,
            min_investment=10000,
            max_investment=1000000
        ),
        ProjectProfile(
            name="Hedge_Fund_Strategy",
            project_type=ProjectType.HEDGE,
            expected_return=0.12,
            volatility=0.10,
            max_drawdown=0.06,
            liquidity_score=0.6,
            complexity_score=0.8,
            success_rate=0.80,
            min_investment=10000,
            max_investment=1000000
        ),
        ProjectProfile(
            name="Scalping_HFT_Strategy",
            project_type=ProjectType.SCALPING,
            expected_return=0.35,
            volatility=0.45,
            max_drawdown=0.30,
            liquidity_score=0.9,
            complexity_score=0.9,
            success_rate=0.55,
            min_investment=10000,
            max_investment=1000000
        ),
        ProjectProfile(
            name="Bollinger_Bands_Strategy",
            project_type=ProjectType.TECHNICAL,
            expected_return=0.16,
            volatility=0.18,
            max_drawdown=0.12,
            liquidity_score=0.8,
            complexity_score=0.3,
            success_rate=0.64,
            min_investment=10000,
            max_investment=1000000
        )
    ]
    return projects

def test_portfolio_builder():
    """Test the portfolio builder with different scenarios"""
    
    # Initialize portfolio builder
    builder = PortfolioBuilder()
    
    # Add sample projects
    sample_projects = create_sample_projects()
    for project in sample_projects:
        builder.add_project(project)
    
    logger.info(f"Added {len(sample_projects)} sample projects")
    
    # Test different investment amounts
    investment_amounts = [100000, 500000, 1000000, 5000000]  # 100K to 5M rubles
    
    for capital in investment_amounts:
        print(f"\n{'='*100}")
        print(f"АНАЛИЗ ПОРТФЕЛЯ ДЛЯ КАПИТАЛА: {capital:,.0f} ₽")
        print(f"{'='*100}")
        
        # Create portfolios for different risk levels
        portfolios = builder.create_risk_based_portfolios(capital)
        
        for risk_level, result in portfolios.items():
            if result.get('success', False):
                print(f"\n📊 ПОРТФЕЛЬ ДЛЯ УРОВНЯ РИСКА: {risk_level.upper()}")
                print("-" * 80)
                
                metrics = result['portfolio_metrics']
                allocation = result['allocation_plan']
                
                print(f"Ожидаемая доходность: {metrics['expected_return']:.2%} годовых")
                print(f"Волатильность: {metrics['volatility']:.2%} годовых")
                print(f"Коэффициент Шарпа: {metrics['sharpe_ratio']:.3f}")
                print(f"Максимальная просадка: {metrics['max_drawdown']:.2%}")
                print(f"Количество проектов: {len(allocation)}")
                
                # Show top 3 allocations
                sorted_allocations = sorted(
                    allocation.items(),
                    key=lambda x: x[1]['amount'],
                    reverse=True
                )[:3]
                
                print("\nТоп-3 проекта по объему инвестиций:")
                for i, (name, alloc) in enumerate(sorted_allocations, 1):
                    print(f"  {i}. {name}: {alloc['amount']:,.0f} ₽ ({alloc['weight']:.1%})")
                
            else:
                print(f"\n❌ Ошибка для уровня риска {risk_level}: {result.get('error', 'Unknown error')}")

def test_custom_portfolio():
    """Test custom portfolio with specific constraints"""
    
    builder = PortfolioBuilder()
    sample_projects = create_sample_projects()
    for project in sample_projects:
        builder.add_project(project)
    
    # Create custom constraints for moderate risk with specific requirements
    constraints = PortfolioConstraints(
        total_capital=2000000,  # 2M rubles
        risk_level=RiskLevel.MODERATE,
        max_projects=6,
        min_projects=4,
        max_single_allocation=0.35,  # 35% max per project
        min_single_allocation=0.08,  # 8% min per project
        target_return=0.15,  # 15% target return
        max_volatility=0.25,  # 25% max volatility
        liquidity_requirement=0.75,  # 75% minimum liquidity
        complexity_limit=0.7  # 70% maximum complexity
    )
    
    print(f"\n{'='*100}")
    print("ТЕСТИРОВАНИЕ КАСТОМНОГО ПОРТФЕЛЯ")
    print(f"{'='*100}")
    
    # Optimize portfolio
    result = builder.optimize_portfolio(constraints, 'utility')
    
    if result.get('success', False):
        report = builder.generate_allocation_report(result)
        print(report)
        
        # Save the analysis
        builder.save_portfolio_analysis(result, 'custom_portfolio_analysis.json')
        
    else:
        print(f"Ошибка оптимизации: {result.get('error', 'Unknown error')}")

def test_risk_comparison():
    """Compare portfolios across different risk levels"""
    
    builder = PortfolioBuilder()
    sample_projects = create_sample_projects()
    for project in sample_projects:
        builder.add_project(project)
    
    capital = 1000000  # 1M rubles
    
    print(f"\n{'='*100}")
    print("СРАВНЕНИЕ ПОРТФЕЛЕЙ ПО УРОВНЯМ РИСКА")
    print(f"{'='*100}")
    
    comparison_data = []
    
    for risk_level in RiskLevel:
        constraints = PortfolioConstraints(
            total_capital=capital,
            risk_level=risk_level
        )
        
        result = builder.optimize_portfolio(constraints, 'utility')
        
        if result.get('success', False):
            metrics = result['portfolio_metrics']
            comparison_data.append({
                'risk_level': risk_level.value,
                'expected_return': metrics['expected_return'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'project_count': len(result['allocation_plan'])
            })
    
    # Display comparison table
    print(f"\n{'Уровень риска':<15} {'Доходность':<12} {'Волатильность':<15} {'Шарп':<8} {'Просадка':<12} {'Проектов':<10}")
    print("-" * 80)
    
    for data in comparison_data:
        print(f"{data['risk_level']:<15} "
              f"{data['expected_return']:<12.2%} "
              f"{data['volatility']:<15.2%} "
              f"{data['sharpe_ratio']:<8.3f} "
              f"{data['max_drawdown']:<12.2%} "
              f"{data['project_count']:<10}")

def main():
    """Main test function"""
    logger.info("Starting Portfolio Builder tests...")
    
    try:
        # Test 1: Basic portfolio building
        test_portfolio_builder()
        
        # Test 2: Custom portfolio with specific constraints
        test_custom_portfolio()
        
        # Test 3: Risk level comparison
        test_risk_comparison()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
