#!/usr/bin/env python3
"""
Command Line Interface for Portfolio Builder
Provides easy-to-use interface for building portfolios based on risk and investment amount
"""

import argparse
import logging
import json
import sys
from datetime import datetime
from portfolio_builder import (
    PortfolioBuilder, 
    PortfolioConstraints, 
    RiskLevel, 
    ProjectType,
    ProjectProfile
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_strategy_data(filename: str) -> dict:
    """Load strategy data from JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File {filename} not found")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file {filename}")
        return {}

def create_portfolio_from_strategies(strategy_file: str, capital: float, risk_level: str, 
                                   output_file: str = None, detailed: bool = False):
    """Create portfolio from strategy results file"""
    
    # Initialize portfolio builder
    builder = PortfolioBuilder()
    
    # Load strategies
    logger.info(f"Loading strategies from {strategy_file}...")
    builder.load_projects_from_strategies(strategy_file)
    
    if not builder.projects:
        logger.error("No projects loaded. Check the strategy file.")
        return
    
    # Create constraints
    try:
        risk_enum = RiskLevel(risk_level.lower())
    except ValueError:
        logger.error(f"Invalid risk level: {risk_level}. Valid options: conservative, moderate, aggressive, very_aggressive")
        return
    
    constraints = PortfolioConstraints(
        total_capital=capital,
        risk_level=risk_enum
    )
    
    # Optimize portfolio
    logger.info(f"Optimizing portfolio for {capital:,.0f} ₽ with {risk_level} risk level...")
    result = builder.optimize_portfolio(constraints, 'utility')
    
    if result.get('success', False):
        # Generate report
        report = builder.generate_allocation_report(result)
        
        if detailed:
            print(report)
        else:
            # Print summary
            metrics = result['portfolio_metrics']
            allocation = result['allocation_plan']
            
            print(f"\n🎯 ПОРТФЕЛЬ ОПТИМИЗИРОВАН УСПЕШНО")
            print(f"{'='*60}")
            print(f"Капитал: {capital:,.0f} ₽")
            print(f"Уровень риска: {risk_level.upper()}")
            print(f"Ожидаемая доходность: {metrics['expected_return']:.2%} годовых")
            print(f"Волатильность: {metrics['volatility']:.2%} годовых")
            print(f"Коэффициент Шарпа: {metrics['sharpe_ratio']:.3f}")
            print(f"Максимальная просадка: {metrics['max_drawdown']:.2%}")
            print(f"Количество проектов: {len(allocation)}")
            
            print(f"\n📊 РАСПРЕДЕЛЕНИЕ КАПИТАЛА:")
            print("-" * 60)
            
            sorted_allocations = sorted(
                allocation.items(),
                key=lambda x: x[1]['amount'],
                reverse=True
            )
            
            for i, (name, alloc) in enumerate(sorted_allocations, 1):
                print(f"{i:2d}. {name}")
                print(f"    Сумма: {alloc['amount']:>10,.0f} ₽ ({alloc['weight']:.1%})")
                print(f"    Тип: {alloc['project_type']}")
        
        # Save results
        if output_file:
            builder.save_portfolio_analysis(result, output_file)
            print(f"\n💾 Результаты сохранены в {output_file}")
        
        # Save detailed report
        report_filename = f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Подробный отчет сохранен в {report_filename}")
        
    else:
        print(f"❌ Ошибка оптимизации: {result.get('error', 'Unknown error')}")

def create_sample_portfolio(capital: float, risk_level: str, output_file: str = None, detailed: bool = False):
    """Create portfolio with sample projects"""
    
    from test_portfolio_builder import create_sample_projects
    
    # Initialize portfolio builder
    builder = PortfolioBuilder()
    
    # Add sample projects
    sample_projects = create_sample_projects()
    for project in sample_projects:
        builder.add_project(project)
    
    logger.info(f"Added {len(sample_projects)} sample projects")
    
    # Create constraints
    try:
        risk_enum = RiskLevel(risk_level.lower())
    except ValueError:
        logger.error(f"Invalid risk level: {risk_level}. Valid options: conservative, moderate, aggressive, very_aggressive")
        return
    
    constraints = PortfolioConstraints(
        total_capital=capital,
        risk_level=risk_enum
    )
    
    # Optimize portfolio
    logger.info(f"Optimizing sample portfolio for {capital:,.0f} ₽ with {risk_level} risk level...")
    result = builder.optimize_portfolio(constraints, 'utility')
    
    if result.get('success', False):
        # Generate report
        report = builder.generate_allocation_report(result)
        
        if detailed:
            print(report)
        else:
            # Print summary
            metrics = result['portfolio_metrics']
            allocation = result['allocation_plan']
            
            print(f"\n🎯 САМПЛ-ПОРТФЕЛЬ ОПТИМИЗИРОВАН УСПЕШНО")
            print(f"{'='*60}")
            print(f"Капитал: {capital:,.0f} ₽")
            print(f"Уровень риска: {risk_level.upper()}")
            print(f"Ожидаемая доходность: {metrics['expected_return']:.2%} годовых")
            print(f"Волатильность: {metrics['volatility']:.2%} годовых")
            print(f"Коэффициент Шарпа: {metrics['sharpe_ratio']:.3f}")
            print(f"Максимальная просадка: {metrics['max_drawdown']:.2%}")
            print(f"Количество проектов: {len(allocation)}")
            
            print(f"\n📊 РАСПРЕДЕЛЕНИЕ КАПИТАЛА:")
            print("-" * 60)
            
            sorted_allocations = sorted(
                allocation.items(),
                key=lambda x: x[1]['amount'],
                reverse=True
            )
            
            for i, (name, alloc) in enumerate(sorted_allocations, 1):
                print(f"{i:2d}. {name}")
                print(f"    Сумма: {alloc['amount']:>10,.0f} ₽ ({alloc['weight']:.1%})")
                print(f"    Тип: {alloc['project_type']}")
        
        # Save results
        if output_file:
            builder.save_portfolio_analysis(result, output_file)
            print(f"\n💾 Результаты сохранены в {output_file}")
        
        # Save detailed report
        report_filename = f"sample_portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Подробный отчет сохранен в {report_filename}")
        
    else:
        print(f"❌ Ошибка оптимизации: {result.get('error', 'Unknown error')}")

def compare_risk_levels(strategy_file: str, capital: float, output_file: str = None):
    """Compare portfolios across all risk levels"""
    
    # Initialize portfolio builder
    builder = PortfolioBuilder()
    
    # Load strategies
    if strategy_file:
        logger.info(f"Loading strategies from {strategy_file}...")
        builder.load_projects_from_strategies(strategy_file)
    else:
        # Use sample projects
        from test_portfolio_builder import create_sample_projects
        sample_projects = create_sample_projects()
        for project in sample_projects:
            builder.add_project(project)
        logger.info("Using sample projects for comparison")
    
    if not builder.projects:
        logger.error("No projects available for comparison")
        return
    
    print(f"\n{'='*100}")
    print(f"СРАВНЕНИЕ ПОРТФЕЛЕЙ ПО УРОВНЯМ РИСКА (Капитал: {capital:,.0f} ₽)")
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
    print(f"\n{'Уровень риска':<20} {'Доходность':<12} {'Волатильность':<15} {'Шарп':<8} {'Просадка':<12} {'Проектов':<10}")
    print("-" * 90)
    
    for data in comparison_data:
        print(f"{data['risk_level']:<20} "
              f"{data['expected_return']:<12.2%} "
              f"{data['volatility']:<15.2%} "
              f"{data['sharpe_ratio']:<8.3f} "
              f"{data['max_drawdown']:<12.2%} "
              f"{data['project_count']:<10}")
    
    # Save comparison
    if output_file:
        comparison_result = {
            'capital': capital,
            'comparison_data': comparison_data,
            'analysis_date': datetime.now().isoformat()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Результаты сравнения сохранены в {output_file}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='Portfolio Builder CLI - Build portfolios based on risk and investment amount',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create portfolio from strategy results
  python portfolio_cli.py build --strategy-file advanced_strategy_results.json --capital 1000000 --risk moderate
  
  # Create sample portfolio
  python portfolio_cli.py sample --capital 500000 --risk aggressive --detailed
  
  # Compare all risk levels
  python portfolio_cli.py compare --capital 1000000 --strategy-file advanced_strategy_results.json
  
  # Quick sample comparison
  python portfolio_cli.py compare --capital 500000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build portfolio from strategy results')
    build_parser.add_argument('--strategy-file', required=True, help='JSON file with strategy results')
    build_parser.add_argument('--capital', type=float, required=True, help='Investment capital amount')
    build_parser.add_argument('--risk', required=True, 
                             choices=['conservative', 'moderate', 'aggressive', 'very_aggressive'],
                             help='Risk tolerance level')
    build_parser.add_argument('--output', help='Output file for results')
    build_parser.add_argument('--detailed', action='store_true', help='Show detailed report')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Build portfolio with sample projects')
    sample_parser.add_argument('--capital', type=float, required=True, help='Investment capital amount')
    sample_parser.add_argument('--risk', required=True,
                              choices=['conservative', 'moderate', 'aggressive', 'very_aggressive'],
                              help='Risk tolerance level')
    sample_parser.add_argument('--output', help='Output file for results')
    sample_parser.add_argument('--detailed', action='store_true', help='Show detailed report')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare portfolios across risk levels')
    compare_parser.add_argument('--capital', type=float, required=True, help='Investment capital amount')
    compare_parser.add_argument('--strategy-file', help='JSON file with strategy results (optional)')
    compare_parser.add_argument('--output', help='Output file for comparison results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'build':
            create_portfolio_from_strategies(
                args.strategy_file, 
                args.capital, 
                args.risk, 
                args.output, 
                args.detailed
            )
        elif args.command == 'sample':
            create_sample_portfolio(
                args.capital, 
                args.risk, 
                args.output, 
                args.detailed
            )
        elif args.command == 'compare':
            compare_risk_levels(
                args.strategy_file, 
                args.capital, 
                args.output
            )
    
    except KeyboardInterrupt:
        print("\n\nОперация прервана пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

