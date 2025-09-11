#!/usr/bin/env python3
"""
Quality Monitor for Trading Strategies
Мониторинг качества и точности торговых стратегий
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QualityMonitor:
    """Мониторинг качества торговых стратегий"""
    
    def __init__(self, config_path: str = "config/parameters/alerts_config.py"):
        self.config = self._load_config(config_path)
        self.alert_history = []
        self.metrics_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Загрузка конфигурации"""
        try:
            # В реальном проекте здесь будет загрузка из файла
            return {
                'email': {
                    'enabled': True,
                    'recipients': ['trader@example.com'],
                    'thresholds': {
                        'low_return': 0.05,
                        'high_drawdown': 0.15,
                        'low_sharpe': 1.0,
                        'low_ml_precision': 0.6
                    }
                },
                'telegram': {
                    'enabled': False,
                    'bot_token': 'YOUR_BOT_TOKEN',
                    'chat_id': 'YOUR_CHAT_ID'
                }
            }
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def check_strategy_quality(self, strategy_name: str, metrics: Dict) -> Dict:
        """Проверка качества стратегии"""
        logger.info(f"🔍 Checking quality for {strategy_name}")
        
        alerts = []
        quality_score = 0
        
        # Проверка доходности
        monthly_return = metrics.get('monthly_return', 0)
        if monthly_return < self.config['email']['thresholds']['low_return']:
            alert = {
                'type': 'LOW_RETURN',
                'message': f"Доходность {strategy_name}: {monthly_return:.2%}",
                'severity': 'WARNING',
                'timestamp': datetime.now()
            }
            alerts.append(alert)
            quality_score -= 20
        else:
            quality_score += 30
        
        # Проверка Sharpe ratio
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        if sharpe_ratio < self.config['email']['thresholds']['low_sharpe']:
            alert = {
                'type': 'LOW_SHARPE',
                'message': f"Sharpe ratio {strategy_name}: {sharpe_ratio:.3f}",
                'severity': 'WARNING',
                'timestamp': datetime.now()
            }
            alerts.append(alert)
            quality_score -= 15
        else:
            quality_score += 25
        
        # Проверка просадки
        max_drawdown = metrics.get('max_drawdown', 0)
        if max_drawdown > self.config['email']['thresholds']['high_drawdown']:
            alert = {
                'type': 'HIGH_DRAWDOWN',
                'message': f"Просадка {strategy_name}: {max_drawdown:.2%}",
                'severity': 'CRITICAL',
                'timestamp': datetime.now()
            }
            alerts.append(alert)
            quality_score -= 30
        else:
            quality_score += 20
        
        # Проверка точности ML (если применимо)
        if 'ml_precision' in metrics:
            ml_precision = metrics['ml_precision']
            if ml_precision < self.config['email']['thresholds']['low_ml_precision']:
                alert = {
                    'type': 'LOW_ML_PRECISION',
                    'message': f"Точность ML {strategy_name}: {ml_precision:.2%}",
                    'severity': 'WARNING',
                    'timestamp': datetime.now()
                }
                alerts.append(alert)
                quality_score -= 10
            else:
                quality_score += 15
        
        # Нормализация качества (0-100)
        quality_score = max(0, min(100, quality_score))
        
        # Отправка алертов
        for alert in alerts:
            self._send_alert(alert)
        
        # Сохранение метрик
        quality_data = {
            'strategy_name': strategy_name,
            'timestamp': datetime.now(),
            'quality_score': quality_score,
            'metrics': metrics,
            'alerts': alerts
        }
        
        self.metrics_history.append(quality_data)
        self._save_metrics(quality_data)
        
        return quality_data
    
    def _send_alert(self, alert: Dict):
        """Отправка уведомления"""
        logger.warning(f"🚨 ALERT: {alert['type']} - {alert['message']}")
        
        # Добавление в историю
        self.alert_history.append(alert)
        
        # Отправка email (если включено)
        if self.config.get('email', {}).get('enabled', False):
            self._send_email_alert(alert)
        
        # Отправка в Telegram (если включено)
        if self.config.get('telegram', {}).get('enabled', False):
            self._send_telegram_alert(alert)
    
    def _send_email_alert(self, alert: Dict):
        """Отправка email уведомления"""
        try:
            # В реальном проекте здесь будет отправка email
            logger.info(f"📧 Email alert sent: {alert['message']}")
        except Exception as e:
            logger.error(f"Error sending email: {e}")
    
    def _send_telegram_alert(self, alert: Dict):
        """Отправка Telegram уведомления"""
        try:
            # В реальном проекте здесь будет отправка в Telegram
            logger.info(f"📱 Telegram alert sent: {alert['message']}")
        except Exception as e:
            logger.error(f"Error sending Telegram: {e}")
    
    def _save_metrics(self, quality_data: Dict):
        """Сохранение метрик"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"monitoring/metrics/quality_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(quality_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"📊 Metrics saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def get_quality_report(self, days: int = 7) -> Dict:
        """Получение отчета о качестве за период"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m['timestamp'] > cutoff_date
        ]
        
        if not recent_metrics:
            return {'error': 'No data for the specified period'}
        
        # Анализ качества
        quality_scores = [m['quality_score'] for m in recent_metrics]
        avg_quality = np.mean(quality_scores)
        
        # Анализ алертов
        all_alerts = []
        for m in recent_metrics:
            all_alerts.extend(m['alerts'])
        
        alert_counts = {}
        for alert in all_alerts:
            alert_type = alert['type']
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        
        # Анализ трендов
        if len(quality_scores) > 1:
            quality_trend = 'improving' if quality_scores[-1] > quality_scores[0] else 'declining'
        else:
            quality_trend = 'stable'
        
        report = {
            'period_days': days,
            'total_checks': len(recent_metrics),
            'average_quality_score': avg_quality,
            'quality_trend': quality_trend,
            'alert_summary': alert_counts,
            'total_alerts': len(all_alerts),
            'strategies_checked': list(set(m['strategy_name'] for m in recent_metrics))
        }
        
        return report
    
    def check_all_strategies(self, strategies_data: Dict[str, Dict]) -> Dict:
        """Проверка качества всех стратегий"""
        logger.info("🔍 Checking quality for all strategies")
        
        results = {}
        overall_quality = 0
        
        for strategy_name, metrics in strategies_data.items():
            quality_data = self.check_strategy_quality(strategy_name, metrics)
            results[strategy_name] = quality_data
            overall_quality += quality_data['quality_score']
        
        # Общая оценка
        if results:
            overall_quality /= len(results)
        
        results['overall'] = {
            'average_quality_score': overall_quality,
            'total_strategies': len(strategies_data),
            'timestamp': datetime.now()
        }
        
        return results

def main():
    """Основная функция для тестирования"""
    monitor = QualityMonitor()
    
    # Тестовые данные стратегий
    test_strategies = {
        'ML_Strategy': {
            'monthly_return': 0.085,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.08,
            'ml_precision': 0.65
        },
        'Aggressive_Strategy': {
            'monthly_return': 0.152,
            'sharpe_ratio': 1.8,
            'max_drawdown': 0.12
        },
        'Combined_Strategy': {
            'monthly_return': 0.118,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.10
        }
    }
    
    # Проверка качества
    results = monitor.check_all_strategies(test_strategies)
    
    # Генерация отчета
    report = monitor.get_quality_report(days=1)
    
    print("📊 QUALITY MONITORING REPORT")
    print("="*50)
    print(f"Overall Quality Score: {results['overall']['average_quality_score']:.1f}/100")
    print(f"Total Strategies: {results['overall']['total_strategies']}")
    print(f"Total Alerts: {report['total_alerts']}")
    
    for strategy_name, data in results.items():
        if strategy_name != 'overall':
            print(f"\n{strategy_name}:")
            print(f"  Quality Score: {data['quality_score']}/100")
            print(f"  Alerts: {len(data['alerts'])}")
    
    # Сохранение отчета
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'monitoring/metrics/quality_report_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    logger.info("✅ Quality monitoring completed")

if __name__ == "__main__":
    main()

