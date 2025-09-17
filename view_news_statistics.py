#!/usr/bin/env python3
"""
Просмотр статистики новостей за 3 года
"""

import os
import json
import pandas as pd
from news_data_manager import NewsDataManager

def main():
    """Просмотр статистики новостей"""
    
    print("📊 СТАТИСТИКА НОВОСТЕЙ ЗА 3 ГОДА")
    print("=" * 60)
    
    # Создаем менеджер
    manager = NewsDataManager()
    
    # Загружаем метаданные
    metadata = manager.load_metadata()
    if metadata:
        print(f"📅 Период: {metadata['date_range']['start']} - {metadata['date_range']['end']}")
        print(f"📊 Символов: {metadata['total_symbols']}")
        print(f"📰 Всего новостей: {metadata['total_news']}")
        print(f"📁 Создано: {metadata['created_at']}")
        print()
    
    # Загружаем статистику
    stats = manager.get_news_statistics()
    if not stats:
        print("❌ Нет данных для отображения")
        return
    
    print("📈 СТАТИСТИКА ПО СИМВОЛАМ:")
    print("-" * 60)
    
    for symbol, symbol_stats in stats['symbols'].items():
        print(f"\n{symbol}:")
        print(f"  📰 Новостей: {symbol_stats['total_news']}")
        print(f"  📊 Средний сентимент: {symbol_stats['avg_sentiment']:.3f}")
        print(f"  🎯 Средняя уверенность: {symbol_stats['avg_confidence']:.3f}")
        
        print(f"  📈 Распределение по настроению:")
        for category, count in symbol_stats['sentiment_distribution'].items():
            percentage = (count / symbol_stats['total_news']) * 100
            print(f"    {category}: {count} ({percentage:.1f}%)")
        
        print(f"  🎯 Распределение по влиянию:")
        for impact, count in symbol_stats['impact_distribution'].items():
            percentage = (count / symbol_stats['total_news']) * 100
            print(f"    {impact}: {count} ({percentage:.1f}%)")
    
    # Общая статистика
    print(f"\n📊 ОБЩАЯ СТАТИСТИКА:")
    print("-" * 60)
    
    total_news = stats['total_news']
    total_symbols = stats['total_symbols']
    
    # Агрегированная статистика
    all_sentiments = []
    all_confidences = []
    all_categories = {'very_positive': 0, 'positive': 0, 'neutral': 0, 'negative': 0, 'very_negative': 0}
    all_impacts = {'very_high': 0, 'high': 0, 'medium': 0, 'low': 0}
    
    for symbol_stats in stats['symbols'].values():
        all_sentiments.append(symbol_stats['avg_sentiment'])
        all_confidences.append(symbol_stats['avg_confidence'])
        
        for category, count in symbol_stats['sentiment_distribution'].items():
            all_categories[category] += count
        
        for impact, count in symbol_stats['impact_distribution'].items():
            all_impacts[impact] += count
    
    print(f"📰 Всего новостей: {total_news}")
    print(f"📊 Символов: {total_symbols}")
    print(f"📈 Средний сентимент: {sum(all_sentiments) / len(all_sentiments):.3f}")
    print(f"🎯 Средняя уверенность: {sum(all_confidences) / len(all_confidences):.3f}")
    
    print(f"\n📈 Общее распределение по настроению:")
    for category, count in all_categories.items():
        percentage = (count / total_news) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")
    
    print(f"\n🎯 Общее распределение по влиянию:")
    for impact, count in all_impacts.items():
        percentage = (count / total_news) * 100
        print(f"  {impact}: {count} ({percentage:.1f}%)")
    
    # Показываем примеры новостей
    print(f"\n📰 ПРИМЕРЫ НОВОСТЕЙ:")
    print("-" * 60)
    
    # Загружаем данные новостей
    news_data = manager.load_news_data()
    if news_data:
        # Показываем по одной новости каждого типа для SBER
        if 'SBER' in news_data:
            sber_news = news_data['SBER']
            
            # Группируем по категориям
            categories = {}
            for news in sber_news:
                category = news.get('category', 'neutral')
                if category not in categories:
                    categories[category] = news
            
            for category, news in categories.items():
                print(f"\n{category.upper()}:")
                print(f"  📰 {news['title']}")
                print(f"  📝 {news['content']}")
                print(f"  📅 {news['published_at']}")
                print(f"  📊 Сентимент: {news['sentiment_score']:.2f}")
                print(f"  🎯 Уверенность: {news['confidence']:.2f}")
                print(f"  💥 Влияние: {news['impact']}")

if __name__ == "__main__":
    main()
